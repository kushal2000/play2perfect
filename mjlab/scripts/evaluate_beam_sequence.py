"""Run two frozen public Torch actors with continuous multipart handoff."""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from play2perfect_mjlab.assets import sha256
from play2perfect_mjlab.native import NativeAssembly
from play2perfect_mjlab.runtime_info import runtime_info
from play2perfect_mjlab.torch_actor import PretrainedActor

FIELDS = ("qpos", "qvel", "ctrl", "mocap_pos", "mocap_quat", "eq_active")


def native_contacts(model, data):
    roots = []
    for body in range(model.nbody):
        root = body
        while model.body_parentid[root] > 0:
            root = int(model.body_parentid[root])
        roots.append(root)
    robot = model.body("base").id
    objects = {model.body(name).id for name in ["object", "second"]}
    return sum(
        contact.dist <= 0
        and robot in {roots[int(model.geom_bodyid[g])] for g in contact.geom}
        and bool(objects & {roots[int(model.geom_bodyid[g])] for g in contact.geom})
        for contact in data.contact
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--stage2-checkpoint", type=Path, required=True)
    parser.add_argument("--conditioning", type=int, choices=[0, 50], required=True)
    parser.add_argument("--backend", choices=["native", "mjlab"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-measured-collider-difference", action="store_true")
    parser.add_argument("--record-every", type=int, default=20)
    parser.add_argument("--steps", type=int, default=6500)
    parser.add_argument("--max-seconds", type=float, default=1200)
    args = parser.parse_args()
    if min(args.record_every, args.steps, args.max_seconds) <= 0:
        parser.error("Recording interval and evaluation budgets must be positive")
    torch.set_num_threads(1)
    proof_path = args.bundle / "composition_proof.json"
    proof = json.loads(proof_path.read_text())
    if (
        not proof["seated_geometry_passed"]
        and not args.allow_measured_collider_difference
    ):
        parser.error("The fused fixture differs; explicit diagnostic scope is required")
    assert proof["composed_manifest_sha256"] == sha256(args.bundle / "manifest.json")
    assert proof["scene_sha256"] == sha256(args.bundle / "scene.xml")
    assert not args.output.exists(), "Preserve existing outcomes"
    args.output.mkdir(parents=True)
    actors = [
        PretrainedActor.from_checkpoint(path, args.conditioning).eval()
        for path in [args.stage1_checkpoint, args.stage2_checkpoint]
    ]
    rows, trace, handoffs = [], [], []
    env = None
    error, monitor = None, None
    h, c = actors[0].initial_state(1)
    stage = 0
    initial = None
    last_failed = False
    source = {}
    if args.backend == "native":
        env = NativeAssembly(args.bundle, sdf_initpoints=50)
        env.base_metadata["block_id"] = args.conditioning
        env.reset()
        env.model.opt.iterations = 100
        env.model.opt.ls_iterations = 50
        env.model.opt.timestep = 1 / 120
        env.model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
        original = env._handoff

        def checked_handoff():
            before = {
                name: getattr(env.data, name).copy()
                for name in ["qpos", "qvel", "ctrl"]
            }
            targets = env.targets.copy()
            original()
            assert all(
                np.array_equal(before[name], getattr(env.data, name)) for name in before
            )
            assert np.array_equal(targets, env.targets)
            handoffs.append(
                dict(control=env.steps, qpos_qvel_ctrl_targets_bit_exact=True)
            )

        env._handoff = checked_handoff
        model = env.model
        source["native"] = sha256(
            Path(__file__).resolve().parents[1] / "src/play2perfect_mjlab/native.py"
        )
    else:
        from mjlab.envs import ManagerBasedRlEnv

        from play2perfect_mjlab.assembly_env_cfg import assembly_cfg, task_contract
        from play2perfect_mjlab.assembly_mdp import geometry
        from play2perfect_mjlab.collision_capacity import configure_epa_horizon
        from play2perfect_mjlab.fabrica import handoff
        from play2perfect_mjlab.source_sapg import PhysicsRolloutMonitor

        configure_epa_horizon(64)
        cfg = assembly_cfg(
            args.bundle,
            1,
            auto_reset=False,
            sdf_initpoints=50,
            restore_source_static_filters=True,
            nconmax=512,
            njmax=2048,
            physics_dt=1 / 120,
            decimation=2,
            solver_iterations=100,
            ls_iterations=50,
        )
        cfg.p2p_metadata["block_id"] = args.conditioning
        env = ManagerBasedRlEnv(cfg, device="cpu")
        obs, _ = env.reset(seed=0)
        # Exercise the exact compiled handoff before playback, then reset.
        before = {
            name: getattr(env.sim.data, name).clone()
            for name in ["qpos", "qvel", "ctrl"]
        }
        handoff(env, env._p2p, torch.ones(1, dtype=torch.bool))
        assert all(
            torch.equal(before[name], getattr(env.sim.data, name)) for name in before
        )
        handoff_preflight = dict(
            qpos_qvel_ctrl_bit_exact=True,
            recurrent_reset_requested=bool(env._p2p["recurrent_reset"][0]),
        )
        assert handoff_preflight["recurrent_reset_requested"]
        obs, _ = env.reset(seed=0)
        monitor = PhysicsRolloutMonitor(env.sim)
        model = env.sim.mj_model
        source["task_contract"] = task_contract(cfg)
        source["compiled_handoff_preflight"] = handoff_preflight
        source["fabrica_sha256"] = sha256(
            Path(__file__).resolve().parents[1] / "src/play2perfect_mjlab/fabrica.py"
        )

    def state():
        if args.backend == "native":
            _, _, obj, goal = env.geometry(env.metadata["reward_offsets"])
            return dict(
                control=env.steps,
                stage=env.stage_index,
                goals=env.successes,
                distance=float(np.linalg.norm(obj - goal, axis=1).max()),
                reward_distance=env.distance,
                completed=bool(env.completed),
                failed=bool(env.failed),
                contacts=native_contacts(model, env.data),
            )
        s = env._p2p
        wd = env.sim.wp_data
        count = int(wd.nacon.numpy()[0])
        pairs = wd.contact.geom.numpy()[:count]
        distances = wd.contact.dist.numpy()[:count]
        robot = np.array(
            [
                (model.geom(i).name or "").startswith("robot/")
                for i in range(model.ngeom)
            ]
        )
        objects = np.array(
            [
                (model.geom(i).name or "").startswith(("object/", "second/"))
                for i in range(model.ngeom)
            ]
        )
        selected = (
            (robot[pairs[:, 0]] & objects[pairs[:, 1]])
            | (objects[pairs[:, 0]] & robot[pairs[:, 1]])
        ) & (distances <= 0)
        _, obj, goal = geometry(env, s["reward_offsets"])
        return dict(
            control=env.common_step_counter,
            stage=int(s["stage_index"][0]),
            goals=int(s["goal_index"][0]),
            distance=float(torch.linalg.vector_norm(obj - goal, dim=-1).max()),
            reward_distance=float(s["distance"][0]),
            completed=bool(s["completed"][0]),
            failed=last_failed,
            contacts=int(selected.sum()),
        )

    def capture(current, terminal=False):
        data = env.data if args.backend == "native" else env.sim.data
        fields = {
            name: (
                getattr(data, name).copy()[None]
                if args.backend == "native"
                else getattr(data, name).numpy().copy()
            )
            for name in FIELDS
        }
        rows.append(
            dict(
                control=np.array(current["control"]),
                active_worlds=np.array([True]),
                terminal_worlds=np.array([terminal]),
                goal_index=np.array([current["goals"]]),
                stage_index=np.array([current["stage"]]),
                handoff_step=np.array([handoffs[0]["control"] if handoffs else -1]),
                distance_m=np.array([current["distance"]]),
                reward_target_distance_m=np.array([current["reward_distance"] or 0.0]),
                robot_object_penetrating_contacts=np.array([current["contacts"]]),
                **fields,
            )
        )

    started = time.monotonic()
    current = state()
    capture(current)
    initial = {name: rows[0][name].tolist() for name in ["qpos", "qvel", "ctrl"]}
    initial_observation = (
        torch.from_numpy(env.observation())[None]
        if args.backend == "native"
        else obs["actor"]
    )
    with torch.inference_mode():
        initial_mean = actors[0](initial_observation, h, c)[0].numpy().tolist()
    initial_observation = initial_observation.numpy().tolist()
    try:
        with torch.inference_mode():
            for _ in range(args.steps):
                observation = (
                    torch.from_numpy(env.observation())[None]
                    if args.backend == "native"
                    else obs["actor"]
                )
                action, h, c = actors[stage](observation, h, c)
                if args.backend == "native":
                    env.step(action[0].numpy())
                else:
                    obs, _, terminated, truncated, _ = env.step(action)
                    monitor.validate()
                    last_failed = bool((terminated | truncated)[0]) and not bool(
                        env._p2p["completed"][0]
                    )
                current = state()
                changed = current["stage"] != stage
                if changed:
                    assert stage == 0 and current["stage"] == 1
                    h.zero_()
                    c.zero_()
                    stage = 1
                    if args.backend == "mjlab":
                        assert bool(env._p2p["recurrent_reset"][0])
                        handoffs.append(
                            dict(control=current["control"], recurrent_h_c_reset=True)
                        )
                trace.append(current)
                data = env.data if args.backend == "native" else env.sim.data
                for field in ["qpos", "qvel", "ctrl"]:
                    value = getattr(data, field)
                    assert (
                        bool(np.isfinite(value).all())
                        if args.backend == "native"
                        else bool(torch.isfinite(value).all())
                    )
                ended = current["completed"] or current["failed"]
                if args.backend == "mjlab":
                    ended |= bool((terminated | truncated)[0])
                if current["control"] % args.record_every == 0 or changed or ended:
                    capture(current, ended)
                if ended or time.monotonic() - started >= args.max_seconds:
                    break
    except BaseException as caught:
        error = repr(caught)
    finally:
        model_path = args.output / "poses.mjb"
        mujoco.mj_saveModel(model, str(model_path))
        if rows[-1]["control"] != current["control"]:
            capture(current)
        np.savez_compressed(
            args.output / "poses.npz",
            **{key: np.stack([row[key] for row in rows]) for key in rows[0]},
        )
        (args.output / "poses.json").write_text(
            json.dumps(
                dict(
                    scope="Continuous public-Torch canonical playback; "
                    "exact recorded poses",
                    model_layout="source_native"
                    if args.backend == "native"
                    else "mjlab_compiled",
                    frames=len(rows),
                    num_envs=1,
                    bundle=str(args.bundle),
                    sdf_initpoints=50,
                    record_every_controls=args.record_every,
                    compiled_model=model_path.name,
                    compiled_model_sha256=sha256(model_path),
                    mujoco_version=mujoco.__version__,
                ),
                indent=2,
            )
            + "\n"
        )
        (args.output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n")
        report = dict(
            scope=__doc__,
            backend=args.backend,
            conditioning=args.conditioning,
            checkpoint_sha256=[
                sha256(path)
                for path in [args.stage1_checkpoint, args.stage2_checkpoint]
            ],
            composition_proof=proof,
            collider_equivalence_claim=False,
            allow_measured_collider_difference=args.allow_measured_collider_difference,
            initial=initial,
            initial_actor_observation=initial_observation,
            initial_actor_mean=initial_mean,
            model_options={
                name: int(getattr(model.opt, name))
                for name in [
                    "solver",
                    "integrator",
                    "cone",
                    "iterations",
                    "ls_iterations",
                    "sdf_initpoints",
                    "sdf_iterations",
                ]
            }
            | {
                "timestep": float(model.opt.timestep),
                "impratio": float(model.opt.impratio),
            },
            final=current,
            handoffs=handoffs,
            contact_free_completion=bool(
                current["completed"] and current["contacts"] == 0
            ),
            error=error,
            wall_seconds=time.monotonic() - started,
            physics_peak=monitor.snapshot() if monitor else None,
            native_warning_counts=env.data.warning.number.tolist()
            if args.backend == "native"
            else None,
            source=source,
            runtime=runtime_info(),
            script_sha256=sha256(Path(__file__)),
        )
        (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
        if args.backend == "mjlab":
            env.close()
    print(json.dumps(report), flush=True)
    if error:
        raise SystemExit(error)


if __name__ == "__main__":
    main()
