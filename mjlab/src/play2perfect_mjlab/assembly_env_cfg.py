"""Manager-based assembly task. Start with the validated tight-insertion scene."""

import json
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import reset_scene_to_default, time_out
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sim import MujocoCfg, SimulationCfg

from play2perfect_mjlab import assembly_mdp as mdp
from play2perfect_mjlab.fabrica import add_freeze_welds, restore_static_excludes
from play2perfect_mjlab.manager_assets import scene_config

TASK_CONTRACT_VERSION = 5
TRAINING_PRESETS = {
    "isaac_lab_tight": ("tight_insertion", "data/released_evaluation/tight_insertion"),
    "isaac_lab_long_leg": ("screwing", "data/long_leg/long_leg"),
    "source_beam_step1": (
        "beam_assembly_step1",
        "data/released_evaluation/beam_assembly_step1",
    ),
    # The original standalone reference overlaps the robot by14.396mm.
    "source_beam_step2": (
        "beam_assembly_step2",
        "data/released_fixture_left/beam_assembly_step2",
    ),
}
TRAINING_PRESET_TASKS = {key: value[0] for key, value in TRAINING_PRESETS.items()}


def training_bundle(preset):
    """Return the validated reference bundle for a named training preset."""
    return Path(TRAINING_PRESETS[preset][1])


@dataclass(kw_only=True)
class AssemblyEnvCfg(ManagerBasedRlEnvCfg):
    p2p_metadata: dict = field(default_factory=dict)
    kinematics_only_rewards: bool = False
    sdf_initpoints: int = 10
    withdrawal_tolerance_m: float = 0.0075
    restore_source_static_filters: bool = False
    source_randomization: dict = field(default_factory=dict)


def assembly_cfg(
    bundle: Path,
    num_envs=1,
    auto_reset=False,
    sdf_initpoints=None,
    kinematics_only_rewards=False,
    start=0,
    restore_source_static_filters=False,
    training_preset=None,
    evaluation_preset=None,
    nconmax=512,
    njmax=2048,
    solver_iterations=100,
    ls_iterations=50,
    physics_dt=None,
    decimation=None,
    preserve_world_geometry=True,
    initialize_object_qpos0=True,
):
    metadata = json.loads((bundle / "manifest.json").read_text())
    metadata["world_geometry_preserved"] = preserve_world_geometry
    if metadata.get("stages"):
        metadata["object_qpos0_preserved"] = initialize_object_qpos0
    if (physics_dt is None) != (decimation is None):
        raise ValueError("Override physics_dt and decimation together")
    if physics_dt is not None:
        policy_dt = metadata["physics_dt"] * metadata["decimation"]
        if (
            physics_dt <= 0
            or decimation < 1
            or abs(physics_dt * decimation - policy_dt) > 1e-9
        ):
            raise ValueError(
                "Physics overrides must preserve the policy control period"
            )
        metadata["physics_dt"], metadata["decimation"] = physics_dt, decimation
    if metadata["task"] not in (
        "tight_insertion",
        "screwing",
        "fabrica",
        "beam_assembly_step1",
        "beam_assembly_step2",
    ):
        raise ValueError("Unknown assembly task")
    if metadata.get("initialization_pool"):
        metadata["initial_qpos"][29:] = metadata["initialization_pool"][start][
            "part_qpos"
        ]
    elif start:
        raise ValueError("This bundle has only canonical start0")
    if sdf_initpoints is not None and sdf_initpoints < 1:
        raise ValueError("SDF sample count must be positive")
    default_points = (
        50 if training_preset in ("source_beam_step1", "source_beam_step2") else 10
    )
    point_count = default_points if sdf_initpoints is None else sdf_initpoints
    randomization = {}
    actor_observation = mdp.observation
    if training_preset is not None or evaluation_preset is not None:
        if training_preset is not None and metadata[
            "task"
        ] != TRAINING_PRESET_TASKS.get(training_preset):
            raise ValueError("Source training preset does not match the task")
        if (
            training_preset == "isaac_lab_long_leg"
            and metadata.get("problem_name") != "long_leg"
        ):
            raise ValueError("Long-leg training requires the source225mm bundle")
        if evaluation_preset not in (None, "public_offline") or (
            evaluation_preset and training_preset
        ):
            raise ValueError("Choose training or public offline evaluation")
        if "fixture_pose_xyzw" not in metadata:
            raise ValueError("Source training requires a released-URDF bundle")
        from play2perfect_mjlab.source_randomization import actor_observation

        config_root = files("play2perfect_mjlab").joinpath("resources", "configs")
        randomization = json.loads(
            config_root.joinpath("isaac_lab_tight.json").read_text()
        )
        long_leg = metadata.get("problem_name") == "long_leg"
        if training_preset == "isaac_lab_long_leg" or (evaluation_preset and long_leg):
            randomization.update(
                json.loads(config_root.joinpath("isaac_lab_long_leg.json").read_text())
            )
        if training_preset in ("source_beam_step1", "source_beam_step2"):
            randomization.update(
                json.loads(config_root.joinpath("source_beam_steps.json").read_text())
            )
            randomization["name"] = training_preset + "_v1"
            step = 1 if training_preset == "source_beam_step1" else 2
            randomization["peer_recipe"] = (
                randomization["source_recipe_directory"]
                + "/"
                + randomization[f"source_step{step}_recipe"]
            )
        if evaluation_preset:
            # Retain object start variation and goal XY bias, as public eval does.
            randomization.update(
                name="public_offline_long_leg_v2" if long_leg else "public_offline_v1",
                robot_position_interpolation=0,
                robot_velocity_range=0,
                table_z_range=0,
                fixture_yaw_degrees=0,
                goal_yaw_observation_degrees=0,
                object_scale_observation_range=[1, 1],
                object_state_queue_length=1,
                observation_queue_length=1,
                action_queue_length=1,
                object_position_observation_std=0,
                object_rotation_observation_degrees=0,
                joint_velocity_observation_std=0,
                # The long-leg browser/native reference advances at7.5mm for
                # one control. Generic released-task scoring uses15mm/10.
                success_steps=metadata["success_steps"] if long_leg else 10,
                advance_tolerance_m=metadata["insertion_tolerance"]
                if long_leg
                else 0.015,
                random_goal_fraction=0.0,
            )
        for key in ("success_steps", "near_goal_counter"):
            metadata[key] = randomization[key]
        metadata["insertion_tolerance"] = randomization["advance_tolerance_m"]
        metadata["withdrawal_tolerance_m"] = randomization["withdrawal_tolerance_m"]
        # The matched recipe samples goal noise on reset, replacing saved bias.
        metadata["goal_yaw_bias"] = 0

    def configure_spec(spec):
        # Attached entities do not propagate global MuJoCo options.
        spec.option.sdf_iterations = 10
        spec.option.sdf_initpoints = point_count
        if metadata.get("stages"):
            if initialize_object_qpos0:
                # Simulation runs native mj_forward before the first reset.
                # Entity keyframes are valid, but both free roots otherwise
                # have qpos0 at the origin and their SDF meshes interpenetrate.
                for name, stage in zip(("object", "second"), metadata["stages"]):
                    address = stage.get("object_qpos_address", 29)
                    pose = metadata["initial_qpos"][address : address + 7]
                    body = spec.body(f"{name}/{name}")
                    body.pos, body.quat = pose[:3], pose[3:]
            reference_model = add_freeze_welds(spec, bundle, metadata)
            if restore_source_static_filters:
                restore_static_excludes(spec, reference_model)

    return AssemblyEnvCfg(
        p2p_metadata=metadata,
        kinematics_only_rewards=kinematics_only_rewards,
        sdf_initpoints=point_count,
        withdrawal_tolerance_m=metadata.get("withdrawal_tolerance_m", 0.0075),
        restore_source_static_filters=restore_source_static_filters,
        source_randomization=randomization,
        scene=scene_config(
            bundle,
            metadata,
            num_envs=num_envs,
            env_spacing=1.2,
            spec_fn=configure_spec,
            preserve_world_geometry=preserve_world_geometry,
        ),
        sim=SimulationCfg(
            nconmax=nconmax,
            njmax=njmax,
            mujoco=MujocoCfg(
                timestep=metadata["physics_dt"],
                integrator="implicitfast",
                cone="elliptic",
                impratio=10,
                iterations=solver_iterations,
                ls_iterations=ls_iterations,
            ),
        ),
        decimation=metadata["decimation"],
        episode_length_s=10,
        auto_reset=auto_reset,
        scale_rewards_by_dt=False,
        observations={
            "actor": ObservationGroupCfg(
                terms={"assembly": ObservationTermCfg(func=actor_observation)},
                enable_corruption=False,
            ),
            "critic": ObservationGroupCfg(
                terms={"assembly": ObservationTermCfg(func=mdp.critic_observation)},
                enable_corruption=False,
            ),
        },
        actions={"assembly": mdp.AssemblyActionCfg(entity_name="robot")},
        events={
            "scene": EventTermCfg(func=reset_scene_to_default, mode="reset"),
            "task": EventTermCfg(func=mdp.reset_task, mode="reset"),
        },
        rewards={"assembly": RewardTermCfg(func=mdp.reward, weight=1)},
        terminations={
            "update_task": TerminationTermCfg(func=mdp.update_task),
            "completed": TerminationTermCfg(func=mdp.completed),
            "object_fell": TerminationTermCfg(func=mdp.object_fell),
            "hand_far": TerminationTermCfg(func=mdp.hand_far),
            "release_failed": TerminationTermCfg(func=mdp.release_failed),
            "timeout": TerminationTermCfg(func=time_out, time_out=True),
        },
    )


def tight_insertion_cfg(bundle: Path, num_envs=1, auto_reset=False):
    if json.loads((bundle / "manifest.json").read_text())["task"] != "tight_insertion":
        raise ValueError("Expected the tight-insertion asset bundle")
    return assembly_cfg(bundle, num_envs, auto_reset)


def task_contract(cfg):
    """Record the actual training contract rather than inferring it from success."""
    md = cfg.p2p_metadata
    return {
        "version": TASK_CONTRACT_VERSION,
        "world_geometry_preserved": md.get("world_geometry_preserved", False),
        "task": md["task"],
        "critic_linear_velocity": "body center of mass",
        "advance_tolerance_m": md.get("insertion_tolerance", 0.015),
        "success_steps": md.get("success_steps", 10),
        "near_goal_counter": md.get("near_goal_counter", "accumulated"),
        "withdrawal_tolerance_m": cfg.withdrawal_tolerance_m,
        "reward_fingertips": "body origins",
        "release_controls": md.get("release_follow_through_controls", 0),
        "physics_dt": cfg.sim.mujoco.timestep,
        "decimation": cfg.decimation,
        "sdf_initpoints": cfg.sdf_initpoints,
        "source_static_filters_restored": cfg.restore_source_static_filters
        if md.get("stages")
        else False,
        **(
            {"object_qpos0_preserved": md.get("object_qpos0_preserved", False)}
            if md.get("stages")
            else {}
        ),
        "kinematics_only_rewards": cfg.kinematics_only_rewards,
        "randomization": cfg.source_randomization or "none",
        "contact_buffer_per_world": cfg.sim.nconmax,
        "constraint_buffer_per_world": cfg.sim.njmax,
        "solver_iterations": cfg.sim.mujoco.iterations,
        "line_search_iterations": cfg.sim.mujoco.ls_iterations,
    }
