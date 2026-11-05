"""Isaac Lab tight-insertion reset and observation/action randomization.

Queue lengths follow the source allocation, so length3 samples delays0/1/2.
Scale noise changes observations, not mesh dimensions or reward keypoints.
The privileged critic stays free of state noise and observation delays.
"""

import math

import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from play2perfect_mjlab import assembly_mdp as mdp


def yaw_quaternion(angle):
    zeros = torch.zeros_like(angle)
    return torch.stack(((angle / 2).cos(), zeros, zeros, (angle / 2).sin()), -1)


def initialize_random_goals(env):
    """Source chooses the free-space subset once, rather than on every reset."""
    s, cfg = mdp.state(env), env.cfg.source_randomization
    s["random_goal_env"] = (
        torch.rand(env.num_envs, device=env.device) < cfg["random_goal_fraction"]
    )
    s["max_goals"][s["random_goal_env"]] = cfg["random_goal_max_successes"]
    s["random_goal_pose"] = torch.zeros(env.num_envs, 7, device=env.device)
    s["random_goal_pose"][:, 3] = 1
    s["random_goal_tolerance"] = cfg["random_goal_initial_tolerance_m"]
    s["last_curriculum_update"] = 0
    s["lift_bonus_active"] = True


def sample_random_goal(env, ids, absolute):
    if not len(ids):
        return
    s, cfg = mdp.state(env), env.cfg.source_randomization
    lo = s["goals"].new_tensor(cfg["random_goal_position_min"])
    hi = s["goals"].new_tensor(cfg["random_goal_position_max"])
    if absolute:
        position = lo + (hi - lo) * torch.rand(len(ids), 3, device=env.device)
        rotation = torch.nn.functional.normalize(
            torch.randn(len(ids), 4, device=env.device), dim=-1
        )
    else:
        previous = s["random_goal_pose"][ids]
        position = (
            previous[:, :3]
            + uniform(env, (len(ids), 3), -1, 1) * cfg["random_goal_delta_position"]
        ).clamp(lo, hi)
        axis = torch.nn.functional.normalize(
            torch.randn(len(ids), 3, device=env.device), dim=-1
        )
        angle = uniform(env, (len(ids),), -1, 1) * math.radians(
            cfg["random_goal_delta_rotation_degrees"]
        )
        delta = torch.cat(
            ((angle / 2).cos()[:, None], (angle / 2).sin()[:, None] * axis), -1
        )
        rotation = quat_mul(delta, previous[:, 3:])
    s["random_goal_pose"][ids] = torch.cat((position, rotation), -1)


def update_curriculum(env):
    s, cfg = mdp.state(env), env.cfg.source_randomization
    selected = s["previous_goal_count"][s["random_goal_env"]]
    if not len(selected):
        return
    mean_goals = float(selected.float().mean())
    if mean_goals >= cfg["lift_bonus_fade_threshold"]:
        s["lift_bonus_active"] = False
    if (
        env.common_step_counter - s["last_curriculum_update"]
        >= cfg["tolerance_curriculum_interval"]
        and mean_goals >= cfg["tolerance_curriculum_success_threshold"]
    ):
        s["random_goal_tolerance"] = max(
            cfg["random_goal_target_tolerance_m"],
            s["random_goal_tolerance"] * cfg["tolerance_curriculum_increment"],
        )
        s["last_curriculum_update"] = env.common_step_counter


def buffers(env):
    if hasattr(env, "_p2p_dr"):
        return env._p2p_dr
    cfg = env.cfg.source_randomization
    env._p2p_dr = {
        name: torch.zeros(env.num_envs, cfg[key], size, device=env.device)
        for name, key, size in (
            ("object", "object_state_queue_length", 7),
            ("observation", "observation_queue_length", 140),
            ("action", "action_queue_length", 29),
        )
    }
    env._p2p_dr.update(
        stamp=torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long),
        actor=torch.zeros(env.num_envs, 141, device=env.device),
        table_z=torch.zeros(env.num_envs, device=env.device),
    )
    return env._p2p_dr


def uniform(env, shape, lower, upper):
    return torch.empty(shape, device=env.device).uniform_(lower, upper)


def reset(env, ids):
    """Sample the source reset distribution; scene reset runs before this hook."""
    cfg, s, dr = env.cfg.source_randomization, mdp.state(env), buffers(env)
    n = len(ids)
    limits = s["limits"]
    default = limits.new_tensor(cfg["robot_default_position"])
    sampled = limits[:, 0] + torch.rand(n, 29, device=env.device) * (
        limits[:, 1] - limits[:, 0]
    )
    # The source uses interpolation over each joint's range, not additive ±0.1.
    position = torch.lerp(default, sampled, cfg["robot_position_interpolation"])
    position = position.clamp(limits[:, 0], limits[:, 1])
    velocity = uniform(
        env, (n, 29), -cfg["robot_velocity_range"], cfg["robot_velocity_range"]
    )
    env.scene["robot"].write_joint_state_to_sim(position, velocity, env_ids=ids)

    table_z = cfg["table_z"] + uniform(
        env, (n,), -cfg["table_z_range"], cfg["table_z_range"]
    )
    dr["table_z"][ids] = table_z
    table = torch.zeros(n, 7, device=env.device)
    table[:, 2], table[:, 3] = table_z, 1
    env.scene["table"].write_mocap_pose_to_sim(table, ids)

    obj = torch.zeros(n, 7, device=env.device)
    obj[:, :3] = uniform(env, (n, 3), -1, 1) * obj.new_tensor(
        cfg["object_position_range"]
    )
    obj[:, 2] += table_z + cfg["object_z_above_table_origin"]
    # Normalized Gaussian four-vectors give a uniform quaternion on S^3/SO(3).
    obj[:, 3:] = torch.nn.functional.normalize(
        torch.randn(n, 4, device=env.device), dim=-1
    )
    env.scene["object"].write_root_link_pose_to_sim(obj, env_ids=ids)
    env.scene["object"].write_root_link_velocity_to_sim(
        torch.zeros(n, 6, device=env.device), env_ids=ids
    )

    fixture = torch.zeros(n, 7, device=env.device)
    fixture[:, 0] = uniform(env, (n,), *cfg["fixture_x_range"])
    fixture[:, 1] = uniform(env, (n,), *cfg["fixture_y_range"])
    base = obj.new_tensor(s["metadata"]["fixture_pose_xyzw"])
    fixture[:, 2] = (
        table_z
        + cfg["table_half_height"]
        + s["metadata"].get("source_fixture_z_offset", 0)
    )
    angle = uniform(env, (n,), -1, 1) * math.radians(cfg["fixture_yaw_degrees"])
    fixture[:, 3:] = yaw_quaternion(angle)
    random_goal = s["random_goal_env"][ids]
    fixture[random_goal, :2] = 0
    fixture[random_goal, 2] = -1
    env.scene["assembly_fixture"].write_mocap_pose_to_sim(fixture, ids)
    inverse = base[[6, 3, 4, 5]] * base.new_tensor([1, -1, -1, -1])
    delta = quat_mul(fixture[:, 3:], inverse.expand(n, -1))
    s["goal_delta_quat"][ids] = delta
    s["goal_delta_pos"][ids] = fixture[:, :3] - quat_apply(
        delta, base[:3].expand(n, -1)
    )

    s["bias"][ids] = 0
    noise = cfg["goal_xy_observation_range"]
    insertion_ids = ids[~random_goal]
    s["bias"][insertion_ids, :2] = uniform(env, (len(insertion_ids), 2), -noise, noise)
    if "goal_observation_yaw" not in s:
        s["goal_observation_yaw"] = torch.zeros(env.num_envs, device=env.device)
    s["goal_observation_yaw"][ids] = 0
    s["goal_observation_yaw"][insertion_ids] = uniform(
        env, (len(insertion_ids),), -1, 1
    ) * math.radians(cfg["goal_yaw_observation_degrees"])
    scale = uniform(env, (n, 3), *cfg["object_scale_observation_range"])
    s["offsets"][ids] = s["stage_offsets"][0] * scale[:, None]
    s["scales"][ids] = s["stage_scales"][0] * scale
    if "random_goal_pose" in s:
        sample_random_goal(env, ids[random_goal], absolute=True)
    for key in ("object", "observation", "action"):
        dr[key][ids] = 0
    dr["stamp"][ids] = -1


def sample_delay(queue, values, ids, flush):
    """Match the source's flush → roll → write-current → random-index order."""
    selected = queue[ids].clone()
    selected[flush] = values[flush, None]
    selected = torch.roll(selected, shifts=1, dims=1)
    selected[:, 0] = values
    index = torch.randint(queue.shape[1], (len(ids),), device=values.device)
    queue[ids] = selected
    return selected[torch.arange(len(ids), device=values.device), index]


def delayed_action(env, action):
    dr = buffers(env)
    ids = mdp.state(env)["env_ids"]
    # Source flushes only at the start of the physical episode, never a goal hit.
    flush = (env.episode_length_buf == 0) & (mdp.state(env)["goal_index"] == 0)
    return sample_delay(dr["action"], action, ids, flush)


def actor_observation(env):
    """Update noisy queues once per control; repeated reads return the same frame."""
    cfg, s, dr = env.cfg.source_randomization, mdp.state(env), buffers(env)
    pending = dr["stamp"] != env.common_step_counter
    ids = pending.nonzero().flatten()
    if not len(ids):
        return dr["actor"]
    flush = (env.episode_length_buf[ids] == 0) & (s["goal_index"][ids] == 0)
    pose = mdp.object_pose(env)[ids]
    delayed = sample_delay(dr["object"], pose, ids, flush)
    noisy_position = (
        delayed[:, :3]
        + torch.randn_like(delayed[:, :3]) * cfg["object_position_observation_std"]
    )
    axis = torch.nn.functional.normalize(
        torch.randn(len(ids), 3, device=env.device), dim=-1
    )
    angle = uniform(env, (len(ids),), -1, 1) * math.radians(
        cfg["object_rotation_observation_degrees"]
    )
    noise_q = torch.cat(
        ((angle / 2).cos()[:, None], (angle / 2).sin()[:, None] * axis), -1
    )
    noisy_rotation = quat_mul(noise_q, delayed[:, 3:])
    offsets = s["offsets"][ids]
    keys = (
        quat_apply(noisy_rotation[:, None].expand(-1, 4, -1), offsets)
        + noisy_position[:, None]
    )
    palm, _ = mdp.palm_and_tips(env)
    _, _, goal = mdp.geometry(env, s["offsets"], observed_goal=True)
    values = mdp.observation(env, observed_goal=False, goal_bias=False, clip=False)[
        ids, :140
    ].clone()
    values[:, 29:58] += (
        torch.randn_like(values[:, 29:58]) * cfg["joint_velocity_observation_std"]
    )
    values[:, 94:98] = noisy_rotation[:, [1, 2, 3, 0]]
    values[:, 113:125] = (keys - palm[ids, None]).flatten(1)
    values[:, 125:137] = (keys - goal[ids]).flatten(1)
    delayed_obs = sample_delay(dr["observation"], values, ids, flush).clamp(-10, 10)
    # PreciseAssembly subtracts per-episode goal XY noise after the base delay/clip.
    delayed_obs[:, 125:137] -= s["bias"][ids, None].expand(-1, 4, -1).flatten(1)
    dr["actor"][ids, :140] = delayed_obs.clamp(-10, 10)
    dr["actor"][ids, 140] = 50
    dr["stamp"][ids] = env.common_step_counter
    return dr["actor"]
