"""Batched assembly observations, controls, source rewards and goal progression."""

from dataclasses import dataclass

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from play2perfect_mjlab.fabrica import handoff


def state(env):
    if hasattr(env, "_p2p"):
        return env._p2p
    m = env.sim.mj_model
    md = env.cfg.p2p_metadata

    def tensor(value):
        return torch.as_tensor(value, device=env.device, dtype=torch.float32)

    def zero():
        return torch.zeros(env.num_envs, device=env.device)

    stages = md.get("stages") or [md]
    stage_offsets = tensor([stage["keypoint_offsets"] for stage in stages])
    stage_scales = tensor([stage["object_scales"] for stage in stages])
    stage_bias = tensor([stage["goal_position_bias"] for stage in stages])
    stage_addresses = [stage.get("object_qpos_address", 29) for stage in stages]
    env._p2p = {
        "metadata": md,
        "goals": tensor(md["goals"]),
        "stage_goals": tensor([stage["goals"] for stage in stages]),
        "stage_offsets": stage_offsets,
        "stage_scales": stage_scales,
        "stage_bias": stage_bias,
        "stage_yaw": tensor([stage.get("goal_yaw_bias", 0) for stage in stages]),
        "stage_addresses": stage_addresses,
        "stage_index": zero().long(),
        "offsets": stage_offsets[0].expand(env.num_envs, -1, -1).clone(),
        "reward_offsets": tensor(md["reward_offsets"]),
        "scales": stage_scales[0].expand(env.num_envs, -1).clone(),
        "bias": stage_bias[0].expand(env.num_envs, -1).clone(),
        "goal_delta_quat": tensor([1, 0, 0, 0]).expand(env.num_envs, -1).clone(),
        "goal_delta_pos": zero()[:, None].expand(-1, 3).clone(),
        "limits": tensor(m.jnt_range[:29]),
        "palm": m.body("robot/link7").id,
        "object_body": zero().long().fill_(m.body("object/object").id),
        "object_address": zero().long().fill_(stage_addresses[0]),
        "env_ids": torch.arange(env.num_envs, device=env.device),
        "body_roots": torch.as_tensor(m.body_rootid, device=env.device),
        "tips": [
            m.body(f"robot/palmleft_{f}_DP").id
            for f in ("index", "middle", "ring", "thumb", "pinky")
        ],
        "goal_index": zero().long(),
        "near_steps": zero().long(),
        "retract": zero().bool(),
        "completed": zero().bool(),
        "withdrawn": zero().bool(),
        "release_remaining": zero().long().fill_(-1),
        "release_failed": zero().bool(),
        "recurrent_reset": zero().bool(),
        "handoff_step": zero().long(),
        "lifted": zero().bool(),
        "closest_kp": zero().fill_(-1),
        "closest_ft": zero()[:, None].expand(-1, 5).clone().fill_(-1),
        "init_z": zero().fill_(md["initial_qpos"][31]),
        "reward": zero(),
        "distance": zero(),
        "reached": zero().bool(),
        "bootstrap_critic": torch.zeros(env.num_envs, 162, device=env.device),
        "episode_completed": zero().bool(),
        "total_goals": zero().long(),
        "episode_goal_count": zero().long(),
        "episode_fell": zero().bool(),
        "previous_goal_count": zero().long(),
        "max_goals": zero().long().fill_(len(stages[0]["goals"])),
        "random_goal_env": zero().bool(),
    }
    if env.cfg.source_randomization.get("random_goal_fraction", 0):
        from play2perfect_mjlab.source_randomization import initialize_random_goals

        initialize_random_goals(env)
    return env._p2p


@dataclass(kw_only=True)
class AssemblyActionCfg(ActionTermCfg):
    def build(self, env):
        return AssemblyAction(self, env)


class AssemblyAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        initial = torch.as_tensor(
            env.cfg.p2p_metadata["initial_ctrl"], device=env.device
        )
        self.targets = initial.expand(env.num_envs, -1).clone()
        self._raw = torch.zeros_like(self.targets)

    @property
    def action_dim(self):
        return 29

    @property
    def raw_action(self):
        return self._raw

    def process_actions(self, actions):
        self._raw[:] = actions.clamp(-1, 1)
        action = self._raw
        if self._env.cfg.source_randomization:
            from play2perfect_mjlab.source_randomization import delayed_action

            action = delayed_action(self._env, action)
        limits = state(self._env)["limits"]
        lower, upper = limits[:, 0], limits[:, 1]
        raw = lower + 0.5 * (action + 1) * (upper - lower)
        raw[:, :7] = (self.targets[:, :7] + 0.025 * action[:, :7]).clamp(
            lower[:7], upper[:7]
        )
        self.targets[:] = (0.1 * raw + 0.9 * self.targets).clamp(lower, upper)

    def apply_actions(self):
        self._entity.set_joint_position_target(self.targets)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self.targets[ids] = self._entity.data.joint_pos[ids]


def object_pose(env):
    addresses = state(env)["object_address"][:, None]
    indices = addresses + torch.arange(7, device=env.device)[None]
    return env.sim.data.qpos.gather(1, indices)


def goal_pose(env, observed=False):
    s = state(env)
    index = s["goal_index"].clamp(max=len(s["goals"]) - 1)
    goal = (
        s["per_world_goals"][s["env_ids"], index]
        if "per_world_goals" in s
        else s["stage_goals"][s["stage_index"], index]
    )
    position = quat_apply(s["goal_delta_quat"], goal[:, :3]) + s["goal_delta_pos"]
    quaternion = quat_mul(s["goal_delta_quat"], goal[:, [6, 3, 4, 5]])
    if "random_goal_pose" in s:
        position = torch.where(
            s["random_goal_env"][:, None], s["random_goal_pose"][:, :3], position
        )
        quaternion = torch.where(
            s["random_goal_env"][:, None], s["random_goal_pose"][:, 3:], quaternion
        )
    if observed:
        half_yaw = (
            s["stage_yaw"][s["stage_index"]] + s.get("goal_observation_yaw", 0)
        ) / 2
        zeros = torch.zeros_like(half_yaw)
        yaw_quaternion = torch.stack((half_yaw.cos(), zeros, zeros, half_yaw.sin()), -1)
        quaternion = quat_mul(yaw_quaternion, quaternion)
    return torch.cat((position, quaternion), -1)


def geometry(env, offsets, observed_goal=False):
    pose = object_pose(env)
    goal = goal_pose(env, observed_goal)
    q = pose[:, None, 3:].expand(-1, 4, -1)
    gq = goal[:, None, 3:].expand(-1, 4, -1)
    if offsets.ndim == 2:
        offsets = offsets[None].expand(env.num_envs, -1, -1)
    obj_keys = quat_apply(q, offsets) + pose[:, None, :3]
    goal_keys = quat_apply(gq, offsets) + goal[:, None, :3]
    return pose, obj_keys, goal_keys


def palm_and_tips(env):
    s, d = state(env), env.sim.data
    palm_offset = d.qpos.new_tensor([0, -0.02, 0.16]).expand(env.num_envs, -1)
    tip_offset = d.qpos.new_tensor([0.02, 0.002, 0]).expand(env.num_envs, 5, -1)
    palm = d.xpos[:, s["palm"]] + quat_apply(d.xquat[:, s["palm"]], palm_offset)
    tips = d.xpos[:, s["tips"]] + quat_apply(d.xquat[:, s["tips"]], tip_offset)
    return palm, tips


def reward_fingertip_distances(env, object_position):
    """Source rewards/termination use body origins; actor observations use offsets."""
    positions = env.sim.data.xpos[:, state(env)["tips"]]
    return torch.linalg.vector_norm(positions - object_position[:, None], dim=-1)


def observation(env, observed_goal=True, goal_bias=True, clip=True):
    s, d = state(env), env.sim.data
    limits = s["limits"]
    joints = 2 * (d.qpos[:, :29] - limits[:, 0]) / (limits[:, 1] - limits[:, 0]) - 1
    palm, tips = palm_and_tips(env)
    pose, obj, goal = geometry(env, s["offsets"], observed_goal=observed_goal)
    palm_q = d.xquat[:, s["palm"]][:, [1, 2, 3, 0]]
    targets = env.action_manager.get_term("assembly").targets
    values = torch.cat(
        (
            joints,
            d.qvel[:, :29],
            targets,
            palm,
            palm_q,
            pose[:, [4, 5, 6, 3]],
            (tips - palm[:, None]).flatten(1),
            (obj - palm[:, None]).flatten(1),
            (obj - goal - (s["bias"][:, None] if goal_bias else 0)).flatten(1),
            s["scales"],
        ),
        dim=1,
    )
    if clip:
        values = values.clamp(-10, 10)
    return torch.cat((values, values.new_full((env.num_envs, 1), 50)), dim=1)


def body_com_velocity(env, body):
    """World linear/angular velocity at each body's COM, as Isaac body_state_w."""
    s, d = state(env), env.sim.data
    if isinstance(body, int):
        velocity = d.cvel[:, body]
        center = d.subtree_com[:, s["body_roots"][body]]
        position = d.xipos[:, body]
    else:
        velocity = d.cvel[s["env_ids"], body]
        center = d.subtree_com[s["env_ids"], s["body_roots"][body]]
        position = d.xipos[s["env_ids"], body]
    linear = velocity[:, 3:] + torch.cross(velocity[:, :3], position - center, dim=-1)
    return torch.cat((linear, velocity[:, :3]), -1)


def critic_observation(env):
    s = state(env)
    actor = observation(env)
    _, obj, goal = geometry(env, s["offsets"])
    clean_keys_goal = obj - goal
    values = torch.cat(
        (
            actor[:, :94],
            body_com_velocity(env, s["palm"]),
            actor[:, 94:98],
            body_com_velocity(env, s["object_body"]),
            actor[:, 98:125],
            clean_keys_goal.flatten(1),
            actor[:, 137:140],
            s["closest_kp"][:, None],
            s["closest_ft"],
            s["lifted"].float()[:, None],
            torch.log1p(env.episode_length_buf.float() / 10)[:, None],
            torch.log1p(s["goal_index"].float())[:, None],
            (s["reward"] * 0.01)[:, None],
        ),
        -1,
    )
    return values.clamp(-10, 10)


def reset_task(env, env_ids):
    s = state(env)
    s["previous_goal_count"][env_ids] = s["total_goals"][env_ids]
    for key in (
        "goal_index",
        "near_steps",
        "retract",
        "completed",
        "lifted",
        "reward",
        "reached",
        "stage_index",
        "withdrawn",
        "release_failed",
        "recurrent_reset",
        "handoff_step",
        "total_goals",
    ):
        s[key][env_ids] = 0
    s["closest_kp"][env_ids] = -1
    s["closest_ft"][env_ids] = -1
    s["offsets"][env_ids] = s["stage_offsets"][0]
    s["scales"][env_ids] = s["stage_scales"][0]
    s["bias"][env_ids] = s["stage_bias"][0]
    s["object_body"][env_ids] = env.sim.mj_model.body("object/object").id
    s["object_address"][env_ids] = s["stage_addresses"][0]
    s["goal_delta_pos"][env_ids] = 0
    s["goal_delta_quat"][env_ids] = s["goal_delta_quat"].new_tensor([1, 0, 0, 0])
    s["release_remaining"][env_ids] = -1
    if env.cfg.source_randomization:
        from play2perfect_mjlab.source_randomization import reset

        reset(env, env_ids)
    s["init_z"][env_ids] = object_pose(env)[env_ids, 2]
    if s["metadata"].get("stages"):
        for index in range(2):
            eq_id = env.sim.mj_model.equality(f"freeze_{index}").id
            env.sim.data.eq_active[env_ids, eq_id] = False
    write_goal(env, env_ids)


def write_goal(env, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    env.scene["goal_object"].write_mocap_pose_to_sim(goal_pose(env)[ids], env_ids)


def update_task(env):
    # Source rewards use post-integration geometry. Refresh once here rather
    # than inheriting the manager's documented one-physics-substep reward lag.
    if env.cfg.kinematics_only_rewards:
        # Refresh poses and spatial velocities without rerunning contacts/solver.
        with wp.ScopedDevice(env.sim.wp_device):
            mjwarp.kinematics(env.sim.wp_model, env.sim.wp_data)
            mjwarp.com_pos(env.sim.wp_model, env.sim.wp_data)
            mjwarp.com_vel(env.sim.wp_model, env.sim.wp_data)
    else:
        env.sim.forward()
    s, d = state(env), env.sim.data
    if env.cfg.source_randomization.get("random_goal_fraction", 0):
        from play2perfect_mjlab.source_randomization import update_curriculum

        update_curriculum(env)
    s["recurrent_reset"].zero_()
    pose, obj, goal = geometry(env, s["reward_offsets"])
    distance = torch.linalg.vector_norm(obj - goal, dim=-1).amax(-1)
    ft = reward_fingertip_distances(env, pose[:, :3])
    z_lift = 0.05 + pose[:, 2] - s["init_z"]
    just_lifted = ~s["lifted"] & (z_lift > 0.15)
    lifted = s["lifted"] | (z_lift > 0.15)
    s["closest_ft"][:] = torch.where(s["closest_ft"] < 0, ft, s["closest_ft"])
    s["closest_kp"][:] = torch.where(s["closest_kp"] < 0, distance, s["closest_kp"])
    s["lifted"][:] = lifted
    was_retract = s["retract"].clone()
    tolerance = distance.new_full(
        distance.shape, s["metadata"].get("insertion_tolerance", 0.015)
    )
    if "random_goal_tolerance" in s:
        tolerance[s["random_goal_env"]] = s["random_goal_tolerance"]
    near = (distance <= tolerance) & ~was_retract
    consecutive = s["metadata"].get("near_goal_counter", "accumulated") == "consecutive"
    if consecutive:
        s["near_steps"][:] = (s["near_steps"] + 1) * near.long()
    else:
        s["near_steps"] += near.long()
    success_steps = s["metadata"].get("success_steps", 10)
    reached = (s["near_steps"] >= success_steps) & ~was_retract
    s["reached"][:] = reached
    s["goal_index"] += reached.long()
    s["total_goals"] += reached.long()
    all_goals = s["goal_index"] >= s["max_goals"]
    s["retract"] |= all_goals & ~s["random_goal_env"]
    # Source _get_dones clears both trackers before _get_rewards on an
    # intermediate goal transition. The final goal retains them for withdrawal.
    next_goal = reached & ~all_goals & ~s["retract"]
    if "random_goal_pose" in s:
        from play2perfect_mjlab.source_randomization import sample_random_goal

        sample_random_goal(
            env, (next_goal & s["random_goal_env"]).nonzero().flatten(), absolute=False
        )
    s["closest_ft"][next_goal] = -1
    s["closest_kp"][next_goal] = -1
    ft_rew = (s["closest_ft"] - ft).clamp(0, 10).sum(-1) * (~lifted) * 50
    kp_rew = (s["closest_kp"] - distance).clamp(0, 100) * lifted * 200
    s["closest_ft"][:] = torch.minimum(s["closest_ft"], ft)
    s["closest_kp"][:] = torch.minimum(s["closest_kp"], distance)
    at_withdrawal_goal = distance <= env.cfg.withdrawal_tolerance_m
    withdrawn = s["retract"] & at_withdrawal_goal & (ft.mean(-1) > 0.1)
    just_withdrawn = withdrawn & ~s["withdrawn"]
    s["withdrawn"] |= withdrawn
    arm_penalty = -0.03 * d.qvel[:, :7].abs().sum(-1)
    hand_penalty = -0.003 * d.qvel[:, 7:29].abs().sum(-1)
    penalty = arm_penalty + hand_penalty
    retract_reward = ft.mean(-1) * at_withdrawal_goal + just_withdrawn * 5000
    # In insertion-only source environments, lift reward and lift bonus are off.
    goal_bonus = reached * 1000 if consecutive else near * (1000 / success_steps)
    reward = ft_rew + kp_rew + penalty + goal_bonus
    if "random_goal_pose" in s:
        dr = env.cfg.source_randomization
        lift = z_lift.clamp(0, 0.5) * ~lifted * dr["lifting_reward_scale"]
        lift_bonus = just_lifted * dr["lifting_bonus"]
        lift_mask = s["random_goal_env"] & s["lift_bonus_active"]
        # Preserve the source's summation/masking order as well as its terms.
        reward = (
            lift
            + lift_bonus
            + ft_rew
            + kp_rew
            + arm_penalty
            + hand_penalty
            + goal_bonus
        )
        reward = reward - lift - lift_bonus + lift * lift_mask + lift_bonus * lift_mask
    s["reward"][:] = torch.where(was_retract, penalty + retract_reward, reward)
    s["distance"][:] = distance
    s["near_steps"][reached] = 0
    env.episode_length_buf[reached] = 0
    if s["metadata"].get("stages"):
        first_part_done = withdrawn & (s["stage_index"] == 0)
        final_part = s["stage_index"] == 1
        counting = (s["release_remaining"] >= 0) & final_part
        s["release_remaining"][counting] -= 1
        start_release = final_part & just_withdrawn
        s["release_remaining"][start_release] = s["metadata"].get(
            "release_follow_through_controls", 60
        )
        release_ends = final_part & (s["release_remaining"] == 0)
        s["completed"] |= release_ends & withdrawn
        s["release_failed"] |= release_ends & ~withdrawn
        handoff(env, s, first_part_done)
    else:
        s["completed"] |= withdrawn | (all_goals & s["random_goal_env"])
    s["episode_completed"][:] = s["completed"]
    # These snapshots survive auto-reset for episode metrics and trainer info.
    s["episode_goal_count"][:] = s["total_goals"]
    s["episode_fell"][:] = pose[:, 2] < 0.1
    write_goal(env)
    # Retain the pre-reset state for correct timeout bootstrapping in PPO.
    s["bootstrap_critic"][:] = critic_observation(env)
    return torch.zeros_like(s["completed"])


def reward(env):
    return state(env)["reward"]


def completed(env):
    return state(env)["completed"]


def object_fell(env):
    return object_pose(env)[:, 2] < 0.1


def release_failed(env):
    return state(env)["release_failed"]


def hand_far(env):
    s = state(env)
    pose, _, _ = geometry(env, s["offsets"])
    distances = reward_fingertip_distances(env, pose[:, :3])
    return (distances.amax(-1) > 1.5) & ~s["retract"]
