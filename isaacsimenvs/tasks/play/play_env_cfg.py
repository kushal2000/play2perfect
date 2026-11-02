"""PlayEnvCfg — typed defaults for the Play goal-pose-reaching task.

Organized into sectioned sub-configclasses that mirror the YAML overlay
in cfg/task/Play.yaml 1:1:

    sim                     → isaaclab.sim.SimulationCfg (+ PhysxCfg)
    scene                   → PlaySceneCfg(InteractiveSceneCfg)
    obs                     → ObsCfg
    student_obs             → StudentObsCfg (disabled by default)
    action                  → ActionCfg
    reward                  → RewardCfg
    reset                   → ResetCfg   (includes goal sampling)
    termination             → TerminationCfg (includes tolerance curriculum)
    domain_randomization    → DomainRandomizationCfg

Values match the legacy isaacgymenvs/cfg/task/Play.yaml defaults with
the following deliberate deviations (see plan file
.claude/plans/we-are-currently-in-twinkling-bengio.md):

  - `controlFrequencyInv` removed; Isaac Lab's `decimation=2` + `sim.dt=1/120`
    yields the same 60 Hz policy / 120 Hz physics as legacy `dt=1/60 +
    substeps=2`.
  - `fallDistance` / `fallPenalty` removed (unused in legacy env.py).
  - `useRelativeControl` removed (legacy True branch not being ported).
  - DR tree pruned to obs/action/object-state delays + force/torque impulses +
    object-scale & joint-vel obs noise (see DomainRandomizationCfg docstring).
  - Curricula pruned to tolerance curriculum only.

The Env class itself (`play_env.py:PlayEnv`) is still a stub —
all DirectRLEnv hooks raise NotImplementedError. Phases B–H populate them.
"""

from __future__ import annotations

from isaaclab.envs import DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass


# ----------------------------------------------------------------------------
# scene — kept as plain InteractiveSceneCfg (num_envs + layout knobs only).
# Isaac Lab's InteractiveScene._add_entities_from_cfg iterates every field
# on the scene cfg and rejects anything that isn't an AssetBaseCfg-derived
# config, so the asset metadata (URDF paths, frictions, procedural knobs)
# must live under a sibling section — see AssetsCfg below.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# assets — URDFs, procedural-generation knobs, static per-material frictions
# ----------------------------------------------------------------------------


@configclass
class AssetsCfg:
    robot_urdf: str = (
        "assets/urdf/kuka_sharpa_description/iiwa14_left_sharpa_adjusted_restricted.urdf"
    )
    table_urdf: str = "assets/urdf/table_narrow.urdf"
    # Per-env scale ranges applied to the table mesh at scene-build time.
    # Sampled independently per env: sx ~ U(table_scale_range_x), sy ~ U(table_scale_range_y).
    # Z is held at 1.0 so the table surface height stays at table_reset_z (which the
    # policy was trained to expect). Default (1, 1) means no scaling (legacy behavior).
    table_scale_range_x: tuple[float, float] = (1.0, 1.0)
    table_scale_range_y: tuple[float, float] = (1.0, 1.0)
    # Number of pre-baked USD variants spanning the scale range. Per env Isaac Lab
    # round-robins across this list at scene build, so each env's table has a
    # different XY footprint drawn from the configured ranges.
    table_scale_num_variants: int = 1

    object_name: str = "handle_head_primitives"
    handle_head_types: tuple[str, ...] = (
        "hammer",
        "screwdriver",
        "marker",
        "spatula",
        "eraser",
        "brush",
    )
    num_assets_per_type: int = 100

    # Shuffle the procedural pool after generation. Legacy default (True)
    # gives env i uniform coverage over types via i % len(pool). Debug/parity
    # runs set this False so pool[0] is the first matching distribution
    # (cuboid hammer ahead of cylinder hammer, etc.) — see
    # debug_differences/policy_rollout_isaacsim.py.
    shuffle_assets: bool = True

    # Static per-material frictions (set once at asset creation, not per-reset DR).
    modify_asset_frictions: bool = True
    robot_friction: float = 0.5
    finger_tip_friction: float = 1.5
    object_friction: float = 0.5
    table_friction: float = 0.5

# ----------------------------------------------------------------------------
# obs
# ----------------------------------------------------------------------------

@configclass
class ObsCfg:
    """Asymmetric actor-critic obs layout + clamping."""

    # Critic sees the full state list; actor sees the obs list subset.
    state_list: tuple[str, ...] = (
        "joint_pos",
        "joint_vel",
        "prev_action_targets",
        "palm_pos",
        "palm_rot",
        "palm_vel",
        "object_rot",
        "object_vel",
        "fingertip_pos_rel_palm",
        "keypoints_rel_palm",
        "keypoints_rel_goal",
        "object_scales",
        "closest_keypoint_max_dist",
        "closest_fingertip_dist",
        "lifted_object",
        "progress",
        "successes",
        "reward",
    )
    obs_list: tuple[str, ...] = (
        "joint_pos",
        "joint_vel",
        "prev_action_targets",
        "palm_pos",
        "palm_rot",
        "object_rot",
        "fingertip_pos_rel_palm",
        "keypoints_rel_palm",
        "keypoints_rel_goal",
        "object_scales",
    )

    clamp_abs_observations: float = 10.0


# ----------------------------------------------------------------------------
# student_obs
# ----------------------------------------------------------------------------


@configclass
class StudentObsCfg:
    """Optional camera + proprio observation path for distillation students.

    This is disabled by default and is not part of DirectRLEnv's normal
    ``_get_observations`` path. Distillation code explicitly calls
    ``env.unwrapped.get_student_obs()`` when this section is enabled.
    """

    enabled: bool = False

    # Proprio fields are assembled in this order from the same canonical joint
    # helper used by the teacher observation path.
    proprio_list: tuple[str, ...] = (
        "joint_pos",
        "joint_vel",
        "prev_action_targets",
    )

    image_enabled: bool = True
    image_modality: str = "depth"  # "depth" | "rgb" | "rgbd"
    image_width: int = 160
    image_height: int = 90
    image_input_width: int = 160
    image_input_height: int = 90
    crop_enabled: bool = False
    crop_top_left: tuple[int, int] = (0, 0)  # (x0, y0), inclusive
    crop_bottom_right: tuple[int, int] = (0, 0)  # (x1, y1), exclusive

    use_camera_delay: bool = False
    camera_delay_max: int = 0
    use_student_obs_delay: bool = False
    student_obs_delay_max: int = 0

    # "clip_divide" | "window_normalize" | "metric"
    depth_preprocess_mode: str = "window_normalize"
    depth_min_m: float = 0.45
    depth_max_m: float = 1.25
    hide_goal_viz: bool = True

    camera_backend: str = "tiled"  # "tiled" | "standard" | "raycaster"
    camera_mount: str = "world"
    camera_convention: str = "ros"
    camera_pos: tuple[float, float, float] = (0.0, -1.0, 1.0)
    camera_quat_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    focal_length: float = 24.0
    horizontal_aperture: float = 33.19737869997174
    # USD PinholeCamera aperture offsets shift the principal point without
    # changing FOV or image dimensions.  Defaults are set so the implied
    # intrinsics match the ZED HD1080 calibration on lab serial 15107
    # (fx=fy=115.67, cx=80.02, cy=47.13 at the 160x90 retrieve resolution).
    # If you ever switch cameras or move to a centered-aperture sensor, set
    # both offsets to 0.0 to recover the geometric-center default.
    #   horizontal_offset_cm = (cx - W/2) * focal_length / fx
    #     = (80.02 - 80) * 24 / 115.67   ~= 0.0035 cm  (sub-pixel)
    #   vertical_offset_cm   = (cy - H/2) * focal_length / fy
    #     = (47.13 - 45) * 24 / 115.67   ~= 0.4418 cm
    horizontal_aperture_offset: float = 0.0035
    vertical_aperture_offset: float = 0.4418
    focus_distance: float = 400.0
    clipping_range: tuple[float, float] = (0.1, 5.0)
    # Sensor update period in seconds. 0.0 = render every policy step (60 Hz
    # at our 60-Hz policy cadence). Set to 1/30 = 0.03333 to render at 30 Hz
    # (matches real ZED), in which case the sensor returns its cached output
    # on alternate policy steps. Cuts depth-render cost ~in half and closes
    # a sim2real fidelity gap (real ZED is 30 Hz, we control at 60 Hz).
    camera_update_period_s: float = 0.0

    # ---- RayCaster-only knobs (consumed when camera_backend == "raycaster") ----
    # USD prim-path globs the raycaster casts against. Entries in
    # `raycast_static_prim_exprs` have their pose cached at sensor init (no
    # per-step view query, cheaper). Entries in `raycast_dynamic_prim_exprs`
    # have their mesh transforms refreshed every step — list every prim
    # whose pose changes (robot links, the peg, the receptive, the table if
    # table-DR is enabled).
    raycast_static_prim_exprs: tuple[str, ...] = ("/World/ground",)
    # Point at each rigid body's `/visuals` subgroup, NOT the rigid body root.
    # The URDF importer attaches the visual-origin xform (e.g.
    # `<origin xyz="0 0 0.38"/>` on a table) at the `/box/visuals` level;
    # targeting the rigid body root collapses across that xform and the
    # raycaster places the geometry at z=0 instead. For the iiwa+sharpa
    # articulation we use `.*` to pick up every link's `/visuals` child —
    # the MultiMeshRayCaster creates a view that tracks each matched prim's
    # world pose independently, so articulation joints update per step.
    raycast_dynamic_prim_exprs: tuple[str, ...] = (
        "/World/envs/env_.*/Table/box/visuals",
        "/World/envs/env_.*/Hole/hole/visuals",
        # Wildcard the Object's link-name subpath so this works for any
        # task / problem URDF (peg, lpeg, fmb_peg_board_*, fabrica beam
        # parts, furniture parts, ...). The single-link case matches one
        # `/visuals` group; multi-link URDFs (fabrica) match each link's
        # `/visuals` group independently, which is what we want.
        "/World/envs/env_.*/Object/.*/visuals",
        # iiwa arm + sharpa hand link visuals. Explicit prefixes (not a
        # broad `/Robot/.*/visuals`) so the parser doesn't try to make
        # rigid-body views for non-link prims like `/Robot/Looks` /
        # `/Robot/joints` (those stall sensor init for several minutes
        # with PhysX retries before timing out).
        "/World/envs/env_.*/Robot/iiwa14_link_.*/visuals",
        "/World/envs/env_.*/Robot/left_.*/visuals",
    )
    # Rays that don't intersect any mesh return max_distance (instead of NaN)
    # when `depth_clipping_behavior == "max"`. Keep at the rasterizer's default
    # ("none" → NaN at infinity) so downstream depth-window normalization
    # treats far rays the same as the TiledCamera path.
    raycast_max_distance_m: float = 10.0
    raycast_depth_clipping_behavior: str = "none"  # "max" | "zero" | "none"

    # ---- Fast-FoundationStereo settings (used when camera_backend == "foundation_stereo") ----
    # Spawns a stereo TiledCamera pair (left at `camera_pos`/`camera_quat_wxyz`,
    # right at left + R_left @ [baseline, 0, 0]), renders RGB at
    # `fs_stereo_width × fs_stereo_height`, runs Fast-FS inference to recover
    # disparity, converts to metric depth, and downsamples to `image_width ×
    # image_height` before the existing depth-noise / crop / normalize chain.
    #
    # See deployment/FAST_FS_SETUP.md for weight download + ONNX/engine build.
    fs_model_dir: str = "third_party/Fast-FoundationStereo/weights/23-36-37"
    # When set, prefer the TRT engine pair at this directory
    # (feature_runner.engine + post_runner.engine + onnx.yaml). Otherwise the
    # wrapper falls back to PyTorch inference on `{fs_model_dir}/model_best_bp2_serialize.pth`.
    fs_engine_dir: str = ""
    fs_valid_iters: int = 4
    fs_max_disp: int = 192
    # ZED 1 stereo baseline. Override per camera serial via SDK calibration.
    fs_stereo_baseline_m: float = 0.120
    # Stereo render size. Multiples of 32. 384x224 matches the team's deployment
    # ONNX export (~3.9 ms TRT / 26 ms PyTorch on RTX 6000 Ada).
    fs_stereo_width: int = 384
    fs_stereo_height: int = 224
    # If true (the default), the FS depth is downsampled to `image_width ×
    # image_height` before _apply_depth_noise / _crop_student_image. If false,
    # the policy-input resolution is set to the stereo resolution (only useful
    # for sanity-check debugging).
    fs_downsample_to_policy_res: bool = True

    # Camera pose randomization (sampled per env at reset, fixed during episode).
    # Master switch + numerical defaults are the team's "medium" preset.
    use_camera_pose_rand: bool = False
    camera_pos_noise_m: tuple[float, float, float] = (0.01, 0.01, 0.01)
    camera_rot_noise_deg: tuple[float, float, float] = (1.0, 1.0, 1.0)

    # Depth-image noise pipeline (5 stages, applied in raw meters BEFORE preprocess).
    # Master switch + numerical defaults are the team's "medium" preset.
    use_depth_aug: bool = False
    depth_aug_gaussian_std_m: float = 0.002
    depth_aug_correlated_std_m: float = 0.003
    depth_aug_correlated_kernel_size: int = 5
    depth_aug_dropout_prob: float = 0.003
    depth_aug_randu_prob: float = 0.003
    depth_aug_randu_min_m: float = 0.50
    depth_aug_randu_max_m: float = 1.30
    depth_aug_stick_prob: float = 0.00025
    depth_aug_max_sticks_per_image: int = 8
    depth_aug_stick_max_len_px: int = 18
    depth_aug_stick_max_width_px: int = 3


# ----------------------------------------------------------------------------
# action
# ----------------------------------------------------------------------------


@configclass
class ActionCfg:
    """Joint-position-target control with moving-average smoothing."""

    arm_moving_average: float = 0.1
    hand_moving_average: float = 0.1
    dof_speed_scale: float = 1.5


# ----------------------------------------------------------------------------
# reward
# ----------------------------------------------------------------------------


@configclass
class RewardCfg:
    """Four-term reward: keypoint + lifting (w/ bonus) + distance-delta +
    reach-goal bonus + action-magnitude penalties.
    """

    keypoint_rew_scale: float = 200.0
    keypoint_scale: float = 1.5
    object_base_size: float = 0.04
    fixed_size: tuple[float, float, float] = (0.141, 0.03025, 0.0271)
    fixed_size_keypoint_reward: bool = True

    lifting_rew_scale: float = 20.0
    lifting_bonus: float = 300.0
    lifting_bonus_threshold: float = 0.15

    distance_delta_rew_scale: float = 50.0
    reach_goal_bonus: float = 1000.0

    kuka_actions_penalty_scale: float = 0.03
    hand_actions_penalty_scale: float = 0.003

    # Where the fingertip-to-object distance behind the distance-delta term is measured from.
    # False (the released Kuka policies): each fingertip body's origin, at the last finger joint.
    # True (the Flexiv + Wuji task): the pad point the policy observes, body +
    # RobotSpec.fingertip_offset, as in SimToolReal's Isaac Gym env.
    fingertip_distance_from_pad: bool = False


# ----------------------------------------------------------------------------
# reset (includes goal sampling — both fire on _reset_idx)
# ----------------------------------------------------------------------------


@configclass
class ResetCfg:
    """Initial-state distribution + goal sampling (sampled at every reset)."""

    # Initial object pose noise
    reset_position_noise_x: float = 0.1
    reset_position_noise_y: float = 0.1
    reset_position_noise_z: float = 0.02
    fixed_start_pose: tuple[float, float, float, float, float, float, float] | None = None

    # Joint state noise on reset
    reset_dof_pos_random_interval_arm: float = 0.1
    reset_dof_pos_random_interval_fingers: float = 0.1
    reset_dof_vel_random_interval: float = 0.5

    # Table reset geometry
    table_reset_z: float = 0.38
    table_reset_z_range: float = 0.01
    table_object_z_offset: float = 0.25
    # Table centre in env XY; objects spawn around it (the Flexiv task moves it to its real setup).
    table_reset_xy: tuple[float, float] = (0.0, 0.0)
    # Per-env XY position noise applied at reset (uniform half-widths in m).
    # Default (0, 0) keeps the table centered on the env origin (legacy behavior).
    table_reset_xy_range_m: tuple[float, float] = (0.0, 0.0)
    # Per-env yaw noise applied at reset (uniform half-width in degrees about z).
    # Default 0.0 preserves the identity quat (legacy behavior).
    table_reset_yaw_range_deg: float = 0.0

    # Goal sampling
    goal_sampling_type: str = "delta"  # "delta" | "absolute"
    delta_goal_distance: float = 0.1
    delta_rotation_degrees: float = 90.0
    target_volume_mins: tuple[float, float, float] = (-0.35, -0.2, 0.6)
    target_volume_maxs: tuple[float, float, float] = (0.35, 0.2, 0.95)
    target_volume_region_scale: float = 1.0
    # When on, each sampled goal's z is raised (past target_volume_maxs if need be) until the
    # object at the goal orientation is goal_table_clearance_m above the table top
    # (utils/goal_table_clearance.py). Off leaves goals where the sampler puts them.
    use_goal_table_clearance: bool = False
    goal_table_clearance_m: float = 0.01

    # Debug only — when set, every reset writes this exact env-local pose
    # to GoalViz instead of sampling. Format: (x, y, z, qw, qx, qy, qz).
    # Used by debug_differences/* to keep both envs visually aligned.
    fixed_goal_pose: tuple[float, float, float, float, float, float, float] | None = None

    # Fixed-trajectory ablation: when ``fixed_trajectory_file`` is non-empty,
    # the env ignores ``goal_sampling_type`` and instead draws goal sequences
    # from a pre-generated pool of (N_total, K, 3+4) trajectories in the JSON
    # file. ``fixed_trajectory_count`` truncates the pool to the first N
    # (0 = use the whole file). Pair with ``termination.max_consecutive_
    # successes == K`` so episodes end exactly when a trajectory is exhausted.
    #
    # Empty-string / 0 defaults are deliberate: isaaclab's configclass type-
    # checks hydra overrides against the default value's *runtime* type, so a
    # ``str | None = None`` field rejects string overrides at parse time.
    fixed_trajectory_file: str = ""
    fixed_trajectory_count: int = 0


# ----------------------------------------------------------------------------
# termination (includes tolerance curriculum — governs success criterion)
# ----------------------------------------------------------------------------


@configclass
class TerminationCfg:
    """Episode-end conditions + success-tolerance curriculum.

    The episode-length-extends-on-goal-hit behavior (legacy
    ``progress_buf[is_success > 0] = 0`` at env.py:2503-2505) lands in
    Phase F's ``_get_dones`` — there it zeros ``self.episode_length_buf``
    for envs that hit a goal, so the framework's default truncation check
    only fires on *time without progress*, not on total time in episode.
    """

    episode_length: int = 600  # steps (policy steps; 600 * decimation * dt = 10s)

    success_tolerance: float = 0.075  # curriculum start
    target_success_tolerance: float = 0.01  # curriculum floor
    eval_success_tolerance: float | None = None

    success_steps: int = 10
    max_consecutive_successes: int = 50
    force_consecutive_near_goal_steps: bool = False

    # Tolerance curriculum (the only curriculum in v1).
    tolerance_curriculum_increment: float = 0.9  # multiplicative per step
    tolerance_curriculum_interval: int = 3000  # env steps across all agents
    tolerance_curriculum_success_threshold: float = 3.0


# ----------------------------------------------------------------------------
# domain_randomization
# ----------------------------------------------------------------------------


@configclass
class DomainRandomizationCfg:
    """Sim2real DR set. Scoped to per-episode / per-step perturbations that
    the paper identifies as essential for transfer. Physics-param DR (gravity,
    DOF damping/stiffness/effort/friction/armature, rigid-body mass,
    rigid-shape friction/restitution) is *not* ported in v1.
    """

    # Obs / action latency
    use_obs_delay: bool = True
    obs_delay_max: int = 3
    use_action_delay: bool = True
    action_delay_max: int = 3

    # Object state delay + noise on the observed object pose.
    use_object_state_delay_noise: bool = True
    object_state_delay_max: int = 10
    object_state_xyz_noise_std: float = 0.01
    object_state_rotation_noise_degrees: float = 5.0
    # Multiplicative per-env scale noise applied to keypoint offsets and to the
    # object_scales obs (legacy env.py:3093-3098,3193-3195).
    object_scale_noise_multiplier_range: tuple[float, float] = (1.0, 1.0)

    # Per-step Gaussian noise on joint-velocity obs (legacy env.py:3251).
    joint_velocity_obs_noise_std: float = 0.1

    # Random force/torque impulses on the object body.
    force_scale: float = 20.0
    force_prob_range: tuple[float, float] = (0.001, 0.1)
    force_decay: float = 0.0
    force_decay_interval: float = 0.08
    force_only_when_lifted: bool = True

    torque_scale: float = 2.0
    torque_prob_range: tuple[float, float] = (0.001, 0.1)
    torque_decay: float = 0.0
    torque_decay_interval: float = 0.08
    torque_only_when_lifted: bool = True

    # Per-env friction randomization, sampled ONCE at scene init (not at
    # reset). Multiplicative scales of the AssetsCfg base values. Default
    # (1.0, 1.0) is a no-op so existing runs are unaffected.
    #
    # Why init-only with bucketing: PhysX caps live materials at 64K and
    # set_material_properties creates a new material per distinct
    # (static, dynamic, restitution) tuple, so per-reset randomization
    # exhausts the limit in seconds. Init-only with discrete buckets caps
    # the material count at ~`friction_n_buckets` per axis.
    #
    # Mass randomization is not exposed: set_masses raises
    # "Failed to set rigid body masses in backend" in this Isaac Lab /
    # PhysX configuration. The proper path is Isaac Lab's
    # EventCfg.ActorMassRandomization, which is a larger refactor.
    object_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    fingertip_friction_scale_range: tuple[float, float] = (1.0, 1.0)
    friction_n_buckets: int = 16


# ----------------------------------------------------------------------------
# Top-level configclass — composes the above, plus DirectRLEnvCfg requireds
# ----------------------------------------------------------------------------


def _default_sim_cfg(gpu_max_rigid_patch_count: int | None = None) -> SimulationCfg:
    """60 Hz policy control / 120 Hz physics (matches legacy dt=1/60 + substeps=2).

    ``gpu_max_rigid_patch_count`` only sizes a PhysX buffer: Isaac Lab's default (163840) overflows
    at 24576 envs with robot self-collision on, and PhysX then drops the contacts that do not fit.
    """
    physx_extra = {}
    if gpu_max_rigid_patch_count is not None:
        physx_extra["gpu_max_rigid_patch_count"] = int(gpu_max_rigid_patch_count)
    return SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=2,
        gravity=(0.0, 0.0, -9.81),
        physx=PhysxCfg(
            solver_type=1,  # 1 = TGS (matches legacy)
            min_position_iteration_count=8,
            max_position_iteration_count=8,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            friction_offset_threshold=0.04,
            friction_correlation_distance=0.025,
            **physx_extra,
        ),
    )


@configclass
class PlayEnvCfg(DirectRLEnvCfg):
    """Top-level configclass for the Play goal-pose-reaching env.

    Structure mirrors ``cfg/task/Play.yaml`` exactly — YAML overlay
    key paths resolve to these fields via ``configclass.from_dict``.
    """

    # --- DirectRLEnvCfg required fields ---
    decimation: int = 2  # 2 physics substeps per policy step
    episode_length_s: float = 10.0  # 600 policy steps * 2 * (1/120) = 10s
    action_space: int = 29  # 7-DOF IIWA + 22-DOF SHARPA hand
    # Obs/state sizes are derived from obs.obs_list / obs.state_list at env init.
    # Placeholder keeps the configclass instantiable before the env computes the
    # final spaces.
    observation_space: int = 140
    state_space: int = 140

    # --- Isaac Lab base fields ---
    sim: SimulationCfg = _default_sim_cfg()
    # Viewer is the camera DirectRLEnv.render('rgb_array') captures from. One
    # render product (omni.replicator) is lazily allocated at this prim path
    # on first render() call — single buffer, num_envs-independent. eye/lookat
    # are world-frame; with replicate_physics=False the central env sits near
    # world origin at large num_envs, so framing the table/robot here works.
    viewer: ViewerCfg = ViewerCfg(
        eye=(0.5, -1.5, 1.2),
        lookat=(0.0, 0.4, 0.5),
        resolution=(640, 480),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=8192,
        env_spacing=1.2,
        # Per-env distinct USDs (MultiUsdFileCfg) require:
        #  - replicate_physics=False so PhysX parses each env as its own
        #    subtree (otherwise variants collapse into a single template;
        #    Isaac Lab also emits a hard warning — see
        #    isaaclab/scene/interactive_scene.py).
        #  - clone_in_fabric=False so the cloner replicates env_0 into the
        #    USD stage (not just Fabric). MultiUsdFileCfg's spawner resolves
        #    the regex prim_path via find_matching_prim_paths, which only
        #    sees USD prims; with clone_in_fabric=True env_1..env_{N-1}
        #    exist only in Fabric and the multi-asset spawn lands in env_0.
        replicate_physics=False,
        clone_in_fabric=False,
    )

    # --- Sectioned sub-configs (mirror YAML sections 1:1) ---
    assets: AssetsCfg = AssetsCfg()
    obs: ObsCfg = ObsCfg()
    student_obs: StudentObsCfg = StudentObsCfg()
    action: ActionCfg = ActionCfg()
    reward: RewardCfg = RewardCfg()
    reset: ResetCfg = ResetCfg()
    termination: TerminationCfg = TerminationCfg()
    domain_randomization: DomainRandomizationCfg = DomainRandomizationCfg()

    # Robot embodiment, a key of robots.ROBOT_SPECS. Fixed per registered task; assets.robot_urdf
    # must be its URDF. PlayEnv sets action_space (and the obs sizes) from it.
    robot: str = "kuka_sharpa"
    # Robot self-collision, minus robots.RobotSpec.self_collision_filter. Off for the Kuka (as its
    # released policies were trained); on for the Flexiv + Wuji.
    robot_self_collision: bool = False


@configclass
class PlayFlexivWujiRightEnvCfg(PlayEnvCfg):
    """Play on a Flexiv Rizon 4s + right Wuji Hand 2 with human-like finger joint limits
    (``Isaacsimenvs-PlayFlexivWujiRight-Direct-v0``).

    The same task as ``PlayEnvCfg`` except the robot, robot self-collision (on), and the
    real setup's table (``cfg/task/PlayFlexivWujiRight.yaml``).
    """

    robot: str = "flexiv_wuji_right_human_like"
    robot_self_collision: bool = True
    action_space: int = 27  # 7-DOF Rizon 4s + 20-DOF Wuji hand
    sim: SimulationCfg = _default_sim_cfg(gpu_max_rigid_patch_count=2**21)
    assets: AssetsCfg = AssetsCfg(
        robot_urdf="assets/urdf/flexiv_wuji/rizon4s_right_wuji_human_like_joint_limits.urdf"
    )
    reward: RewardCfg = RewardCfg(fingertip_distance_from_pad=True)


__all__ = [
    "PlayEnvCfg",
    "PlayFlexivWujiRightEnvCfg",
    "AssetsCfg",
    "ObsCfg",
    "StudentObsCfg",
    "ActionCfg",
    "RewardCfg",
    "ResetCfg",
    "TerminationCfg",
    "DomainRandomizationCfg",
]
