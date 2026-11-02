"""Robot embodiments the Play task can run: everything about the robot that the task code reads.

``PlayEnvCfg.robot`` names one of :data:`ROBOT_SPECS`; each registered task fixes it, so a robot
is chosen by task id (``Isaacsimenvs-Play-Direct-v0`` / ``Isaacsimenvs-PlayFlexivWujiRight-Direct-v0``)
rather than by a runtime flag. ``cfg.assets.robot_urdf`` must be the matching URDF; the env checks
that its joints are exactly the spec's.

Kit-free on purpose (stdlib only), so tests and scripts can read a spec without booting Isaac.

The action pipeline (``action_utils.py``, byte-identical to upstream SimToolReal) treats the first
7 joints in Isaac Lab's order as the arm and the rest as the hand. ``allocate_state_buffers``
asserts that ordering for whichever robot is loaded.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class RobotSpec:
    name: str
    arm_joint_regex: str
    hand_joint_regex: str
    #: Policy-facing joint order (arm first). Isaac Lab tensors are permuted to and from it.
    joint_names_canonical: tuple[str, ...]
    #: Body the palm observation is read from, and the offset (in that body's frame) to the
    #: palm centre the policy sees.
    palm_body_name: str
    palm_center_offset: tuple[float, float, float]
    #: Bodies that carry the fingertip collision after the URDF importer merges fixed joints,
    #: and the offset (in each one's frame) to the pad centre.
    fingertip_body_regex: str
    fingertip_link_names: tuple[str, ...]
    #: One offset for all five, or one per fingertip_link_names entry.
    fingertip_offset: tuple[float, float, float] | tuple[tuple[float, float, float], ...]
    arm_stiffness: dict[str, float]
    arm_damping: dict[str, float]
    hand_stiffness: dict[str, float]
    hand_damping: dict[str, float]
    hand_armature: dict[str, float]
    #: Reset pose; hand joints not listed default to 0.
    arm_default_joint_pos: dict[str, float]
    #: Body pairs excluded from robot self-collision (the task turns it on), body -> its partners,
    #: by post-merge_fixed_joints body name. scene_utils authors them as FilteredPairsAPI and
    #: fails if any name is not a body of the imported robot.
    self_collision_filter: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Hand drive limits and joint friction; None keeps what the robot's asset has.
    hand_effort_limit: dict[str, float] | None = None
    hand_velocity_limit: float | None = None
    hand_friction: float | None = None

    @property
    def fingertip_offsets(self) -> tuple[tuple[float, float, float], ...]:
        """The pad offset of each fingertip, in fingertip_link_names order."""
        if isinstance(self.fingertip_offset[0], (int, float)):
            return (tuple(self.fingertip_offset),) * len(self.fingertip_link_names)
        return tuple(tuple(o) for o in self.fingertip_offset)

    @property
    def num_joints(self) -> int:
        return len(self.joint_names_canonical)

    @property
    def arm_joint_names(self) -> tuple[str, ...]:
        return tuple(self.arm_stiffness)

    @property
    def hand_joint_names(self) -> tuple[str, ...]:
        return tuple(self.hand_stiffness)

    def __post_init__(self) -> None:
        arm, hand = self.arm_joint_names, self.hand_joint_names
        if len(arm) != 7:
            raise ValueError(f"{self.name}: action_utils assumes a 7-DoF arm, got {len(arm)}")
        if self.joint_names_canonical != arm + hand:
            raise ValueError(f"{self.name}: canonical order must be the arm gains' keys, then the hand's")
        for table in ("arm_damping", "arm_default_joint_pos"):
            if set(getattr(self, table)) != set(arm):
                raise ValueError(f"{self.name}: {table} must cover exactly the arm joints")
        for table in ("hand_damping", "hand_armature"):
            if set(getattr(self, table)) != set(hand):
                raise ValueError(f"{self.name}: {table} must cover exactly the hand joints")
        if self.hand_effort_limit is not None:
            if set(self.hand_effort_limit) != set(hand):
                raise ValueError(f"{self.name}: hand_effort_limit must cover exactly the hand joints")
        if len(self.fingertip_link_names) != 5:
            raise ValueError(f"{self.name}: the task observes 5 fingertips")
        if len(self.fingertip_offsets) != 5 or any(len(o) != 3 for o in self.fingertip_offsets):
            raise ValueError(f"{self.name}: fingertip_offset must be one 3-vector or five")


# ----------------------------------------------------------------------------
# Kuka iiwa14 + left Sharpa: SimToolReal's robot. Values verified with the pretrained checkpoint;
# scene_utils / obs_utils re-export them under their original names.
# ----------------------------------------------------------------------------

_SHARPA_JOINTS = (
    "left_1_thumb_CMC_FE", "left_thumb_CMC_AA", "left_thumb_MCP_FE",
    "left_thumb_MCP_AA", "left_thumb_IP",
    "left_2_index_MCP_FE", "left_index_MCP_AA", "left_index_PIP", "left_index_DIP",
    "left_3_middle_MCP_FE", "left_middle_MCP_AA", "left_middle_PIP", "left_middle_DIP",
    "left_4_ring_MCP_FE", "left_ring_MCP_AA", "left_ring_PIP", "left_ring_DIP",
    "left_5_pinky_CMC", "left_pinky_MCP_FE", "left_pinky_MCP_AA",
    "left_pinky_PIP", "left_pinky_DIP",
)

# SimToolReal's own list (isaacgymenvs/tasks/simtoolreal/adjacent_links.py,
# LEFT_SHARPA_KUKA_LINK_TO_ADJACENT_LINKS), verbatim: the recipe, so the Kuka control runs it.
# Its hand pairs are each jointed pair, the palm with the thumb MC (touching at rest), and the
# palm (or MC) with each proximal phalanx across the two-joint MCP knuckle.
_SHARPA_SELF_COLLISION_FILTER = {
    "iiwa14_link_0": ("iiwa14_link_1",),
    "iiwa14_link_1": ("iiwa14_link_0", "iiwa14_link_2"),
    "iiwa14_link_2": ("iiwa14_link_1", "iiwa14_link_3"),
    "iiwa14_link_3": ("iiwa14_link_2", "iiwa14_link_4"),
    "iiwa14_link_4": ("iiwa14_link_3", "iiwa14_link_5"),
    "iiwa14_link_5": ("iiwa14_link_4", "iiwa14_link_6"),
    "iiwa14_link_6": ("iiwa14_link_5", "iiwa14_link_7"),
    "iiwa14_link_7": (
        "iiwa14_link_6", "left_thumb_CMC_VL", "left_thumb_MC", "left_index_MCP_VL", "left_index_PP",
        "left_middle_MCP_VL", "left_middle_PP", "left_ring_MCP_VL", "left_ring_PP", "left_pinky_MC",
    ),
    "left_index_MCP_VL": ("iiwa14_link_7", "left_index_PP"),
    "left_index_PP": ("iiwa14_link_7", "left_index_MCP_VL", "left_index_MP"),
    "left_index_MP": ("left_index_PP", "left_index_DP"),
    "left_index_DP": ("left_index_MP",),
    "left_middle_MCP_VL": ("iiwa14_link_7", "left_middle_PP"),
    "left_middle_PP": ("iiwa14_link_7", "left_middle_MCP_VL", "left_middle_MP"),
    "left_middle_MP": ("left_middle_PP", "left_middle_DP"),
    "left_middle_DP": ("left_middle_MP",),
    "left_pinky_MC": ("iiwa14_link_7", "left_pinky_MCP_VL", "left_pinky_PP"),
    "left_pinky_MCP_VL": ("left_pinky_MC", "left_pinky_PP"),
    "left_pinky_PP": ("left_pinky_MC", "left_pinky_MCP_VL", "left_pinky_MP"),
    "left_pinky_MP": ("left_pinky_PP", "left_pinky_DP"),
    "left_pinky_DP": ("left_pinky_MP",),
    "left_ring_MCP_VL": ("iiwa14_link_7", "left_ring_PP"),
    "left_ring_PP": ("iiwa14_link_7", "left_ring_MCP_VL", "left_ring_MP"),
    "left_ring_MP": ("left_ring_PP", "left_ring_DP"),
    "left_ring_DP": ("left_ring_MP",),
    "left_thumb_CMC_VL": ("iiwa14_link_7", "left_thumb_MC"),
    "left_thumb_MC": ("iiwa14_link_7", "left_thumb_CMC_VL", "left_thumb_MCP_VL", "left_thumb_PP"),
    "left_thumb_MCP_VL": ("left_thumb_MC", "left_thumb_PP"),
    "left_thumb_PP": ("left_thumb_MC", "left_thumb_MCP_VL", "left_thumb_DP"),
    "left_thumb_DP": ("left_thumb_PP",),
}

KUKA_SHARPA = RobotSpec(
    name="kuka_sharpa",
    arm_joint_regex="iiwa14_joint_.*",
    hand_joint_regex="left_.*",
    joint_names_canonical=tuple(f"iiwa14_joint_{i}" for i in range(1, 8)) + _SHARPA_JOINTS,
    palm_body_name="iiwa14_link_7",
    # Policy was trained against the palm center, not the raw wrist body.
    palm_center_offset=(-0.0, -0.02, 0.16),
    # Merged fingertip bodies land on the DP links in both sims.
    fingertip_body_regex="left_(index|middle|ring|thumb|pinky)_DP",
    fingertip_link_names=(
        "left_index_DP", "left_middle_DP", "left_ring_DP",
        "left_thumb_DP", "left_pinky_DP",
    ),
    # Shift fingertip body origins to the approximate pad centers.
    fingertip_offset=(0.02, 0.002, 0.0),
    arm_stiffness={
        "iiwa14_joint_1": 600.0, "iiwa14_joint_2": 600.0, "iiwa14_joint_3": 500.0,
        "iiwa14_joint_4": 400.0, "iiwa14_joint_5": 200.0, "iiwa14_joint_6": 200.0,
        "iiwa14_joint_7": 200.0,
    },
    arm_damping={
        "iiwa14_joint_1": 27.027026473513512, "iiwa14_joint_2": 27.027026473513512,
        "iiwa14_joint_3": 24.672186769721083, "iiwa14_joint_4": 22.067474708266914,
        "iiwa14_joint_5": 9.752538131173853, "iiwa14_joint_6": 9.147747263670984,
        "iiwa14_joint_7": 9.147747263670984,
    },
    hand_stiffness={
        "left_1_thumb_CMC_FE": 6.95, "left_thumb_CMC_AA": 13.2, "left_thumb_MCP_FE": 4.76,
        "left_thumb_MCP_AA": 6.62, "left_thumb_IP": 0.9,
        "left_2_index_MCP_FE": 4.76, "left_index_MCP_AA": 6.62,
        "left_index_PIP": 0.9, "left_index_DIP": 0.9,
        "left_3_middle_MCP_FE": 4.76, "left_middle_MCP_AA": 6.62,
        "left_middle_PIP": 0.9, "left_middle_DIP": 0.9,
        "left_4_ring_MCP_FE": 4.76, "left_ring_MCP_AA": 6.62,
        "left_ring_PIP": 0.9, "left_ring_DIP": 0.9,
        "left_5_pinky_CMC": 1.38, "left_pinky_MCP_FE": 4.76, "left_pinky_MCP_AA": 6.62,
        "left_pinky_PIP": 0.9, "left_pinky_DIP": 0.9,
    },
    hand_damping={
        "left_1_thumb_CMC_FE": 0.28676845, "left_thumb_CMC_AA": 0.40845109,
        "left_thumb_MCP_FE": 0.20394083, "left_thumb_MCP_AA": 0.24044435,
        "left_thumb_IP": 0.04190723,
        "left_2_index_MCP_FE": 0.20859232, "left_index_MCP_AA": 0.24595532,
        "left_index_PIP": 0.04243185, "left_index_DIP": 0.03504461,
        "left_3_middle_MCP_FE": 0.2085923, "left_middle_MCP_AA": 0.24595532,
        "left_middle_PIP": 0.04243185, "left_middle_DIP": 0.03504461,
        "left_4_ring_MCP_FE": 0.20859226, "left_ring_MCP_AA": 0.24595528,
        "left_ring_PIP": 0.04243183, "left_ring_DIP": 0.0350446,
        "left_5_pinky_CMC": 0.02782345, "left_pinky_MCP_FE": 0.20859229,
        "left_pinky_MCP_AA": 0.24595528, "left_pinky_PIP": 0.04243183,
        "left_pinky_DIP": 0.0350446,
    },
    hand_armature={
        "left_1_thumb_CMC_FE": 0.0032, "left_thumb_CMC_AA": 0.0032,
        "left_thumb_MCP_FE": 0.00265, "left_thumb_MCP_AA": 0.00265, "left_thumb_IP": 0.0006,
        "left_2_index_MCP_FE": 0.00265, "left_index_MCP_AA": 0.00265,
        "left_index_PIP": 0.0006, "left_index_DIP": 0.00042,
        "left_3_middle_MCP_FE": 0.00265, "left_middle_MCP_AA": 0.00265,
        "left_middle_PIP": 0.0006, "left_middle_DIP": 0.00042,
        "left_4_ring_MCP_FE": 0.00265, "left_ring_MCP_AA": 0.00265,
        "left_ring_PIP": 0.0006, "left_ring_DIP": 0.00042,
        "left_5_pinky_CMC": 0.00012, "left_pinky_MCP_FE": 0.00265,
        "left_pinky_MCP_AA": 0.00265, "left_pinky_PIP": 0.0006, "left_pinky_DIP": 0.00042,
    },
    # Proven-working default arm pose.
    arm_default_joint_pos={
        "iiwa14_joint_1": -1.571, "iiwa14_joint_2": 1.571, "iiwa14_joint_3": 0.0,
        "iiwa14_joint_4": 1.376, "iiwa14_joint_5": 0.0, "iiwa14_joint_6": 1.485,
        "iiwa14_joint_7": 1.308,
    },
    self_collision_filter=_SHARPA_SELF_COLLISION_FILTER,
)


# ----------------------------------------------------------------------------
# Flexiv Rizon 4s + left Wuji Hand 2. Not released as a task here: it is the base the right-hand
# spec below mirrors. Geometry is derived from the models; the arm DYNAMICS ARE PROVISIONAL
# placeholders, to be refined against the hardware.
# ----------------------------------------------------------------------------

_RIZON_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
_RIZON_LINKS = ("base_link", *(f"link{i}" for i in range(1, 8)))
#: Rizon 4s joint torque limits (flexiv_description joint_limits.yaml), Nm.
_RIZON_EFFORT = (123.0, 123.0, 64.0, 64.0, 39.0, 39.0, 39.0)
_KUKA_ARM_K = tuple(KUKA_SHARPA.arm_stiffness.values())
_KUKA_ARM_D = tuple(KUKA_SHARPA.arm_damping.values())
#: Provisional: stiffness 2 Nm/rad per Nm of torque limit, so every joint saturates at the same
#: 0.5 rad of tracking error. Damping keeps the Kuka's per-joint damping ratio at equal link
#: inertia (d scales with sqrt(k)), which over-damps a little since the Rizon's links are lighter.
_RIZON_K = tuple(2.0 * e for e in _RIZON_EFFORT)
_RIZON_D = tuple(d * math.sqrt(k / kk) for d, k, kk in zip(_KUKA_ARM_D, _RIZON_K, _KUKA_ARM_K))

#: Wuji Hand 2 joints, thumb to pinky, 4 per finger (vendor URDF order).
_WUJI_JOINTS = tuple(
    f"l_{finger}_{joint}"
    for finger, joints in (
        ("thumb", ("cmc_flex", "cmc_abd", "mcp", "ip")),
        ("index_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
        ("middle_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
        ("ring_finger", ("mcp_flex", "mcp_abd", "pip", "dip")),
        ("pinky", ("mcp_flex", "mcp_abd", "pip", "dip")),
    )
    for joint in joints
)
#: Wuji's own Isaac Lab drives for the Hand 2 (wuji-technology/wuji-isaaclab@be53b71,
#: src/wuji_isaaclab/tasks/reorient/config/wuji_hand2/hand2_constants.py HAND2_DRIVE, written for
#: the right hand; the vendor MJCF is the same on both sides): (stiffness Nm/rad, damping Nm s/rad,
#: effort limit Nm, armature kg m^2), in _WUJI_JOINTS order. About 3x stiffer than the vendor
#: MJCF's position servos, which these replaced; the armature is the MJCF's. Their randomization
#: of these is not copied.
_WUJI_DRIVE = (
    (1.354985, 0.04769, 1.02, 0.0005), (2.198821, 0.062838, 0.83, 0.0005),
    (0.886467, 0.027982, 0.35, 0.0002), (0.654026, 0.020457, 0.21, 0.0002),
    (1.345207, 0.047402, 1.05, 0.0005), (0.88946, 0.027239, 0.38, 0.0002),
    (0.779361, 0.024489, 0.31, 0.0002), (0.656424, 0.020508, 0.30, 0.0002),
    (1.299313, 0.045699, 1.06, 0.0005), (0.942641, 0.027846, 0.39, 0.0002),
    (0.73725, 0.023408, 0.32, 0.0002), (0.710314, 0.022366, 0.29, 0.0002),
    (1.276892, 0.046209, 1.02, 0.0005), (1.013334, 0.028704, 0.41, 0.0002),
    (0.859159, 0.027436, 0.28, 0.0002), (0.648064, 0.020448, 0.30, 0.0002),
    (1.102915, 0.043816, 0.87, 0.0005), (0.910201, 0.026593, 0.36, 0.0002),
    (0.861968, 0.027476, 0.28, 0.0002), (0.656704, 0.020611, 0.30, 0.0002),
)
_WUJI_KP, _WUJI_KV, _WUJI_EFFORT, _WUJI_ARMATURE = zip(*_WUJI_DRIVE)
#: Theirs too: no joint friction, and a velocity limit "not a hardware limit: it only has to stay
#: above anything the fingers ever reach".
_WUJI_FRICTION = 0.0
_WUJI_VELOCITY_LIMIT = 100.0

# Derived by scripts/flexiv_wuji/self_collision_pairs.py (tests/test_robot_specs.py reruns it):
# each jointed pair, plus the palm (link7, into which l_wrist merges) with every finger's
# proximal_abd -- touching at reset on the ring and pinky, and through the knuckle's range on
# the rest, as SimToolReal filters the Sharpa's palm-to-proximal pairs.
_WUJI_SELF_COLLISION_FILTER = {
        'base_link': ('link1',),
        'l_index_finger_distal': ('l_index_finger_middle',),
        'l_index_finger_middle': ('l_index_finger_distal', 'l_index_finger_proximal_abd'),
        'l_index_finger_proximal': ('l_index_finger_proximal_abd', 'link7'),
        'l_index_finger_proximal_abd': ('l_index_finger_middle', 'l_index_finger_proximal', 'link7'),
        'l_middle_finger_distal': ('l_middle_finger_middle',),
        'l_middle_finger_middle': ('l_middle_finger_distal', 'l_middle_finger_proximal_abd'),
        'l_middle_finger_proximal': ('l_middle_finger_proximal_abd', 'link7'),
        'l_middle_finger_proximal_abd': ('l_middle_finger_middle', 'l_middle_finger_proximal', 'link7'),
        'l_pinky_distal': ('l_pinky_middle',),
        'l_pinky_middle': ('l_pinky_distal', 'l_pinky_proximal_abd'),
        'l_pinky_proximal': ('l_pinky_proximal_abd', 'link7'),
        'l_pinky_proximal_abd': ('l_pinky_middle', 'l_pinky_proximal', 'link7'),
        'l_ring_finger_distal': ('l_ring_finger_middle',),
        'l_ring_finger_middle': ('l_ring_finger_distal', 'l_ring_finger_proximal_abd'),
        'l_ring_finger_proximal': ('l_ring_finger_proximal_abd', 'link7'),
        'l_ring_finger_proximal_abd': ('l_ring_finger_middle', 'l_ring_finger_proximal', 'link7'),
        'l_thumb_distal': ('l_thumb_middle',),
        'l_thumb_middle': ('l_thumb_distal', 'l_thumb_proximal_abd'),
        'l_thumb_proximal': ('l_thumb_proximal_abd', 'link7'),
        'l_thumb_proximal_abd': ('l_thumb_middle', 'l_thumb_proximal', 'link7'),
        'link1': ('base_link', 'link2'),
        'link2': ('link1', 'link3'),
        'link3': ('link2', 'link4'),
        'link4': ('link3', 'link5'),
        'link5': ('link4', 'link6'),
        'link6': ('link5', 'link7'),
        'link7': ('l_index_finger_proximal', 'l_index_finger_proximal_abd', 'l_middle_finger_proximal', 'l_middle_finger_proximal_abd', 'l_pinky_proximal', 'l_pinky_proximal_abd', 'l_ring_finger_proximal', 'l_ring_finger_proximal_abd', 'l_thumb_proximal', 'l_thumb_proximal_abd', 'link6'),
}

FLEXIV_WUJI = RobotSpec(
    name="flexiv_wuji",
    arm_joint_regex="joint[1-7]",
    hand_joint_regex="l_.*",
    joint_names_canonical=_RIZON_JOINTS + _WUJI_JOINTS,
    # The importer merges flange, l_mount and l_wrist into link7.
    palm_body_name="link7",
    # The Kuka's palm centre carried over through the canonical hand frame (middle MCP origin,
    # knuckle-defined axes; scripts/flexiv_wuji/viser_compare.py): same spot relative to the
    # knuckles. Depends on flexiv_wuji.MOUNT_*; tests/test_robot_specs.py recomputes it.
    palm_center_offset=(-0.0018, -0.015, 0.1948),
    fingertip_body_regex="l_(thumb|index_finger|middle_finger|ring_finger|pinky)_distal",
    fingertip_link_names=(
        "l_index_finger_distal", "l_middle_finger_distal", "l_ring_finger_distal",
        "l_thumb_distal", "l_pinky_distal",
    ),
    # The tip is 24.8 mm down each distal link's -z (29.8 on the thumb); the Sharpa's offset sits
    # ~77% of the way to its tip, and each finger's is the same fraction of its own -- on the
    # thumb too, which a single 19 mm left inside the thumb (64% of the way).
    fingertip_offset=(
        (0.0, 0.0, -0.019), (0.0, 0.0, -0.019), (0.0, 0.0, -0.019),
        (0.0, 0.0, -0.0229), (0.0, 0.0, -0.019),
    ),
    arm_stiffness=dict(zip(_RIZON_JOINTS, _RIZON_K)),
    arm_damping=dict(zip(_RIZON_JOINTS, _RIZON_D)),
    hand_stiffness=dict(zip(_WUJI_JOINTS, _WUJI_KP)),
    hand_damping=dict(zip(_WUJI_JOINTS, _WUJI_KV)),
    hand_armature=dict(zip(_WUJI_JOINTS, _WUJI_ARMATURE)),
    hand_effort_limit=dict(zip(_WUJI_JOINTS, _WUJI_EFFORT)),
    hand_velocity_limit=_WUJI_VELOCITY_LIMIT,
    hand_friction=_WUJI_FRICTION,
    # The real robot's home pose with joint 7 at 120 deg, palm to the table (flexiv_wuji.FLEXIV_DEFAULT_ARM_POS).
    arm_default_joint_pos={
        "joint1": -2.0944, "joint2": 0.0, "joint3": 0.5236, "joint4": 1.7453,
        "joint5": 2.618, "joint6": 1.3963, "joint7": 2.0944,
    },
    self_collision_filter=_WUJI_SELF_COLLISION_FILTER,
)

# ----------------------------------------------------------------------------
# Flexiv Rizon 4s + right Wuji Hand 2, human-like limits (scripts/flexiv_wuji/flexiv_wuji.py
# RIGHT_HUMAN_LIKE_URDF): FLEXIV_WUJI_HUMAN_LIKE with the hand mirrored. The r_ joints keep the
# left's gains (the vendor MJCF's are the same on both sides); the mount is turned 180 deg about the
# flange axis (RIGHT_MOUNT_RPY) so the palm faces the table at the same arm pose.
# ----------------------------------------------------------------------------

def _right(name: str) -> str:
    return "r_" + name[2:] if name.startswith("l_") else name


# self_collision_pairs.py derives exactly the left's pairs, renamed (tests/test_robot_specs.py reruns it).
_WUJI_RIGHT_SELF_COLLISION_FILTER = {
    _right(a): tuple(_right(b) for b in bs) for a, bs in _WUJI_SELF_COLLISION_FILTER.items()
}

FLEXIV_WUJI_RIGHT_HUMAN_LIKE = RobotSpec(
    name="flexiv_wuji_right_human_like",
    arm_joint_regex=FLEXIV_WUJI.arm_joint_regex,
    hand_joint_regex="r_.*",
    joint_names_canonical=_RIZON_JOINTS + tuple(_right(n) for n in _WUJI_JOINTS),
    # The importer merges flange, r_mount and r_wrist into link7.
    palm_body_name="link7",
    # FLEXIV_WUJI's palm centre mirrored across the hand (canonical frame, palm side flipped); the
    # same point over the table at the default pose. tests/test_robot_specs.py recomputes it.
    palm_center_offset=(0.0018, -0.0149, 0.1948),
    fingertip_body_regex="r_(thumb|index_finger|middle_finger|ring_finger|pinky)_distal",
    fingertip_link_names=tuple(_right(n) for n in FLEXIV_WUJI.fingertip_link_names),
    fingertip_offset=FLEXIV_WUJI.fingertip_offset,
    arm_stiffness=FLEXIV_WUJI.arm_stiffness,
    arm_damping=FLEXIV_WUJI.arm_damping,
    hand_stiffness={_right(k): v for k, v in FLEXIV_WUJI.hand_stiffness.items()},
    hand_damping={_right(k): v for k, v in FLEXIV_WUJI.hand_damping.items()},
    hand_armature={_right(k): v for k, v in FLEXIV_WUJI.hand_armature.items()},
    hand_effort_limit={_right(k): v for k, v in FLEXIV_WUJI.hand_effort_limit.items()},
    hand_velocity_limit=FLEXIV_WUJI.hand_velocity_limit,
    hand_friction=FLEXIV_WUJI.hand_friction,
    arm_default_joint_pos=FLEXIV_WUJI.arm_default_joint_pos,
    self_collision_filter=_WUJI_RIGHT_SELF_COLLISION_FILTER,
)

ROBOT_SPECS: dict[str, RobotSpec] = {
    spec.name: spec for spec in (KUKA_SHARPA, FLEXIV_WUJI_RIGHT_HUMAN_LIKE)
}


def get_robot_spec(name: str) -> RobotSpec:
    try:
        return ROBOT_SPECS[name]
    except KeyError:
        raise ValueError(f"unknown robot {name!r}; one of {sorted(ROBOT_SPECS)}") from None
