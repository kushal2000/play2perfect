"""Register the everyday-object insertion problems.

  fork.tol1mm     YCB fork into a flatware holder, handle first, 1 mm/side
  plug.tol0p5mm   USB-C wall adapter into a power-strip socket, 0.5 mm/side

All poses are in the receptive frame (origin at the receptacle centroid) and are
derived from the meshes: the pre-insert pose puts the part's lowest point exactly
on the opening (touching, zero penetration) and the final pose is the seated
depth. Both objects are authored long-axis-X with the entry end at -X, so the
insertion quaternion is the peg's -90 deg about Y.
"""

from evaluation.objects import get_repo_root_dir
from evaluation.problems import PROBLEM_REGISTRY, Problem

_ASSETS = get_repo_root_dir() / "assets"
_INSERT_QUAT = (0.0, -0.70710678, 0.0, 0.70710678)

# Fork -> flatware holder. Holder block is 85 mm tall (hole_z_offset = half).
# The seated pose is 78 mm below pre-insert and is floor-limited, so it does
# not depend on the clearance.
_FORK_HOLE_Z_OFFSET = 0.0425
_FORK_PRE_INSERT_Z = 0.141277
_FORK_FINAL_Z = 0.063277
_FORK_TOLERANCES = ("1mm",)

# Plug -> power-strip socket. Socket block is 30 mm tall. Final pose has the
# plug body flush with the socket face, blades 15.9 mm into the cavity.
_PLUG_HOLE_Z_OFFSET = 0.0150
_PLUG_PRE_INSERT_Z = 0.044400
_PLUG_FINAL_Z = 0.028500
_PLUG_TOLERANCES = ("0p5mm",)


def _register(prefix, object_name, urdf_fmt, tags, pre_z, final_z, z_offset) -> None:
    for tag in tags:
        urdf_rel = urdf_fmt.format(tag=tag)
        if not (_ASSETS / urdf_rel).exists():
            continue
        name = f"{prefix}.tol{tag}"
        PROBLEM_REGISTRY[name] = Problem(
            name=name,
            insertion_object_name=object_name,
            receptive_urdf=urdf_rel,
            insert_pose_rel_receptive=(
                (0.0, 0.0, pre_z, *_INSERT_QUAT),
                (0.0, 0.0, final_z, *_INSERT_QUAT),
            ),
            hole_z_offset=z_offset,
            insertion_direction=(0.0, 0.0, -1.0),
        )


_register("fork", "ycb_fork", "urdf/plug_fork/holder/holder_tol{tag}.urdf",
          _FORK_TOLERANCES, _FORK_PRE_INSERT_Z, _FORK_FINAL_Z, _FORK_HOLE_Z_OFFSET)
_register("plug", "usbc_plug", "urdf/plug_fork/socket/socket_tol{tag}.urdf",
          _PLUG_TOLERANCES, _PLUG_PRE_INSERT_Z, _PLUG_FINAL_Z, _PLUG_HOLE_Z_OFFSET)
