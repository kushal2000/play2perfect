"""Register the everyday-object inserters (fork, plug) in NAME_TO_OBJECT.

``scale`` is the object's metric bounding box passed through
``rescale_by_factor(..., 25)`` -- the same convention as the peg. It is fed to
the policy as an object-size observation, not applied to the mesh.
"""

from evaluation.objects import NAME_TO_OBJECT, Object, get_repo_root_dir, rescale_by_factor

ASSETS_DIR = get_repo_root_dir() / "assets" / "urdf" / "plug_fork"

PLUG_FORK_NAME_TO_OBJECT = {
    # YCB 030_fork, a real scan at true scale (197.6 x 27.1 x 15.9 mm, 55 g).
    # Visual is the textured scan; collision is 9 CoACD hulls.
    "ycb_fork": Object(
        urdf_path=ASSETS_DIR / "fork" / "fork.urdf",
        scale=rescale_by_factor((0.198, 0.027, 0.016), factor=25),
        need_vhacd=False,
    ),
    # USB-C wall adapter (41.5 x 42.5 x 27 mm, 58.5 g) with 1.5 mm NEMA 1-15P
    # blades. Compact rather than elongated, and a two-feature insertion: both
    # blades enter together. Collision is 3 exact boxes.
    "usbc_plug": Object(
        urdf_path=ASSETS_DIR / "plug" / "plug.urdf",
        scale=rescale_by_factor((0.0429, 0.0425, 0.0415), factor=25),
        need_vhacd=False,
    ),
}

NAME_TO_OBJECT.update(PLUG_FORK_NAME_TO_OBJECT)
