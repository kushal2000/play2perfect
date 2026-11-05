"""A scene split must retain the floor as a static collision surface."""

import mujoco

from play2perfect_mjlab.scene_xml import world_geometry_xml


def test_split_floor_stops_a_falling_body(tmp_path):
    (tmp_path / "scene.xml").write_text(
        """<mujoco><asset>
        <material name="floor_mat" rgba="0.2 0.3 0.4 1"/>
        <mesh name="unused" vertex="0 0 0 1 0 0 0 1 0 0 0 1"/>
        </asset><worldbody>
        <geom name="floor" type="plane" pos="0 0 0.2" size="1 1 0.05"
          material="floor_mat" friction="0.7 0.003 0.0002"/>
        <body name="removed"><freejoint/><geom size="0.02"/></body>
        </worldbody></mujoco>"""
    )
    static = mujoco.MjSpec.from_string(world_geometry_xml(tmp_path))
    scene = mujoco.MjSpec.from_string(
        """<mujoco><worldbody><body pos="0 0 0.4"><freejoint/>
        <geom type="sphere" size="0.05"/></body></worldbody></mujoco>"""
    )
    scene.attach(static, prefix="source_world/", frame=scene.worldbody.add_frame())
    model = scene.compile()
    floor = model.geom("source_world/floor")
    assert model.geom_bodyid[floor.id] == 0
    assert model.nmesh == 0 and model.nv == 6
    assert tuple(floor.friction) == (0.7, 0.003, 0.0002)
    data = mujoco.MjData(model)
    for _ in range(round(1 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    assert 0.245 < data.qpos[2] < 0.255


def test_body_only_scene_has_no_extra_world_geometry(tmp_path):
    (tmp_path / "scene.xml").write_text(
        '<mujoco><worldbody><body><geom size="0.1"/></body></worldbody></mujoco>'
    )
    assert world_geometry_xml(tmp_path) is None
