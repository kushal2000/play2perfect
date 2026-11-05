"""Continuous Fabrica handoff: preserve physics, freeze the beam, move cap goals."""

import mujoco
from mjlab.utils.lab_api.math import quat_apply, quat_mul

from play2perfect_mjlab.native import load_model


def add_freeze_welds(spec, bundle, metadata):
    """Use the same weld data as the browser's compiled native model."""
    model = load_model(bundle, metadata)
    for index, part in enumerate(("object", "second")):
        anchor = f"freeze_anchor_{index}"
        spec.worldbody.add_body(name=anchor, mocap=True)
        spec.add_equality(
            name=f"freeze_{index}",
            type=mujoco.mjtEq.mjEQ_WELD,
            objtype=mujoco.mjtObj.mjOBJ_BODY,
            name1=f"{part}/{part}",
            name2=anchor,
            data=model.eq_data[index],
            active=False,
            solref=model.eq_solref[index],
            solimp=model.eq_solimp[index],
        )
    return model


def restore_static_excludes(spec, model):
    """Restore native world-weld filtering after MJLab adds mocap root bodies."""
    static_bodies = sorted(
        {
            int(body)
            for geom, body in enumerate(model.geom_bodyid)
            if body != 0
            and model.body_weldid[body] == 0
            and (model.geom_contype[geom] or model.geom_conaffinity[geom])
        }
    )

    def mapped_name(body):
        root = body
        while model.body_parentid[root] != 0:
            root = int(model.body_parentid[root])
        entity = model.body(root).name
        if entity == "base":
            entity = "robot"
        return f"{entity}/{model.body(body).name}"

    for index, first in enumerate(static_bodies):
        for second in static_bodies[index + 1 :]:
            spec.add_exclude(
                name=f"source_static_{first}_{second}",
                bodyname1=mapped_name(first),
                bodyname2=mapped_name(second),
            )


def handoff(env, state, mask):
    """Advance only selected worlds; leave their qpos/qvel/targets unchanged."""
    ids = mask.nonzero(as_tuple=False).flatten()
    if not len(ids):
        return
    data, model = env.sim.data, env.sim.mj_model
    body = model.body("object/object").id
    achieved_pos = data.xpos[ids, body]
    achieved_quat = data.xquat[ids, body]
    target = state["stage_goals"][0, -1]
    target_quat = target[[6, 3, 4, 5]].expand(len(ids), -1).clone()
    target_quat[:, 1:] *= -1
    delta = quat_mul(achieved_quat, target_quat)
    translation = achieved_pos - quat_apply(delta, target[:3].expand(len(ids), -1))
    state["goal_delta_quat"][ids] = delta
    state["goal_delta_pos"][ids] = translation
    anchor = int(model.body("freeze_anchor_0").mocapid[0])
    data.mocap_pos[ids, anchor] = achieved_pos
    data.mocap_quat[ids, anchor] = achieved_quat
    data.eq_active[ids, model.equality("freeze_0").id] = True
    state["stage_index"][ids] = 1
    state["offsets"][ids] = state["stage_offsets"][1]
    state["scales"][ids] = state["stage_scales"][1]
    state["bias"][ids] = state["stage_bias"][1]
    state["object_body"][ids] = model.body("second/second").id
    state["object_address"][ids] = state["stage_addresses"][1]
    state["init_z"][ids] = data.qpos[ids, state["stage_addresses"][1] + 2]
    for key in ("goal_index", "near_steps", "retract", "withdrawn", "lifted"):
        state[key][ids] = 0
    state["closest_kp"][ids] = -1
    state["closest_ft"][ids] = -1
    state["recurrent_reset"][ids] = True
    state["handoff_step"][ids] = env.common_step_counter
    env.episode_length_buf[ids] = 0
