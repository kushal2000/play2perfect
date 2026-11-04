"""Split the validated website scene into MJLab entities, preserving XML gains."""

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.scene import SceneCfg

from play2perfect_mjlab.scene_xml import world_geometry_xml


def scene_config(
    bundle: Path,
    metadata=None,
    *,
    num_envs=1,
    env_spacing=1.2,
    spec_fn=None,
    preserve_world_geometry=True,
):
    """Build entities and retain authored world geoms as static geometry.

    World planes must attach directly to the scene. An Entity would wrap them
    in a movable mocap root. The opt-out supports historical parity diagnostics.
    """
    world_xml = world_geometry_xml(bundle)

    def configure(spec):
        if preserve_world_geometry and world_xml is not None:
            frame = spec.worldbody.add_frame()
            spec.attach(
                mujoco.MjSpec.from_string(world_xml),
                prefix="source_world/",
                frame=frame,
            )
        if spec_fn is not None:
            spec_fn(spec)

    return SceneCfg(
        entities=scene_entities(bundle, metadata),
        num_envs=num_envs,
        env_spacing=env_spacing,
        spec_fn=configure,
    )


def scene_entities(bundle: Path, metadata=None):
    """One entity per root body; robot joint order remains the policy order."""
    if metadata is None:
        metadata = json.loads((bundle / "manifest.json").read_text())
    root = ET.fromstring((bundle / "scene.xml").read_text())
    model = mujoco.MjModel.from_xml_string(
        ET.tostring(root).decode(),
        {name: (bundle / name).read_bytes() for name in metadata["files"]},
    )
    names = [model.joint(i).name for i in range(29)]
    world = root.find("worldbody")
    # Cross-entity contacts are restored by the task's scene callback. MjSpec
    # attachment drops references outside the attached body. Retaining the full
    # contact library in every grain nevertheless creates gigabytes of XML and
    # millions of temporary specs before that happens. Copy only local entries.
    contact_entries = [list(contact) for contact in root.findall("contact")]
    template = copy.deepcopy(root)
    for contact in template.findall("contact"):
        for entry in list(contact):
            contact.remove(entry)
    entities = {}
    for body in world.findall("body"):
        body_name = body.get("name")
        name = "robot" if body_name == "base" else body_name
        if name is None:
            raise ValueError("Expected named root bodies")
        xml = copy.deepcopy(template)
        selected_world = xml.find("worldbody")
        for child in list(selected_world):
            if child.tag != "body" or child.get("name") != body_name:
                selected_world.remove(child)
        for tag in ("keyframe", "equality", "sensor"):
            for element in xml.findall(tag):
                xml.remove(element)
        if name != "robot":
            for actuators in xml.findall("actuator"):
                xml.remove(actuators)
        else:
            joints = {joint.get("name") for joint in selected_world.iter("joint")}
            for actuators in xml.findall("actuator"):
                for actuator in list(actuators):
                    if actuator.get("joint") not in joints:
                        actuators.remove(actuator)
        body_names = {node.get("name") for node in selected_world.iter("body")}
        geom_names = {node.get("name") for node in selected_world.iter("geom")}
        body_names.discard(None)
        geom_names.discard(None)
        for contact, entries in zip(
            xml.findall("contact"), contact_entries, strict=True
        ):
            for entry in entries:
                if entry.tag == "pair":
                    keys, selected_names = ("geom1", "geom2"), geom_names
                elif entry.tag == "exclude":
                    keys, selected_names = ("body1", "body2"), body_names
                else:
                    raise ValueError(f"Unsupported contact entry: {entry.tag}")
                if all(entry.get(key) in selected_names for key in keys):
                    contact.append(copy.deepcopy(entry))
        # MjSpec.attach prefixes every declared mesh, including unused ones.
        # Retaining the whole scene library in each entity duplicates SDF/BVH
        # storage and changes the backend's mesh indexing unnecessarily.
        used_meshes = {
            node.get("mesh") for node in selected_world.iter() if node.get("mesh")
        }
        used_meshes.update(
            node.get("mesh")
            for node in xml.findall(".//default//geom")
            if node.get("mesh")
        )
        for asset in xml.findall("asset"):
            for mesh in asset.findall("mesh"):
                if mesh.get("name") not in used_meshes:
                    asset.remove(mesh)
        # Every entity also prefixes its material/texture library. Hundreds of
        # grains otherwise exceed MuJoCo's rendering material limit. Defaults
        # may reference materials even when the selected body does not use them.
        used_materials = {
            node.get("material") for node in xml.iter() if node.get("material")
        }
        for asset in xml.findall("asset"):
            for material in asset.findall("material"):
                if material.get("name") not in used_materials:
                    asset.remove(material)
        used_textures = {
            node.get("texture") for node in xml.iter() if node.get("texture")
        }
        for asset in xml.findall("asset"):
            for texture in asset.findall("texture"):
                # The world scene supplies the skybox once. Entities need only
                # textures referenced by their retained materials or defaults.
                if texture.get("name") not in used_textures:
                    asset.remove(texture)
        # Absolute asset paths keep factory closures small and avoid rereading
        # every unrelated mesh for each entity. MjSpec loads referenced assets.
        for mesh in xml.findall(".//asset/mesh"):
            mesh.set("file", str((bundle / mesh.get("file")).resolve()))
        pos = tuple(float(v) for v in body.get("pos", "0 0 0").split())
        rot = tuple(float(v) for v in body.get("quat", "1 0 0 0").split())
        body_id = model.body(body_name).id
        joint_id = model.body_jntadr[body_id]
        if joint_id >= 0 and model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            address = model.jnt_qposadr[joint_id]
            pose = np.asarray(metadata["initial_qpos"])[address : address + 7]
            pos, rot = tuple(pose[:3]), tuple(pose[3:])
        # MJLab wraps fixed bodies in a mocap parent. Put the root transform
        # in InitialStateCfg only, or the robot/table translation is applied twice.
        selected_body = selected_world.find("body")
        selected_body.set("pos", "0 0 0")
        selected_body.set("quat", "1 0 0 0")
        xml_text = ET.tostring(xml).decode()
        initial = EntityCfg.InitialStateCfg(pos=pos, rot=rot)
        articulation = None
        if name == "robot":
            initial.joint_pos = dict(
                zip(names, map(float, metadata["initial_qpos"][:29]), strict=True)
            )
            initial.joint_vel = dict(
                zip(
                    names,
                    map(float, metadata.get("initial_qvel", [0.0] * 29)[:29]),
                    strict=True,
                )
            )
            articulation = EntityArticulationInfoCfg(
                actuators=(
                    XmlActuatorCfg(target_names_expr=(".*",), command_field="position"),
                )
            )
        entities[name] = EntityCfg(
            spec_fn=lambda text=xml_text: mujoco.MjSpec.from_string(text),
            init_state=initial,
            articulation=articulation,
        )
    return entities
