"""Import fixed-link assembly assets, preserving each convex/SDF collision part.

Adapted from the MIT Play2Perfect native reference's mujoco_fabrica.add_part.
Only fixed zero-origin joints are supported; reject other assets explicitly.
"""

import math
import xml.etree.ElementTree as ET

import numpy as np


def rpy_quaternion(text):
    r, p, y = np.fromstring(text, sep=" ") / 2
    cr, sr, cp, sp, cy, sy = (
        math.cos(r),
        math.sin(r),
        math.cos(p),
        math.sin(p),
        math.cos(y),
        math.sin(y),
    )
    return [
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ]


def pose_attributes(pose):
    return {
        "pos": " ".join(map(str, pose[:3])),
        "quat": " ".join(map(str, np.asarray(pose)[[6, 3, 4, 5]])),
    }


def add_rigid_asset(root, name, path, pose, dynamic, color):
    urdf = ET.parse(path).getroot()
    for joint in urdf.findall("joint"):
        origin = joint.find("origin")
        if joint.get("type") != "fixed" or (
            origin is not None
            and (
                not np.allclose(np.fromstring(origin.get("xyz", "0 0 0"), sep=" "), 0)
                or not np.allclose(
                    np.fromstring(origin.get("rpy", "0 0 0"), sep=" "), 0
                )
            )
        ):
            raise ValueError("Rigid assembly assets require zero-origin fixed joints")
    body = ET.SubElement(
        root.find("worldbody"), "body", name=name, **pose_attributes(pose)
    )
    if dynamic:
        ET.SubElement(body, "freejoint", name=name + "_free_joint")
        inertials = urdf.findall("link/inertial")
        if len(inertials) != 1:
            raise ValueError(
                "Expected one explicit inertia for the dynamic rigid asset"
            )
        inertial = inertials[0]
        origin, inertia = inertial.find("origin"), inertial.find("inertia")
        if not np.allclose(np.fromstring(origin.get("rpy", "0 0 0"), sep=" "), 0):
            raise ValueError("Rotated inertia frames need explicit tensor conversion")
        ET.SubElement(
            body,
            "inertial",
            mass=inertial.find("mass").get("value"),
            pos=origin.get("xyz", "0 0 0"),
            fullinertia=" ".join(
                inertia.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
            ),
        )
    for index, collision in enumerate(urdf.findall("link/collision")):
        origin = collision.find("origin")
        attrs = dict(
            name=f"{name}_collision_{index}",
            density="0",
            rgba=color,
            pos=origin.get("xyz", "0 0 0") if origin is not None else "0 0 0",
            quat=" ".join(
                map(
                    str,
                    rpy_quaternion(
                        origin.get("rpy", "0 0 0") if origin is not None else "0 0 0"
                    ),
                )
            ),
            friction=".5 .001 .00001",
            solref=".005 1",
            solimp=".95 .99 .0001",
            contype="1",
            conaffinity="1",
            group="0",
            condim="3",
        )
        geometry = collision.find("geometry")
        mesh, box = geometry.find("mesh"), geometry.find("box")
        if mesh is not None:
            mesh_name = f"released_{name}_mesh_{index}"
            filename = (path.parent / mesh.get("filename")).resolve()
            ET.SubElement(
                root.find("asset"),
                "mesh",
                name=mesh_name,
                file=str(filename),
                scale=mesh.get("scale", "1 1 1"),
            )
            attrs.update(
                type="sdf" if collision.find("sdf") is not None else "mesh",
                mesh=mesh_name,
            )
        elif box is not None:
            attrs.update(
                type="box",
                size=" ".join(map(str, np.fromstring(box.get("size"), sep=" ") / 2)),
            )
        elif geometry.find("cylinder") is not None:
            cylinder = geometry.find("cylinder")
            attrs.update(
                type="cylinder",
                size=f"{cylinder.get('radius')} {float(cylinder.get('length')) / 2}",
            )
        elif geometry.find("sphere") is not None:
            attrs.update(type="sphere", size=geometry.find("sphere").get("radius"))
        else:
            raise ValueError("Unsupported released URDF collision primitive")
        ET.SubElement(body, "geom", **attrs)
    return body
