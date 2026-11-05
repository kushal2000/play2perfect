"""Extract authored static world geometry without duplicating body assets."""

import xml.etree.ElementTree as ET
from pathlib import Path


def world_geometry_xml(bundle: Path) -> str | None:
    root = ET.fromstring((bundle / "scene.xml").read_text())
    world = root.find("worldbody")
    for child in list(world):
        if child.tag != "geom":
            world.remove(child)
    if not len(world):
        return None
    for tag in ("actuator", "keyframe", "equality", "sensor", "contact", "tendon"):
        for element in root.findall(tag):
            root.remove(element)
    used = {node.get("mesh") for node in world.iter() if node.get("mesh")}
    used.update(
        node.get("mesh")
        for node in root.findall(".//default//geom")
        if node.get("mesh")
    )
    for asset in root.findall("asset"):
        for mesh in asset.findall("mesh"):
            if mesh.get("name") not in used:
                asset.remove(mesh)
    for asset in root.findall(".//asset/*"):
        if asset.get("file"):
            asset.set("file", str((bundle / asset.get("file")).resolve()))
    return ET.tostring(root).decode()
