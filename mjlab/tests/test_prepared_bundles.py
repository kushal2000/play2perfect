"""Compile prepared public scenes and retain robot and multipart physics."""

from pathlib import Path

import mujoco
import numpy as np
import pytest

from play2perfect_mjlab.bundles import catalog, verify_bundle


def prepared_model(name):
    definition = catalog()["bundles"][name]
    root = Path(__file__).resolve().parents[1] / "data" / definition["destination"]
    if not root.exists():
        pytest.skip("Run p2p-mjlab-assets before compiled-scene checks")
    verify_bundle(name, root)
    return mujoco.MjModel.from_xml_path(str(root / "scene.xml"))


@pytest.mark.parametrize("name", list(catalog()["bundles"]))
def test_prepared_scene_has_physical_arm_and_complete_actuation(name):
    model = prepared_model(name)
    assert [model.nq, model.nv, model.nu] == catalog()["bundles"][name][
        "model_dimensions"
    ]
    for index in range(7):
        assert model.joint(index).name == f"joint{index + 1}"
        assert model.body_mass[model.jnt_bodyid[index]] > 0
        assert index in model.actuator_trnid[:, 0]
    assert np.all(np.isfinite(model.body_inertia))


def test_composed_parts_keep_the_separate_stage_mass_and_inertia():
    composed = prepared_model("multipart")
    for stage, body in (("beam_step1", "object"), ("beam_step2", "second")):
        reference = prepared_model(stage)
        before = reference.body("object").id
        after = composed.body(body).id
        for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
            np.testing.assert_array_equal(
                getattr(reference, field)[before], getattr(composed, field)[after]
            )
