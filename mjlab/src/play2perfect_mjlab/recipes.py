"""Named historical training recipes; simulator changes require a new name."""

RECIPES = {
    "long_leg_fine_v1": {
        "training_preset": "isaac_lab_long_leg",
        "physics_dt": 1 / 600,
        "decimation": 10,
        "sdf_initpoints": 10,
        "nconmax": 128,
        "num_envs": 1536,
        "minibatch_size": 12288,
        "warp_overlay": ".cache/mujoco_warp_sdf_gyro",
    },
    "tight_stabilized_v1": {
        "training_preset": "isaac_lab_tight",
        "physics_dt": 1 / 120,
        "decimation": 2,
        "sdf_initpoints": 10,
        "nconmax": 128,
        "num_envs": 1536,
        "minibatch_size": 12288,
        "warp_overlay": ".cache/mujoco_warp_sdf_gyro",
    },
    "beam_step1_v1": {
        "training_preset": "source_beam_step1",
        "physics_dt": 1 / 120,
        "decimation": 2,
        "sdf_initpoints": 50,
        "nconmax": 512,
        "num_envs": 768,
        "minibatch_size": 6144,
    },
    "beam_step2_v1": {
        "training_preset": "source_beam_step2",
        "physics_dt": 1 / 120,
        "decimation": 2,
        "sdf_initpoints": 50,
        "nconmax": 512,
        "num_envs": 768,
        "minibatch_size": 6144,
    },
}


def training_arguments(name):
    settings = {
        "checkpoint": "data/pretrained_policy/model.pth",
        "horizon": 16,
        "njmax": 2048,
        "epa_horizon": 64,
        "solver_iterations": 100,
        "ls_iterations": 50,
        "seed": 0,
        **RECIPES[name],
    }
    return [
        item
        for key, value in settings.items()
        for item in ("--" + key.replace("_", "-"), str(value))
    ]
