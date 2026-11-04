"""Observe every physics substep of the unchanged released-policy collector.

Keep the collector and task packages unchanged. Save capacity flags and the
first nonfinite physical state before an automatic reset could erase it.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

from play2perfect_mjlab.assets import sha256
from play2perfect_mjlab.source_sapg import PhysicsRolloutMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--health", type=Path, required=True)
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    collector = Path(__file__).with_name("evaluate_released.py")
    spec = importlib.util.spec_from_file_location("released_collector", collector)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args.health.parent.mkdir(parents=True, exist_ok=True)
    created = []

    class FiniteMonitor(PhysicsRolloutMonitor):
        def __init__(self, sim):
            super().__init__(sim)
            self.original_forward = sim.forward
            self.forward_refreshes = 0
            sim.forward = self.forward

        def check_finite(self):
            data = self.sim.data
            finite = torch.isfinite(data.qpos).all(-1) & torch.isfinite(data.qvel).all(
                -1
            )
            if not finite.all():
                np.savez_compressed(
                    args.health.with_suffix(".npz"),
                    **{
                        key: getattr(data, key).cpu().numpy()
                        for key in ["qpos", "qvel", "qacc", "ctrl", "qacc_warmstart"]
                    },
                )
                raise FloatingPointError(
                    f"Nonfinite physical state after substep{self.substeps}: "
                    f"worlds{(~finite).nonzero().flatten().tolist()}"
                )

        def step(self):
            super().step()
            self.check_finite()

        def forward(self):
            self.original_forward()
            data = self.sim.wp_data
            sample = torch.stack(
                (
                    wp.to_torch(data.nacon)[0],
                    wp.to_torch(data.nefc).max(),
                    wp.to_torch(data.solver_niter).max(),
                )
            )
            torch.maximum(self.peaks, sample, out=self.peaks)
            overflow = wp.to_torch(data.overflow)
            self.flags |= ((overflow[:, None] & self.flag_bits) != 0).any(0)
            self.forward_refreshes += 1
            self.check_finite()
            self.validate()

        def snapshot(self):
            return super().snapshot() | dict(
                scope="Peak over every physics substep and forward refresh",
                forward_refreshes=self.forward_refreshes,
            )

    class MonitoredEnv(module.ManagerBasedRlEnv):
        def __init__(self, *positional, **keywords):
            super().__init__(*positional, **keywords)
            self.physical_monitor = FiniteMonitor(self.sim)
            self.monitor_closed = False
            created.append(self)

        def step(self, action):
            result = super().step(action)
            self.physical_monitor.validate()
            return result

        def close(self):
            if self.monitor_closed:
                return
            report = dict(
                scope=__doc__,
                physics_peak=self.physical_monitor.snapshot(),
                wrapper_sha256=sha256(Path(__file__)),
                collector_sha256=sha256(collector),
                model_options=dict(
                    solver_enum=int(self.sim.mj_model.opt.solver),
                    warp_solver_enum=int(self.sim.wp_model.opt.solver),
                    iterations=int(self.sim.mj_model.opt.iterations),
                    ls_iterations=int(self.sim.mj_model.opt.ls_iterations),
                    timestep=float(self.sim.mj_model.opt.timestep),
                    sdf_initpoints=int(self.sim.mj_model.opt.sdf_initpoints),
                ),
                physical_parameters_changed=False,
            )
            args.health.write_text(json.dumps(report, indent=2) + "\n")
            try:
                super().close()
            finally:
                self.monitor_closed = True

    module.ManagerBasedRlEnv = MonitoredEnv
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [str(collector), *arguments]
    try:
        module.main()
    finally:
        for env in created:
            env.close()


if __name__ == "__main__":
    main()
