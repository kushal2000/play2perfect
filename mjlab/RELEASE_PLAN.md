# MJLab assembly release plan

This document records the initial release plan. The package and commands are now
being implemented; see README.md for the current qualification status.
It starts from official Play2Perfect `main` at
`621459224dd3618456de273871e507e380d4dedb` (November 3, 2026).
The research implementation already trains assembly policies in MJLab.
The remaining work makes that implementation usable from a clean checkout.

## Recommendation

Add an optional `mjlab/` package to the official Play2Perfect repository through
a focused branch and pull request. Keep its dependencies and commands separate
from Isaac Lab. Merge to `main` after the release checks below pass.

Use a separate public companion repository only if the maintainers prefer to
keep the experimental physics dependencies outside the official release.
Do not publish the existing private research repository wholesale: its history,
experiment archives, machine paths and tool-use work are outside this release.

The current GitHub account has read access, but no push access, to
`kushal2000/play2perfect`. The eventual upstream contribution therefore needs
a public fork and PR, or a maintainer-owned branch. This local branch is based
on the public repository, not the private research history.

## Minimal scope

| Component | First public MJLab release |
|---|---|
| Robot | KUKA iiwa 14 + Sharpa hand |
| Starting policy | Original SimToolReal play checkpoint, SHA-256 `4cb0d09833326de6878ee5681c2b14521823fa616e01b135214a51ffe41c63b9` |
| Screwing | Tested long-leg asset and named fine-step training profile |
| Tight insertion | L-peg fine-tuning, frozen-policy evaluation and rendering |
| Multipart | Two separately trained beam stages and composed evaluation |
| Learner | Existing six-block SAPG, recurrent actor and asymmetric critic |
| User operations | Prepare assets, train, resume, evaluate, record and render |
| Outputs | Checkpoints, reward and success curves, actual physics settings and source hashes |

Full play pretraining, tool-use tasks, cluster launchers and research dashboards
are outside this first release. Flexiv/Wuji already works in upstream Isaac Lab;
it is not yet a validated MJLab robot. Do not imply that both robots work in
both simulators.

## What the review found

- The upstream `rl_games` tree is unchanged from our pinned learner:
  `dc653791ca2b606205c5314c4b9cebd6e5b8ce37`. We can reuse that implementation.
- An initial import-closure inventory contains 43 files, 810,232 bytes, including
  two environment lockfiles. It excludes assets, checkpoints, tests and docs.
  See [release_inventory.json](release_inventory.json). Dynamic imports still
  need validation; this is not a complete distributable manifest.
- The extracted public asset bootstrap now replaces the private-checkout
  builders. It includes two small derived meshes and fetches the other inputs
  from pinned public commits. All five reconstructed bundles match the tested
  compiled geometry and physics arrays exactly.
- The tested runtime pins MJLab `8bb2bc9e`, MuJoCo 3.12, Warp 1.18, SDF candidate
  `230fe303`, and a free-body stabilization patch. Package this explicitly as an
  experimental backend. Stock MJLab is not yet a validated drop-in replacement.
- Upstream now uses 5 mm screwing success tolerance. Our tested long-leg profile
  uses 7.5 mm. Keep the historical profile named and versioned; do not silently
  apply new thresholds to old results.
- Multipart trains separate stages and fixes the first part after handoff in
  the composed demonstration. The prepared stage and composed bundles use the
  same 68.57 g / 85.98 g part masses. An earlier 3×/1× mass label in the research
  notes is not supported by these hash-matched models; use their actual values.
  Publish stage scores separately from joint sequence success.

## Proposed layout

```text
play2perfect/
  isaacsimenvs/                 existing Isaac Lab backend
  evaluation/                  existing task and robot definitions
  rl_games/                    shared, unchanged learner
  mjlab/
    pyproject.toml             installable backend package
    environments/{cpu,cuda}/   pinned environments and lockfiles
    src/play2perfect_mjlab/    managers, policy bridge, recording
    configs/                  four stage recipes and physics profiles
    scripts/                  asset preparation and small CLI entry points
    patches/                  explicit, hash-checked Warp patch
    tests/                    portable regression and resume checks
    README.md                 commands, support matrix and limitations
```

Initially preserve the tested task definitions. Share simulator-independent
definitions with current upstream only after a differential check. Avoid a
large Isaac Lab refactor in the same PR.

## Work sequence and estimate

These are engineering estimates, not measured completion times. Tasks overlap.
No full retraining is needed merely to validate the packaging.

| Step | Deliverable | Estimated focused effort |
|---|---|---|
| 1. Extract | Select assembly core; remove workstation paths and unused trainer dependencies | 2–4 hours |
| 2. Make inputs public | Reproducible assets, checkpoint download revisions, hashes and notices | 4–8 hours; main uncertainty |
| 3. Package commands | Installable CPU/CUDA profiles; prepare/train/eval/render commands; resume | 3–5 hours |
| 4. Verify clean checkout | Install, all four stage smoke checks, checkpoint resume and one GPU training check | 4–8 hours plus queue/install time |
| 5. Release documentation | Task matrix, recipe differences, checkpoint ancestry and example outputs | 2–4 hours |

A reviewable extraction is plausible within a day. Budget roughly 2–3 focused
days for a small, tested public release. New physics failures, upstream version
migration or Flexiv/Wuji porting are additional work.

## Release acceptance

1. A fresh clone needs no private repository, Stanford mount or manual
   `PYTHONPATH`. Commands run from documented locations.
2. Assets and starting weights download by pinned revision and verify their
   hashes. Third-party notices travel with redistributed inputs.
3. CPU checks construct screwing, tight insertion and both beam stages; load
   the foundation; run a learner update; and restore the saved learner state.
4. A bounded one-GPU run advances training and writes readable checkpoints and
   metrics. Existing measured task results remain separate from this smoke test.
5. Frozen evaluations retain every declared start, report unfinished episodes,
   verify actual physics, and distinguish goal completion from clean withdrawal.
6. Rendering shows the physical arm and recorded state sequence. It does not
   hide failed runs or imply real-robot validation.
7. CI runs the portable tests and formatting checks. GPU reproducibility has a
   documented command and result; a GPU is not required for every PR.
8. The release diff contains only the selected code, docs, small fixtures and
   notices. Large checkpoints and videos use separate artifact hosting.

## Suggested conference wording

After the code and checks are public:

> Play2Perfect assembly fine-tuning supports Isaac Lab and an experimental
> MJLab backend. Our MJLab release includes KUKA/Sharpa screwing, tight insertion
> and two-stage beam assembly.

Avoid “any simulator” or “any robot.” The framework is portable, but each
backend and robot still needs its own physics and policy validation.

## Sources

- [Current public Play2Perfect](https://github.com/kushal2000/play2perfect/tree/621459224dd3618456de273871e507e380d4dedb)
- [Robot specifications](https://github.com/kushal2000/play2perfect/blob/621459224dd3618456de273871e507e380d4dedb/isaacsimenvs/tasks/play/robots.py)
- [Current task tolerances](https://github.com/kushal2000/play2perfect/blob/621459224dd3618456de273871e507e380d4dedb/evaluation/problems/__init__.py)
- [Public videos](https://mjlab-dexterous-showcase-tyler-20261102.tylergwlum.chatgpt.site/presentation.html#multipart)
