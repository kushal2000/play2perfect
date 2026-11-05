# Minimal release validation

The [November 5 review](2026-11-05/review.json) rechecks all 32 tests, formatting,
the wheel, documentation links and public video hashes. It updates the Long demo
to 1B steps and removes historical planning material. Runtime code, dependency
locks and recipes are unchanged from the qualification below.

Checked November 4, 2026 on Linux, Python 3.11 and one RTX 4090. The tested
learner implementation is commit `9ac320b`. Later changes add documentation,
reference plots, and explicit evaluation history selection. Both environments use
their committed dependency locks. Public
asset and checkpoint downloads succeed without the private research checkout.
Local environment installation reuses a package cache through hardlinks.

| Check | Result |
|---|---|
| Public prepared assets | All five bundles verify; 24 compiled arrays per bundle match the research reference |
| Public play foundation | Exact 413,895,044-byte file and SHA-256 match |
| Warp stabilization overlay | All 76 Python source hashes match the declared patch |
| Four CPU recipes | Real update, exact restoration, then another update: 24 → 48 transitions |
| Four CUDA recipes | Same checks, including CUDA random state and AMP scaler |
| Physics monitoring | Zero invalid resets and zero capacity overflow in every short check |
| Evaluation | Long leg, insertion and composed multipart run for 60 controls |
| Dense recording | Exact compiled model and every 60 Hz control state saved |
| Native rendering | Long and multipart MP4s render at 60 fps with no integration or interpolation |
| Reward plots | Initial and resumed metrics combine with the saved training clock |
| Automated tests | 32 pass; lint and formatting pass |
| Source-history evaluation | Six CPU controls / 60 substeps, 10/3/3 queues, finite states and no overflow |
| Reference plotting | All four full scalar histories verify and render on CPU |
| GitHub CI | Fresh install, prepared assets, tests and wheel build pass |

The last pre-review clean-install CI passes at `a31211e`:
[41908809158](https://github.com/tylerlum/play2perfect/actions/runs/41908809158).
See [current release-branch runs](https://github.com/tylerlum/play2perfect/actions/workflows/mjlab.yml?query=branch%3Arelease%2Fmjlab-minimal-assembly)
for checks on later documentation and media revisions.
The workflow itself is included at
[`../../.github/workflows/mjlab.yml`](../../.github/workflows/mjlab.yml).

Machine-readable results are in [2026-11-04](2026-11-04/summary.json). Per-recipe
reports retain exact restore checks, actual physics settings and checkpoint
hashes. Small generated test checkpoints are disposable and are not distributed.

The [history check](2026-11-04/evaluation_history.json) records the new evaluator's
actual queue and physics settings. Tests confirm that history selection changes
only evaluation queues, preserving immediate feedback and training settings.
Download tests also cover transient HTTP failures, interrupted partial transfers,
bounded retries, and preservation of corrupt existing inputs for inspection.

These are software qualification checks. The tiny learner runs and one-second
evaluations do **not** measure task success or reproduce a learning curve. The
short multipart test also exercises a compiled handoff preflight; it does not
complete a natural policy-driven handoff. Longer research runs use the same
named physics recipes, but this release does not ship those checkpoints or
claim a fresh full-scale training reproduction. Only KUKA/Sharpa is qualified.
