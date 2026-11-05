# Sources and notices

This package follows the parent Play2Perfect MIT license. Its bundled converted
scene descriptions and long-leg geometry retain that license. See `../LICENSE`.

## Geometry and task inputs

The asset catalog is
`src/play2perfect_mjlab/resources/bundles/catalog.json`. It records an immutable
public URL, byte count and SHA-256 for each downloaded mesh. The downloader
checks all three before exposing a prepared scene.

- Play2Perfect source: `kushal2000/play2perfect`, commit
  `ff2fc62873f6c6cd93164123c8358e9ce2d65c9b`, MIT.
- Browser reference: `play2perfect/play2perfect.github.io`, commit
  `a61913d2720a16e9062114555823fc4af7c4ed6c`, MIT.
- Converted scenes preserve the tested robot, actuator order, collision filters,
  object inertia, task goals and starting states. They are frozen conversion
  outputs, not a new general-purpose URDF converter.
- The 225 mm long-leg variant contains two derived meshes in
  `resources/bundles/derived/`. The threaded mesh is 1,161,925 bytes, SHA-256
  `f543365f159a073275da36db4a3ae8ded898343c4f2bb02b7b210e27df716e47`.
  The convex body is 423 bytes, SHA-256
  `4c7bb244b48ce6b0a757fdc1affc2f193b48deb6ac4d6dd051ada5b11cda9ee0`.
  These are the exact historical experiment inputs. The source URDF hash was
  `28c8d7423fbc956ff1345d525fcbda3aaed962ec4412a06ca7fa87c9640af45b`.

The first release distributes these small derived meshes directly. It does not
require the private development checkout that originally generated them. The
manifest's `source_long_leg_225mm` name describes total leg length; the original
asset directory used the name `leg4_200mm`.

## Foundation policy

Weights are downloaded from the public upstream Hugging Face repository
[`Kushal20/play2perfect-checkpoints`](https://huggingface.co/Kushal20/play2perfect-checkpoints),
revision `91537ffa9584dbe74548623f2cc88e0922e9cbe3`, path `play/model.pth`.
They are not included in Git or in this Python distribution. The exact hash
appears in `play2perfect_mjlab/foundation.py`. This is the SimToolReal play
foundation, before task-specific Play2Perfect fine-tuning.

## Physics dependencies

The environment lockfiles retain the tested MJLab, MuJoCo, MuJoCo Warp, Warp,
Torch and learner revisions. These dependencies remain separate packages with
their own licenses. In particular, this release uses an experimental mesh-SDF
candidate; it does not claim compatibility with arbitrary current MJLab versions.

The included free-body stabilization patch derives from MuJoCo Warp commits
`02540c86ca6145d35f45942788199f8f53144eaa` and
`d8a241ef1643295ebb85bfdfdf9171360bfb2c0e`, applied to
`230fe3034a834cb10dd9b450218f962290f3eaff`. It is covered by the accompanying
`mujoco-warp-LICENSE` (Apache 2.0). The patch and resulting Python files are
hash-checked by `scripts/prepare_warp_stabilization.py`.

The optional overlay affects only the launched process. Preparation never
edits the installed dependency package.
