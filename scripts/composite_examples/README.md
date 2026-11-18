# Three composite GEN3C examples

Activate the configured GPU environment from
[GCP_COMPOSITE_INFERENCE.md](../../docs/GCP_COMPOSITE_INFERENCE.md), then run:

```bash
conda activate cosmos-predict1
python scripts/composite_examples/run_duck.py --output-dir outputs/duck_example
python scripts/composite_examples/run_cat.py --output-dir outputs/cat_example
python scripts/composite_examples/run_creature.py --output-dir outputs/creature_example
```

Each script owns its example's preparation and generation pipeline. They accept
`--prepare-only`, `--generate-only`, and `--output-dir`. For example:

```bash
.venv/bin/python scripts/composite_examples/run_duck.py \
  --prepare-only --output-dir outputs/duck_example
```

Each Python script runs SAM segmentation, labeled source/cache persistence,
placement, 24-frame CPU previews, conversion to a 121-camera trajectory at
704 × 1280, and GEN3C diffusion for its example. They use the existing
[MVP driver](../evaluate_real_cache_mvp.py) and follow
[REAL_CACHE_MVP_EVALUATION.md](../../docs/REAL_CACHE_MVP_EVALUATION.md) and the
GCP guide. They expect dependencies, example batches, SAM tiny weights, and GEN3C
checkpoints to be installed already. Each uses all four offload flags and disables
guardrails, matching the GCP guide's inference command. Downloads and environment
installation remain the setup steps in those guides.

| Case | Composition | Placement in 320 × 240 preview |
| --- | --- | --- |
| `duck` | Duck from `batch_0000` in `batch_0001` | Box includes red feet; height 120 pixels (half the image); lowest selected pixels anchored to the measured water surface at (160, 204), with no foreground depth offset |
| `cat` | Cat from `batch_0002` in `batch_0007`, plus candle from `batch_0008` | Smaller cat on tabletop at man's anatomical right / image-left; base (115, 216), height 100; depth multiplier 0.95 puts it in front of the man and behind the monitor; handheld camera replaced with the lit candle and wax puddle |
| `creature` | Creature from `batch_0008` in the bed of `batch_0002` | Select creature rather than candle; base (113, 178), height 145; segment and exclude the original cat |

The half-height duck scales to approximately 352 pixels in the output. Imported
objects default to textured planes at median source depth, prepared and recorded
before labeling. This preserves the first-frame silhouette: raw sample depth
relief can fold the duck's feet into its body and fragment the creature after
placement. Destination RGB-D geometry stays intact for occlusion. These are
static source caches with a forward and sideways dolly toward the scene center; source motion is not
replayed and the planes do not reconstruct full object shape. Set
`IMPORT_DEPTH_MODE=rgbd` during preparation to keep the imported source depth
instead. SAM determines the silhouette, and depth
determines occlusion. Check the previews for feet, ears, and monitor overlap.
Removing the bed cat leaves uncovered geometry. The seed builder now fills its
RGB background with CPU LaMa; source caches keep that geometry unknown. Set up
the TorchScript checkpoint using [LaMa setup](../../docs/LAMA_SEED_SETUP.md),
with no new packages. The camera-removal cat preparation saves a source-grid
`seed_removal_mask.png` for holes whose original IDs were discarded; rerun
preparation if an older result lacks this mask. Generation forwards it automatically.
Set `SEED_OCCLUSION_POLICY=allow` only after reviewing the seed uncertainty mask
when the inserted object overlaps synthesized background. The example runners
pass this setting to generation; the default remains `reject`.

The duck's bottom pivot is placed at the water's measured 3D position, so the
contact follows the scene geometry through the dolly. Its base is slightly
higher in the frame to leave room during forward motion. The prompt requests
water contact and ripples; those appearance effects are synthesized by GEN3C.

The cat example also segments out the camera between the man's hands and
inserts the candle from `batch_0008`, including its flame and melted wax puddle.
The candle is placed at base (162, 223), height 60 preview pixels, with its lowest
selected wax pixels anchored to a plane fitted to the visible tabletop below
the removed camera. This smaller, lower placement moves it slightly toward the
man in depth. Retained foreground geometry, including hands,
occludes it according to depth. Uncovered camera pixels remain holes for GEN3C
to complete. Segmentation and natural finger contact require visual review.

To fit GEN3C's two conditioning slots, candle RGB and planar depth are projected
into a new environment RGB-D source before cache construction; the cat uses
the other buffer. Original environment/camera labels and original candle source
and labels remain saved in `inputs/`. Inspect `candle_original_labels.png`,
`environment_labels.png` (camera removal), `environment_with_candle.png`, and
`camera_replacement.json`, along with the usual composite previews. This step
runs directly inside `run_cat.py`; no separate replacement script is needed.
The boxes are defined near the top of that script and can also be overridden:

```bash
CANDLE_BOX='163 94 240 218' CAMERA_REMOVE_BOX='139 164 186 226' \
  python scripts/composite_examples/run_cat.py
```

`CANDLE_BOX` refers to `batch_0008`; `CAMERA_REMOVE_BOX` refers to `batch_0007`.
Both use XYXY coordinates in the 320 × 240 source previews. Positive candle
points inside `run_cat.py` help SAM include the flame and wide wax base.

The camera moves forward and to the right while aiming at a fixed point near
scene center. Translation produces depth parallax: nearby surfaces shift more
than distant ones. Both CPU previews and the 121-frame diffusion trajectory
use the same path, starting at the original input view.

Defaults are `CAMERA_DOLLY_FORWARD=0.15` (15% of nearby selected geometry depth)
and `CAMERA_DOLLY_SIDEWAYS=0.20` (20% of view width at that depth). Travel adapts
to each scene's depth and focal length. For a stronger move:

```bash
CAMERA_DOLLY_FORWARD=0.25 CAMERA_DOLLY_SIDEWAYS=0.30 \
  python scripts/composite_examples/run_duck.py
```

Set `CAMERA_DOLLY_SIDEWAYS=0` for a straight dolly into the center, or use a
negative value to move left. Set both values to zero for a stationary camera.
Forward accepts 0 to 0.5; sideways accepts -0.5 to 0.5. Larger moves expose
unseen background, can crop subjects, and reveal the flat shape of imported
planar objects. Rerun preparation to update an existing scene; `--generate-only`
uses its saved path. The former `CAMERA_PAN_FRACTION` setting is no longer used.
`inputs/camera_path.json` records the dolly settings.

For CPU preparation and later GPU generation in the same output directory:

```bash
.venv/bin/python scripts/composite_examples/run_duck.py \
  --prepare-only --output-dir outputs/duck_example

# On the GPU machine, in the configured GEN3C environment:
python scripts/composite_examples/run_duck.py \
  --generate-only --output-dir outputs/duck_example
```

Scene JSON stores absolute source artifact paths. If moving prepared outputs to
another machine, keep the same absolute paths or update `prepared_path` and
`labels_path` in `inputs/scene_121.json` for each case.

Use the corresponding script to run each composition. `camera_path.py` shares
only the dolly calculation; each Python script contains its own boxes, placement,
preparation, trajectory conversion, prompt, and diffusion invocation.
Run each script with your chosen Python executable; `CHECKPOINT_DIR` chooses the diffusion
checkpoint root (SAM stays at `checkpoints/sam2.1_hiera_tiny.pt`, as in the MVP).
`CAT_DEPTH_MULTIPLIER` adjusts monitor occlusion while preserving initial size
and position. `REMOVE_BED_CAT=0` keeps the original cat if desired. Set these
placement controls during preparation. `SEED`, `NUM_STEPS`, and `GUIDANCE`
configure diffusion (defaults: 1, 35, 1). Adjust segmentation boxes/base/height in
the script's `prepare()` function and prepare again for other placements.

Outputs under the chosen directory:

- `duck/duck.mp4`, `cat/cat.mp4`, `creature/creature.mp4`.
- Each case's `inputs/`: `object_labels.png`, `environment_labels.png`,
  `preview.gif`, `seed.png`, prepared source/label files, `scene.json`, and
  `scene_121.json`.
- Each case's `<case>_review/`: full-resolution seed, conditioning buffers,
  generation manifest, and prompt; `prepare.log` and `inference.log` sit beside it.

The default output directory has a timestamp. An explicit `--output-dir` reuses
that directory and can overwrite outputs for the selected cases. The script
stops on errors; use the same example script with `--generate-only` to retry it.
