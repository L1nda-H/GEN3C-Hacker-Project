# Static composite generation

`Gen3cCompositePipeline` combines labeled sources and wraps the existing
`Gen3cPipeline` for diffusion. `gen3c_composite.py` is its dedicated CLI.
`gen3c_multiview.py` remains unchanged and handles observations of one scene.
The current composite pipeline accepts one prepared image per source.

## Prepare reproducible scene inputs

Persist the **same PreparedSource used for labeling**, together with its
`LabeledSource`. Do not reconstruct these from resized images, rerun depth
estimation, or resize label maps. Existing label artifacts omit RGB/depth/cameras,
so the prepared input must also be saved. Both paths refer to local files.

```python
from pathlib import Path
import torch

from cosmos_predict1.diffusion.inference.composite_scene import (
    SceneSource, SceneSpec, save_prepared_source,
)
from cosmos_predict1.diffusion.inference.instance_labeling import save_labeled_source

# room_labels and object_labels are validated LabeledSource results from the
# existing labeling stage. ID 0 must be GUI/user-confirmed environment.
# Object IDs below come from the source's registry, never SAM proposal indices.
root = Path("datasets/composite_example").resolve()
root.mkdir(parents=True, exist_ok=True)
for name, labeled in [("room", room_labels), ("objects", object_labels)]:
    save_prepared_source(root / f"{name}.npz", labeled.prepared_source)
    save_labeled_source(root / f"{name}_labels", labeled)

placement = torch.eye(4, dtype=torch.float64)
placement[:3, 3] = torch.tensor([0.3, 0.0, 0.0])  # calibrate in room units
scene = SceneSpec(
    sources=[
        SceneSource(room_labels.registry.source_id, root / "room.npz",
                    root / "room_labels", room_labels.source_fingerprint, [0]),
        SceneSource(object_labels.registry.source_id, root / "objects.npz",
                    root / "objects_labels", object_labels.source_fingerprint,
                    selected_ids=[1], source_transform=placement),
    ],
    environment_source=room_labels.registry.source_id,
    target_w2cs=target_w2cs,              # [1,T,4,4], OpenCV world-to-camera
    target_intrinsics=target_intrinsics,  # [1,T,3,3]
    image_size=(704, 1280),               # exact diffusion output dimensions
)
scene.save(root / "scene.json")
```

Use distinct registry source IDs. The environment source must explicitly select
0 and retain its identity source transform. All other selected IDs must be
registered positive IDs; unknown/unassigned regions are excluded. Object
transforms are optional `{positive_id: matrix}` mappings. Placement applies
`T_source @ T_object @ p_source`, with orientation-preserving uniform scale,
rotation and translation. Transforms never modify the child geometry.

The prepared NPZ schema stores a JSON `metadata` scalar containing version 1,
the original observation key, preparation settings, and exact source
fingerprint. Its `rgb`, `depth`, `intrinsics`, `w2c`, and `geometry_mask` arrays
preserve their original dtypes and values. Source and label fingerprints are
checked before cache construction. This is separate from legacy VIPE exports.

`SceneSpec.load("scene.json")` resolves artifact paths relative to that JSON
file. `SceneSpec.save()` records absolute resolved paths. Direct Python models
and loaded JSON share validation; call `scene.validate()` after GUI edits.

## Generate

On the configured CUDA inference machine, with the existing GEN3C dependencies
and checkpoints installed:

```bash
python -m cosmos_predict1.diffusion.inference.gen3c_composite \
  --scene_spec datasets/composite_example/scene.json \
  --checkpoint_dir checkpoints \
  --prompt "A red chair in a furnished room" \
  --num_video_frames 241 --height 704 --width 1280 \
  --video_save_folder outputs/composite --video_save_name chair
```

Use `120*N+1` frames, at least 121, with enough target cameras. The CLI shares
the existing Hydra/model initialization and supported offloading options, without
initializing SAM. Every prompt must be explicit and nonempty, including batch
entries. Prompt upsampling is disabled; the composite CLI exposes no upsampler
or unused tokenizer-directory flags. It accepts no
legacy `--npz_path`, `--labels_path`, or `--selected_ids` flags; selections belong
to the scene. It records the supplied `--checkpoint_revision`, or computes a
SHA256 of the diffusion `model.pt` when omitted (this reads the entire file).

For Python/GUI callers with an already initialized diffusion pipeline:

```python
from cosmos_predict1.diffusion.inference.gen3c_composite_pipeline import Gen3cCompositePipeline

output = Gen3cCompositePipeline(diffusion_pipeline).generate(
    scene, prompt="A red chair in a furnished room", device="cuda",
    review_dir="outputs/composite/chair_review",
    generation_settings={"checkpoint_revision": "your immutable checkpoint revision"},
)
# output is (uint8_video_THWC, final_prompt), or None if a guardrail blocks a chunk.
```

The first camera must cover at least half of the output pixels by default.
`--min_seed_coverage` adjusts that threshold. Insufficient coverage saves the
seed and previews, then raises an error so the scene author can choose a better
starting camera; camera order is never changed automatically. Seed holes use
the renderer's black fill (-1). The seed latent path has no coverage mask, so
inspect hole artifacts during real inference. Warp conditioning does carry
binary coverage. All selected sources share one joint depth test, with each
winning surface routed into its source buffer. Selected objects from the same
source share that buffer. Buffers are ordered by sorted source ID, recorded as
`conditioning_source_ids` in the
manifest. Closer geometry occludes farther geometry across sources, including
the environment. Hidden RGB, depth, and coverage are cleared to -1/0/0. Sources
with no selection are omitted; empty or fully hidden selected sources retain slots.
Source placement must establish consistent destination positions and scale;
back-projection alone does not align unrelated scenes.

The existing model independently encodes each RGB and mask buffer, concatenates
those latents along channels, and pads unused slots with zero latents. The
number of buffers must not exceed the checkpoint's `frame_buffer_max` (currently
2 in the GEN3C inference configuration); generation rejects overflow rather than
silently discarding sources or changing model capacity. Model weights and APIs
are unchanged. This is the current code's latent concatenation strategy, not
the paper's feature max pooling.

`render_cache()` and `render_diagnostics()` return combined scene renders;
`render_conditioning()` routes the same joint winners into separate source
buffers. The initial seed and preview joint images are recombined from visible
buffers without another projection. Recombination sums covered RGB only and
fills uncovered pixels with -1. Rendering does not fill gaps between points.

Chunks overlap by one frame. Each continuation uses the preceding generated
frame as its seed. Source caches remain static and generated-frame cache
updates are disabled. Video segments are concatenated once after generation.
Persistent-server support remains deferred.

## Review artifacts

The CLI saves `<name>.mp4` and `<name>_review/` containing:

- `scene.json`: resolved scene configuration and placement matrices.
- `generation.json`: source fingerprints, label revisions/provenance and hashes,
  resolved selections/transforms, generation settings, code/checkpoint revision,
  seed coverage, ordered conditioning source IDs, strategy, visibility policy,
  depth tolerance, and completion/blocked/failure status.
- `seed.npy` and `seed.png`: exact normalized seed `[1,3,1,H,W]` and preview.
- `render_<start>.npz`: complete per-chunk RGB `[1,T,N,3,H,W]` in [-1,1],
  metric camera-Z depth `[1,T,N,H,W]` and coverage `[1,T,N,1,H,W]` in {0,1}.
- `rgb_<start>_buffer_<index>.png`, `depth_<start>_buffer_<index>.png`,
  `coverage_<start>_buffer_<index>.png`: the first frame of each source buffer
  in each chunk. `N` is the number of selected source caches. Depth PNGs normalize to each frame's maximum; use NPZ
  arrays for depth measurements.
- `prompt.txt`: final prompt on successful CLI generation.

CPU contract tests use mocked diffusion. The MVP still requires manual CUDA
validation on GCP: inspect placement scale, generated occlusion boundaries, excluded
object leakage, seed holes, recognizability, and temporal consistency. Record
visual failures separately from errors already visible in geometric renders.

Every rank renders RGB/coverage with joint visibility; only the writer returns
review depth. Rendering does not allocate categorical winner or nearest-depth
diagnostics. Review depth is released before diffusion, and conditioning buffers
before the next chunk.
Other ranks omit review-only buffers, hashes, and files. Source provenance comes
from the already validated label records. Preview tools retain full diagnostics.
