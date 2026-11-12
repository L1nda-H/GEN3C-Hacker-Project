"""Writer-rank review artifacts, separate from composite generation."""

import json
from pathlib import Path

import cv2
import numpy as np

from cosmos_predict1.diffusion.inference.instance_segmentation import file_sha256


def source_manifest(scene, composite):
    """Use already validated label provenance; hash label JSON only on the writer."""
    return [dict(
        source_id=source.source_id, fingerprint=source.fingerprint,
        labels_sha256=file_sha256(Path(source.labels_path).with_suffix('.json')),
        **composite.source_provenance[source.source_id],
        resolved_ids=sorted(composite.selections[source.source_id]),
        source_transform=composite.source_transforms[source.source_id].tolist(),
        object_transforms={str(i): t.tolist() for (owner, i), t in composite.object_transforms.items()
                           if owner == source.source_id},
    ) for source in scene.sources]


def save_manifest(directory, manifest):
    (directory / 'generation.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')


def save_render(directory, start, rendered):
    """Copy review arrays to CPU; do not retain GPU tensors after returning."""
    arrays = {name: value.cpu().numpy() for name, value in rendered.items()}
    np.savez_compressed(directory / f'render_{start:06d}.npz', **arrays)
    first_rgb = ((arrays['rgb'][0, 0, 0].transpose(1, 2, 0) + 1) * 127.5).round().astype(np.uint8)
    cv2.imwrite(str(directory / f'rgb_{start:06d}.png'), first_rgb[:, :, ::-1])
    cv2.imwrite(str(directory / f'coverage_{start:06d}.png'),
                (arrays['coverage'][0, 0, 0, 0] * 255).astype(np.uint8))
    z = arrays['depth'][0, 0, 0]
    cv2.imwrite(str(directory / f'depth_{start:06d}.png'),
                (z / max(float(z.max()), 1e-8) * 65535).round().astype(np.uint16))
    if start == 0:
        np.save(directory / 'seed.npy', arrays['rgb'][:, 0, 0][:, :, None])
        cv2.imwrite(str(directory / 'seed.png'), first_rgb[:, :, ::-1])
