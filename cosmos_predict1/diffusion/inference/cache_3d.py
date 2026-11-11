# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy

import torch
from einops import rearrange

from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import (
    forward_warp,
    reliable_depth_mask_range_batch,
    unproject_points,
)
from cosmos_predict1.diffusion.inference.camera_utils import align_depth

from cosmos_predict1.diffusion.inference.instance_segmentation import _cache_layout


def _checked_labels(labels, shape, registry):
    if (labels.dtype == torch.bool or labels.is_floating_point() or labels.is_complex()
            or tuple(labels.shape) != tuple(shape)):
        raise ValueError("Labels must be single-channel integer tensors aligned with RGB")
    # uint64 values must fit in signed int64 before converting.
    if labels.dtype == torch.uint64 and labels.numel() and labels.cpu().numpy().max() > 2**63 - 1:
        raise ValueError("Labels exceed int64 range")
    labels = labels.detach().to(device="cpu", dtype=torch.int64).clone()
    if (labels < -1).any():
        raise ValueError("Labels must be -1, 0, or registered positive IDs")
    if registry is None:
        raise ValueError("Labels require a source object registry")
    for instance_id in labels.unique().tolist():
        if instance_id > 0 and registry.resolve(instance_id) != instance_id:
            raise ValueError("Rebuild label maps after merging IDs")
    return labels


class Cache3D_Base:
    def __init__(
        self,
        input_image,
        input_depth,
        input_w2c,
        input_intrinsics,
        input_mask=None,
        input_format=None,
        input_points=None,
        weight_dtype=torch.float32,
        is_depth=True,
        device="cuda",
        filter_points_threshold=1.0,
        foreground_masking=False,
        input_labels=None,
        object_registry=None,
        observation_keys=None,
        source_fingerprint=None,
        segmentation_revision=None,
    ):
        """
        input_image: Tensor with varying dimensions.
        input_format: List of dimension labels corresponding to input_image's dimensions.
                      E.g., ['B', 'C', 'H', 'W'], ['B', 'F', 'C', 'H', 'W'], etc.
        """
        self.weight_dtype = weight_dtype
        self.is_depth = is_depth
        self.device = device
        self.filter_points_threshold = filter_points_threshold
        self.foreground_masking = foreground_masking
        if input_format is None:
            assert input_image.dim() == 4
            input_format = ["B", "C", "H", "W"]

        image = _cache_layout(input_image, input_format)
        B, F, N, V, C, H, W = image.shape
        if C != 3:
            raise ValueError("Cache RGB must have three channels")
        self.input_image = image.to(device="cpu", dtype=weight_dtype)
        self.input_mask = None
        if input_mask is not None:
            mask = _cache_layout(input_mask, input_format)
            if mask.shape != (*image.shape[:4], 1, H, W):
                raise ValueError("Geometry mask must be single-channel and aligned with RGB")
            self.input_mask = mask.to("cpu").clone()
        self.object_registry = deepcopy(object_registry)
        self.input_labels = None if input_labels is None else _checked_labels(
            _cache_layout(input_labels, input_format), (*image.shape[:4], 1, H, W), self.object_registry,
        )
        if segmentation_revision is not None and (
            self.object_registry is None or segmentation_revision not in self.object_registry.revision_fingerprints
        ):
            raise ValueError("Unknown segmentation revision")
        self.source_fingerprint = source_fingerprint
        self.segmentation_revision = segmentation_revision
        self._label_binding = (None if self.object_registry is None else self.object_registry.source_id,
                               source_fingerprint, segmentation_revision)
        self._retired_revisions = set()
        self.observation_keys = None if observation_keys is None else tuple(observation_keys)
        if self.observation_keys is not None and (
            len(self.observation_keys) != F * N * V or len(set(self.observation_keys)) != F * N * V
        ):
            raise ValueError("Observation keys must match cache frame/view samples")
        self.observation_to_offset = {} if self.observation_keys is None else {
            key: (i // (N * V), (i // V) % N, i % V) for i, key in enumerate(self.observation_keys)
        }

        # Capture validity before depth sanitization, without changing geometry.
        if input_points is not None:
            valid = torch.isfinite(input_points.reshape(B, F, N, V, H, W, 3)).all(-1).unsqueeze(4).cpu()
        else:
            valid = (torch.isfinite(input_depth) & (input_depth > 0)).reshape(B, F, N, V, 1, H, W).cpu()
        self.input_mask = valid if self.input_mask is None else self.input_mask * valid

        if input_points is not None:
            self.input_points = input_points.reshape(B, F, N, V, H, W, 3).to("cpu")
            self.input_depth = None
        else:
            input_depth = torch.nan_to_num(input_depth, nan=100)
            input_depth = torch.clamp(input_depth, min=0, max=100)
            if weight_dtype == torch.float16:
                input_depth = torch.clamp(input_depth, max=70)
            self.input_points = (
                self._compute_input_points(
                    input_depth.reshape(-1, 1, H, W),
                    input_w2c.reshape(-1, 4, 4),
                    input_intrinsics.reshape(-1, 3, 3),
                )
                .to(weight_dtype)
                .reshape(B, F, N, V, H, W, 3)
                .to("cpu")
            )
            self.input_depth = input_depth

        if self.filter_points_threshold < 1.0 and input_depth is not None:
            input_depth = input_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(input_depth, ratio_thresh=self.filter_points_threshold).reshape(B, F, N, V, 1, H, W)
            if self.input_mask is None:
                self.input_mask = depth_mask.to("cpu")
            else:
                self.input_mask = self.input_mask * depth_mask.to(self.input_mask.device)
        self.boundary_mask = None
        if foreground_masking:
            input_depth = input_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(input_depth)
            self.boundary_mask = (~depth_mask).reshape(B, F, N, V, 1, H, W).to("cpu")

    def _selected_mask(self, selected_ids):
        if selected_ids is None:
            return self.input_mask
        if self.input_labels is None:
            raise ValueError("Explicit selection requires a labeled cache")
        resolved = []
        for instance_id in selected_ids:
            if type(instance_id) is not int or instance_id < 0:
                raise ValueError("Select canonical IDs or 0 for environment; -1 is unresolved")
            resolved.append(0 if instance_id == 0 else self.object_registry.resolve(instance_id))
        selected = torch.isin(self.input_labels, torch.tensor(resolved, dtype=torch.int64))
        return selected if self.input_mask is None else self.input_mask * selected

    def replace_labels(self, labeled_source):
        """Attach a corrected snapshot to unchanged geometry, preserving ID aliases."""
        kwargs = labeled_source.cache_kwargs()  # Validates maps, source and revision.
        registry = kwargs['object_registry']
        revision = kwargs['segmentation_revision']
        if (self.source_fingerprint is None or kwargs['source_fingerprint'] != self.source_fingerprint
                or self.object_registry.source_id != registry.source_id
                or tuple(kwargs['observation_keys']) != self.observation_keys):
            raise ValueError("Replacement labels must match the original source and observation layout")
        if (revision in self._retired_revisions
                or self.segmentation_revision not in registry.revision_fingerprints):
            raise ValueError("Stale or unrelated segmentation revision")
        if any(registry.revision_fingerprints.get(old) != fingerprint
               for old, fingerprint in self.object_registry.revision_fingerprints.items()):
            raise ValueError("Stale or conflicting segmentation history")
        if any(old not in registry.aliases or registry.resolve(old) != registry.resolve(target)
               for old, target in self.object_registry.aliases.items()):
            raise ValueError("Stale registry would undo an identity merge")
        labels = _checked_labels(_cache_layout(kwargs['input_labels'], kwargs['input_format']),
                                 self.input_labels.shape, registry)
        if revision != self.segmentation_revision:
            self._retired_revisions.add(self.segmentation_revision)
        self.input_labels = labels
        self.object_registry = deepcopy(registry)
        self.segmentation_revision = revision
        self._label_binding = (registry.source_id, self.source_fingerprint, revision)

    def check_label_consistency(self):
        """Check IDs/layout and report semantic versus renderable point counts."""
        if self.input_labels is None:
            raise ValueError("Cache has no labels")
        if self._label_binding != (self.object_registry.source_id, self.source_fingerprint, self.segmentation_revision):
            raise ValueError("Cache source/revision no longer matches its label attachment")
        shape = (*self.input_image.shape[:4], 1, *self.input_image.shape[-2:])
        if self.input_labels.dtype != torch.int64 or self.input_labels.device.type != "cpu":
            raise ValueError("Cache labels must be CPU int64")
        _checked_labels(self.input_labels, shape, self.object_registry)
        if self.input_mask.shape != self.input_labels.shape or self.input_points.shape != (*shape[:4], *shape[-2:], 3):
            raise ValueError("Labels, points and validity are misaligned")
        if self.segmentation_revision is not None and self.segmentation_revision not in self.object_registry.revision_fingerprints:
            raise ValueError("Unknown segmentation revision")
        return {i: dict(labeled=int((self.input_labels == i).sum()),
                        renderable=int(((self.input_labels == i) & self.input_mask.bool()).sum()))
                for i in self.input_labels.unique().tolist()}

    def _compute_input_points(self, input_depth, input_w2c, input_intrinsics):
        input_points = unproject_points(
            input_depth,
            input_w2c,
            input_intrinsics,
            is_depth=self.is_depth,
        )
        return input_points

    def update_cache(self):
        raise NotImplementedError

    def input_frame_count(self) -> int:
        return self.input_image.shape[1]

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0, selected_ids=None):
        bs, F_target, _, _ = target_w2cs.shape

        B, F, N, V, C, H, W = self.input_image.shape
        assert bs == B
        if V != 1:
            raise NotImplementedError("Rendering V > 1 is deferred; use N for static source views")
        if type(start_frame_idx) is not int or start_frame_idx < 0 or F_target <= 0:
            raise ValueError("Render window requires a nonnegative offset and positive frame count")
        if F > 1 and start_frame_idx + F_target > F:
            raise ValueError("Render window exceeds cached temporal frames")
        window = slice(0, 1) if F == 1 else slice(start_frame_idx, start_frame_idx + F_target)
        selected_mask = self._selected_mask(selected_ids)

        target_w2cs = target_w2cs.reshape(B, F_target, 1, 4, 4).expand(B, F_target, N, 4, 4).reshape(-1, 4, 4)
        target_intrinsics = (
            target_intrinsics.reshape(B, F_target, 1, 3, 3).expand(B, F_target, N, 3, 3).reshape(-1, 3, 3)
        )

        # Keep large tensors on CPU; move only per-chunk slices to GPU inside the loop
        first_images = rearrange(self.input_image[:, window].expand(B, F_target, N, V, C, H, W), "B F N V C H W-> (B F N) V C H W")
        first_points = rearrange(
            self.input_points[:, window].expand(B, F_target, N, V, H, W, 3), "B F N V H W C-> (B F N) V H W C"
        )
        first_masks = rearrange(
            selected_mask[:, window].expand(B, F_target, N, V, 1, H, W), "B F N V C H W-> (B F N) V C H W"
        ) if selected_mask is not None else None
        boundary = self.boundary_mask
        if boundary is not None and selected_ids is not None:
            boundary = boundary & selected_mask.bool()
        boundary_masks = rearrange(
            boundary[:, window].expand(B, F_target, N, V, 1, H, W),
            "B F N V C H W-> (B F N) V C H W",
        ) if boundary is not None else None

        if first_images.shape[1] == 1:
            warp_chunk_size = 2
            rendered_warp_images = []
            rendered_warp_masks = []
            rendered_warp_depth = []

            first_images = first_images.squeeze(1)
            first_points = first_points.squeeze(1)
            first_masks = first_masks.squeeze(1) if first_masks is not None else None
            for i in range(0, first_images.shape[0], warp_chunk_size):
                with torch.no_grad():
                    imgs_chunk = first_images[i : i + warp_chunk_size].to(self.device, non_blocking=True)
                    pts_chunk = first_points[i : i + warp_chunk_size].to(self.device, non_blocking=True)
                    masks_chunk = (
                        first_masks[i : i + warp_chunk_size].to(self.device, non_blocking=True).clone()
                        if first_masks is not None
                        else None
                    )
                    bmask_chunk = (
                        boundary_masks[i : i + warp_chunk_size, 0, 0].to(self.device, non_blocking=True)
                        if boundary_masks is not None
                        else None
                    )
                    (
                        rendered_warp_images_chunk,
                        rendered_warp_masks_chunk,
                        rendered_warp_depth_chunk,
                        _,
                    ) = forward_warp(
                        imgs_chunk,
                        mask1=masks_chunk,
                        depth1=None,
                        transformation1=None,
                        transformation2=target_w2cs[i : i + warp_chunk_size],
                        intrinsic1=target_intrinsics[i : i + warp_chunk_size],
                        intrinsic2=target_intrinsics[i : i + warp_chunk_size],
                        render_depth=render_depth,
                        world_points1=pts_chunk,
                        foreground_masking=self.foreground_masking,
                        boundary_mask=bmask_chunk,
                    )
                    rendered_warp_images.append(rendered_warp_images_chunk.to("cpu"))
                    rendered_warp_masks.append(rendered_warp_masks_chunk.to("cpu"))
                    if render_depth:
                        rendered_warp_depth.append(rendered_warp_depth_chunk.to("cpu"))
                    del imgs_chunk, pts_chunk, masks_chunk, bmask_chunk
                    del rendered_warp_images_chunk, rendered_warp_masks_chunk
                    if render_depth:
                        del rendered_warp_depth_chunk
                    torch.cuda.empty_cache()
            rendered_warp_images = torch.cat(rendered_warp_images, dim=0)
            rendered_warp_masks = torch.cat(rendered_warp_masks, dim=0)
            if render_depth:
                rendered_warp_depth = torch.cat(rendered_warp_depth, dim=0)

        else:
            raise NotImplementedError

        pixels = rearrange(rendered_warp_images, "(b f n) c h w -> b f n c h w", b=bs, f=F_target, n=N)
        masks = rearrange(rendered_warp_masks, "(b f n) c h w -> b f n c h w", b=bs, f=F_target, n=N)
        if render_depth:
            pixels = rearrange(rendered_warp_depth, "(b f n) h w -> b f n h w", b=bs, f=F_target, n=N)
        return pixels.to(self.device), masks.to(self.device)


class Cache3D_Buffer(Cache3D_Base):
    def __init__(self, frame_buffer_max=0, noise_aug_strength=0, generator=None, **kwargs):
        super().__init__(**kwargs)
        self.frame_buffer_max = frame_buffer_max
        self.noise_aug_strength = noise_aug_strength
        self.generator = generator

    def update_cache(self, new_image, new_depth, new_w2c, new_mask=None, new_intrinsics=None, depth_alignment=True, alignment_method="non_rigid", new_labels=None):  # 3D cache
        B, F, N, V, C, H, W = self.input_image.shape
        if F != 1 or V != 1 or new_image.shape != (B, C, H, W) or new_depth.shape != (B, 1, H, W):
            raise ValueError("Buffer updates require aligned [B,C,H,W] images and [B,1,H,W] depth")
        if self.input_labels is not None:
            if new_labels is None:
                raise ValueError("Labeled cache updates require associated new_labels")
            new_labels = _checked_labels(new_labels, (B, 1, H, W), self.object_registry)
        elif new_labels is not None:
            raise ValueError("Cannot add labels to an unlabeled buffer update")
        valid = (torch.isfinite(new_depth) & (new_depth > 0)).cpu()
        if new_mask is not None:
            if new_mask.shape != valid.shape:
                raise ValueError("Update validity must match depth")
            valid = valid & new_mask.cpu().bool()
        new_mask = valid
        new_image = new_image.to(self.weight_dtype).to(self.device)
        new_depth = new_depth.to(self.weight_dtype).to(self.device)
        new_w2c = new_w2c.to(self.weight_dtype).to(self.device)
        if new_intrinsics is not None:
            new_intrinsics = new_intrinsics.to(self.weight_dtype).to(self.device)

        new_depth = torch.nan_to_num(new_depth, nan=1e4)
        new_depth = torch.clamp(new_depth, min=0, max=1e4)

        if depth_alignment:
            target_depth, target_mask = self.render_cache(
                new_w2c.unsqueeze(1), new_intrinsics.unsqueeze(1), render_depth=True
            )
            target_depth, target_mask = target_depth[:, :, 0], target_mask[:, :, 0]
            if alignment_method == "rigid":
                new_depth = (
                    align_depth(
                        new_depth.squeeze(),
                        target_depth.squeeze(),
                        target_mask.bool().squeeze(),
                    )
                    .reshape_as(new_depth)
                    .detach()
                )
            elif alignment_method == "non_rigid":
                with torch.enable_grad():
                    new_depth = (
                        align_depth(
                            new_depth.squeeze(),
                            target_depth.squeeze(),
                            target_mask.bool().squeeze(),
                            k=new_intrinsics.squeeze(),
                            c2w=torch.inverse(new_w2c.squeeze()),
                            alignment_method="non_rigid",
                            num_iters=100,
                            lambda_arap=0.1,
                            smoothing_kernel_size=3,
                        )
                        .reshape_as(new_depth)
                        .detach()
                    )
            else:
                raise NotImplementedError
        new_points = unproject_points(new_depth, new_w2c, new_intrinsics, is_depth=self.is_depth).cpu()
        new_image = new_image.cpu()

        # Keep every per-point field in the same buffer slots.
        new_mask = new_mask & (torch.isfinite(new_depth) & (new_depth > 0)).cpu()
        if self.filter_points_threshold < 1.0:
            new_mask &= reliable_depth_mask_range_batch(
                new_depth, ratio_thresh=self.filter_points_threshold,
            ).cpu().bool()
        fields = dict(input_image=new_image[:, None, None, None],
                      input_points=new_points[:, None, None, None],
                      input_mask=new_mask[:, None, None, None])
        if new_labels is not None:
            fields['input_labels'] = new_labels[:, None, None, None]
        if self.boundary_mask is not None:
            boundary = ~reliable_depth_mask_range_batch(new_depth).cpu().bool()
            fields['boundary_mask'] = boundary[:, None, None, None]
        for name, value in fields.items():
            previous = getattr(self, name)
            if self.frame_buffer_max > 1:
                if N < self.frame_buffer_max:
                    value = torch.cat([value, previous], dim=2)
                else:
                    value = torch.cat([value, previous[:, :, 1:]], dim=2)
            setattr(self, name, value)
        self.input_depth = new_depth.cpu()
        # Updated geometry no longer corresponds to the original source snapshot.
        self.source_fingerprint = None
        self.segmentation_revision = None
        self._label_binding = (None if self.object_registry is None else self.object_registry.source_id, None, None)
        self.observation_keys = None
        self.observation_to_offset = {}


    def render_cache(
        self,
        target_w2cs,
        target_intrinsics,
        render_depth: bool = False,
        start_frame_idx: int = 0,  # For consistency with Cache4D
        selected_ids=None,
    ):

        output_device = target_w2cs.device
        target_w2cs = target_w2cs.to(self.weight_dtype).to(self.device)
        target_intrinsics = target_intrinsics.to(self.weight_dtype).to(self.device)
        pixels, masks = super().render_cache(
            target_w2cs, target_intrinsics, render_depth, start_frame_idx, selected_ids
        )
        pixels = pixels.to(output_device)
        masks = masks.to(output_device)
        if not render_depth:
            noise = torch.randn(pixels.shape, generator=self.generator, device=pixels.device, dtype=pixels.dtype)
            per_buffer_noise = (
                torch.arange(start=pixels.shape[2] - 1, end=-1, step=-1, device=pixels.device)
                * self.noise_aug_strength
            )
            pixels = pixels + noise * per_buffer_noise.reshape(1, 1, -1, 1, 1, 1)  # B, F, N, C, H, W
        return pixels, masks


class Cache3D_BufferSelector(Cache3D_Base):
    def __init__(self, frame_buffer_max=1, mask_for_max_buffer_model: bool = True, mask_full_threshold: float = 0.9, **kwargs):
        """A buffer that holds many initialization frames and selects top-K by overlap per target.

        This class does not support update_cache. It assumes multiple source frames are provided
        at initialization time via the 'N' (buffer) dimension.
        """
        super().__init__(**kwargs)
        self.frame_buffer_max = max(int(frame_buffer_max), 1)
        self.mask_for_max_buffer_model = bool(mask_for_max_buffer_model)
        self.mask_full_threshold = float(mask_full_threshold)

    def update_cache(self, *args, **kwargs): 
        raise NotImplementedError("Cache3D_BufferSelector does not support update_cache")

    def render_cache(
        self,
        target_w2cs,
        target_intrinsics,
        render_depth: bool = False,
        start_frame_idx: int = 0,
        selected_ids=None,
    ):
        # Warp from all buffer frames first
        output_device = target_w2cs.device
        target_w2cs = target_w2cs.to(self.weight_dtype).to(self.device)
        target_intrinsics = target_intrinsics.to(self.weight_dtype).to(self.device)

        pixels_all, masks_all = super().render_cache(
            target_w2cs, target_intrinsics, render_depth, start_frame_idx, selected_ids
        )  # shapes: [B, F, N, C, H, W] (pixels) and [B, F, N, 1, H, W] (masks)

        B, F, N = pixels_all.shape[0], pixels_all.shape[1], pixels_all.shape[2]
        if N <= self.frame_buffer_max:
            pixels_sel, masks_sel = pixels_all, masks_all
        else:
            # Compute per-buffer overlap score: sum over frames and pixels
            # masks_all: [B, F, N, 1, H, W]
            overlap_scores = masks_all.sum(dim=(1, 3, 4, 5))  # -> [B, N]

            # Select top-K for each batch independently
            k = min(self.frame_buffer_max, N)
            topk_indices = overlap_scores.topk(k=k, dim=1, largest=True, sorted=True).indices  # [B, k]

            # Gather along N dimension
            selected_pixels_list = []
            selected_masks_list = []
            for b in range(B):
                idx_b = topk_indices[b]  # [k]
                selected_pixels_list.append(pixels_all[b : b + 1, :, idx_b])  # [1, F, k, C, H, W]
                selected_masks_list.append(masks_all[b : b + 1, :, idx_b])    # [1, F, k, 1, H, W]

            pixels_sel = torch.cat(selected_pixels_list, dim=0)
            masks_sel = torch.cat(selected_masks_list, dim=0)
        if self.mask_for_max_buffer_model and not render_depth and selected_ids is None:
            # masks_sel: [B, F, k, 1, H, W]
            _masks = masks_sel.mean(dim=[3, 4, 5])  # -> [B, F, k]
            Bm, Fm, Nm = _masks.shape
            _masks_flat = rearrange(_masks, "b t n -> (b t) n")
            result_mask = torch.zeros_like(_masks_flat)
            near_full = _masks_flat >= self.mask_full_threshold
            has_near_full = near_full.any(dim=1)

            indices = near_full.float().argmax(dim=1)
            valid_rows = torch.arange(near_full.size(0), device=_masks_flat.device)[has_near_full]
            valid_indices = indices[has_near_full]
            result_mask[valid_rows, valid_indices] = 1

            invalid_rows = torch.arange(near_full.size(0), device=_masks_flat.device)[~has_near_full]
            if invalid_rows.numel() > 0:
                result_mask[invalid_rows] = 1
            result_mask = rearrange(result_mask, "(b t) n -> b t n", b=Bm, t=Fm)

            result_mask_expanded = result_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, F, k, 1, 1, 1]
            pixels_sel = (pixels_sel + 1) * result_mask_expanded - 1
            masks_sel = masks_sel * result_mask_expanded
        return pixels_sel.to(output_device), masks_sel.to(output_device)


class Cache4D(Cache3D_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def update_cache(self, **kwargs):
        raise NotImplementedError

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0, selected_ids=None):
        rendered_warp_images, rendered_warp_masks = super().render_cache(target_w2cs, target_intrinsics, render_depth, start_frame_idx, selected_ids)
        return rendered_warp_images, rendered_warp_masks
