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

import torch
from einops import rearrange

from cosmos_predict1.diffusion.inference.forward_warp_utils_pytorch import (
    forward_warp,
    motion_label,
    reliable_depth_mask_range_batch,
    unproject_points,
)
from cosmos_predict1.diffusion.inference.camera_utils import align_depth

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

        # Map dimension names to their indices in input_image
        format_to_indices = {dim: idx for idx, dim in enumerate(input_format)}
        input_shape = input_image.shape
        if input_mask is not None:
            input_image = torch.cat([input_image, input_mask], dim=format_to_indices.get("C"))

        # B (batch size), F (frame count), N dimensions: no aggregation during warping.
        # Only broadcasting over F to match the target w2c.
        # V: aggregate via concatenation or duster
        B = input_shape[format_to_indices.get("B", 0)] if "B" in format_to_indices else 1  # batch
        F = input_shape[format_to_indices.get("F", 0)] if "F" in format_to_indices else 1  # frame
        N = input_shape[format_to_indices.get("N", 0)] if "N" in format_to_indices else 1  # buffer
        V = input_shape[format_to_indices.get("V", 0)] if "V" in format_to_indices else 1  # view
        H = input_shape[format_to_indices.get("H", 0)] if "H" in format_to_indices else None
        W = input_shape[format_to_indices.get("W", 0)] if "W" in format_to_indices else None

        # Desired dimension order
        desired_dims = ["B", "F", "N", "V", "C", "H", "W"]

        # Build permute order based on input_format
        permute_order = []
        for dim in desired_dims:
            idx = format_to_indices.get(dim)
            if idx is not None:
                permute_order.append(idx)
            else:
                # Placeholder for dimensions to be added later
                permute_order.append(None)

        # Remove None values for permute operation
        permute_indices = [idx for idx in permute_order if idx is not None]
        input_image = input_image.permute(*permute_indices)

        # Insert dimensions of size 1 where necessary
        for i, idx in enumerate(permute_order):
            if idx is None:
                input_image = input_image.unsqueeze(i)

        # Now input_image has the shape B x F x N x V x C x H x W
        if input_mask is not None:
            self.input_image, self.input_mask = input_image[:, :, :, :, :3], input_image[:, :, :, :, 3:]
            self.input_mask = self.input_mask.to("cpu")
        else:
            self.input_mask = None
            self.input_image = input_image
        self.input_image = self.input_image.to(weight_dtype).to("cpu")

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

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        bs, F_target, _, _ = target_w2cs.shape

        B, F, N, V, C, H, W = self.input_image.shape
        assert bs == B

        target_w2cs = target_w2cs.reshape(B, F_target, 1, 4, 4).expand(B, F_target, N, 4, 4).reshape(-1, 4, 4)
        target_intrinsics = (
            target_intrinsics.reshape(B, F_target, 1, 3, 3).expand(B, F_target, N, 3, 3).reshape(-1, 3, 3)
        )

        # Keep large tensors on CPU; move only per-chunk slices to GPU inside the loop
        first_images = rearrange(self.input_image[:, start_frame_idx:start_frame_idx+F_target].expand(B, F_target, N, V, C, H, W), "B F N V C H W-> (B F N) V C H W")
        first_points = rearrange(
            self.input_points[:, start_frame_idx:start_frame_idx+F_target].expand(B, F_target, N, V, H, W, 3), "B F N V H W C-> (B F N) V H W C"
        )
        first_masks = rearrange(
            self.input_mask[:, start_frame_idx:start_frame_idx+F_target].expand(B, F_target, N, V, 1, H, W), "B F N V C H W-> (B F N) V C H W"
        ) if self.input_mask is not None else None
        boundary_masks = rearrange(
            self.boundary_mask.expand(B, F_target, N, V, 1, H, W), "B F N V C H W-> (B F N) V C H W"
        ) if self.boundary_mask is not None else None

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
                        first_masks[i : i + warp_chunk_size].to(self.device, non_blocking=True)
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

    def update_cache(self, new_image, new_depth, new_w2c, new_mask=None, new_intrinsics=None, depth_alignment=True, alignment_method="non_rigid"):  # 3D cache
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

        if self.filter_points_threshold < 1.0:
            B, F, N, V, C, H, W = self.input_image.shape
            new_depth = new_depth.reshape(-1, 1, H, W)
            depth_mask = reliable_depth_mask_range_batch(new_depth, ratio_thresh=self.filter_points_threshold).reshape(B, 1, H, W)
            if new_mask is None:
                new_mask = depth_mask.to("cpu")
            else:
                new_mask = new_mask * depth_mask.to(new_mask.device)
        if new_mask is not None:
            new_mask = new_mask.cpu()
        if self.frame_buffer_max > 1:  # newest frame first
            if self.input_image.shape[2] < self.frame_buffer_max:
                self.input_image = torch.cat([new_image[:, None, None, None], self.input_image], 2)
                self.input_points = torch.cat([new_points[:, None, None, None], self.input_points], 2)
                if self.input_mask is not None:
                    self.input_mask = torch.cat([new_mask[:, None, None, None], self.input_mask], 2)
            else:
                self.input_image[:, :, 0] = new_image[:, None, None]
                self.input_points[:, :, 0] = new_points[:, None, None]
                if self.input_mask is not None:
                    self.input_mask[:, :, 0] = new_mask[:, None, None]
        else:
            self.input_image = new_image[:, None, None, None]
            self.input_points = new_points[:, None, None, None]


    def render_cache(
        self,
        target_w2cs,
        target_intrinsics,
        render_depth: bool = False,
        start_frame_idx: int = 0,  # For consistency with Cache4D
    ):
        assert start_frame_idx == 0, "start_frame_idx must be 0 for Cache3D_Buffer"

        output_device = target_w2cs.device
        target_w2cs = target_w2cs.to(self.weight_dtype).to(self.device)
        target_intrinsics = target_intrinsics.to(self.weight_dtype).to(self.device)
        pixels, masks = super().render_cache(
            target_w2cs, target_intrinsics, render_depth
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
    ):
        # Warp from all buffer frames first
        output_device = target_w2cs.device
        target_w2cs = target_w2cs.to(self.weight_dtype).to(self.device)
        target_intrinsics = target_intrinsics.to(self.weight_dtype).to(self.device)

        pixels_all, masks_all = super().render_cache(
            target_w2cs, target_intrinsics, render_depth, start_frame_idx
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
        if self.mask_for_max_buffer_model and not render_depth:
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

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        rendered_warp_images, rendered_warp_masks = super().render_cache(target_w2cs, target_intrinsics, render_depth, start_frame_idx)
        return rendered_warp_images, rendered_warp_masks


class Cache3D_Labeled(Cache3D_Base):
    """Full per-frame cloud plus an is-moving label on every point.

    The frame axis is the ``F`` dimension of ``input_format``. Passing
    ``(F, C, H, W)`` requires ``input_format=["F", "C", "H", "W"]``. The 5x5
    depth filter stays off, so moving points are not removed.
    """

    def __init__(self, rel_thresh=0.05, **kwargs):
        kwargs["filter_points_threshold"] = 1.0
        kwargs["foreground_masking"] = False
        input_w2c = kwargs["input_w2c"]
        input_intrinsics = kwargs["input_intrinsics"]
        super().__init__(**kwargs)
        if self.input_depth is None:
            raise ValueError("Cache3D_Labeled requires input_depth.")
        self.rel_thresh = rel_thresh
        self.motion_label = self._compute_motion_label(input_w2c, input_intrinsics)

    def _compute_motion_label(self, input_w2c, input_intrinsics):
        b, frames, n, v, _, height, width = self.input_image.shape
        depth = rearrange(
            self.input_depth.reshape(b, frames, n, v, height, width),
            "b f n v h w -> (b n v) f 1 h w",
        ).float()
        w2c = rearrange(
            input_w2c.reshape(b, frames, n, v, 4, 4),
            "b f n v i j -> (b n v) f i j",
        ).to(device=depth.device, dtype=torch.float32)
        intrinsics = rearrange(
            input_intrinsics.reshape(b, frames, n, v, 3, 3),
            "b f n v i j -> (b n v) f i j",
        ).to(device=depth.device, dtype=torch.float32)
        labeled = motion_label(depth, w2c, intrinsics, self.rel_thresh)
        return rearrange(labeled, "(b n v) f c h w -> b f n v c h w", b=b, n=n, v=v).to(
            dtype=self.weight_dtype, device="cpu"
        )

    def update_cache(self, **kwargs):
        raise NotImplementedError

    def render_cache(self, target_w2cs, target_intrinsics, render_depth=False, start_frame_idx=0):
        """Warp each source frame to its target camera.

        Returns RGB, the coverage mask, and the splatted is-moving label.
        Moving points are included. The label uses the same points and validity
        mask as the color, so it lands on the same pixels.
        """
        if render_depth:
            raise NotImplementedError("Cache3D_Labeled returns RGB, coverage, and the motion label.")
        output_device = target_w2cs.device
        batch, frame_count, buffers, views, channels, height, width = self.input_image.shape
        if views != 1:
            raise NotImplementedError
        target_batch, target_frames, _, _ = target_w2cs.shape
        if target_batch != batch:
            raise ValueError(f"target batch {target_batch} does not match cache batch {batch}")
        end_frame_idx = start_frame_idx + target_frames
        if end_frame_idx > frame_count:
            raise ValueError(
                f"frames {start_frame_idx}:{end_frame_idx} exceed the cache length {frame_count}"
            )

        images = self.input_image[:, start_frame_idx:end_frame_idx, :, 0]
        points = self.input_points[:, start_frame_idx:end_frame_idx, :, 0]
        labels = self.motion_label[:, start_frame_idx:end_frame_idx, :, 0]
        depth = self.input_depth.reshape(batch, frame_count, buffers, views, 1, height, width)
        depth = depth[:, start_frame_idx:end_frame_idx, :, 0]
        flat = batch * target_frames * buffers
        images = images.reshape(flat, channels, height, width)
        points = points.reshape(flat, height, width, 3)
        labels = labels.reshape(flat, 1, height, width)
        depth = depth.reshape(flat, 1, height, width)
        cameras = (
            target_w2cs.reshape(batch, target_frames, 1, 4, 4)
            .expand(batch, target_frames, buffers, 4, 4)
            .contiguous()
            .reshape(flat, 4, 4)
        )
        intrinsics = (
            target_intrinsics.reshape(batch, target_frames, 1, 3, 3)
            .expand(batch, target_frames, buffers, 3, 3)
            .contiguous()
            .reshape(flat, 3, 3)
        )

        warped_images = []
        warped_masks = []
        warped_labels = []
        chunk = 2
        with torch.no_grad():
            for start in range(0, flat, chunk):
                stop = start + chunk
                depth_chunk = depth[start:stop].to(self.device, dtype=self.weight_dtype)
                valid = (torch.isfinite(depth_chunk) & (depth_chunk > 0)).to(self.weight_dtype)
                cameras_chunk = cameras[start:stop].to(self.device, dtype=self.weight_dtype)
                intrinsics_chunk = intrinsics[start:stop].to(self.device, dtype=self.weight_dtype)
                points_chunk = points[start:stop].to(self.device, dtype=self.weight_dtype)
                rgb, coverage, _, _ = forward_warp(
                    images[start:stop].to(self.device, dtype=self.weight_dtype),
                    mask1=valid,
                    depth1=None,
                    transformation1=None,
                    transformation2=cameras_chunk,
                    intrinsic1=intrinsics_chunk,
                    intrinsic2=intrinsics_chunk,
                    is_image=True,
                    world_points1=points_chunk,
                )
                splatted_label, _, _, _ = forward_warp(
                    labels[start:stop].to(self.device, dtype=self.weight_dtype),
                    mask1=valid,
                    depth1=None,
                    transformation1=None,
                    transformation2=cameras_chunk,
                    intrinsic1=intrinsics_chunk,
                    intrinsic2=intrinsics_chunk,
                    is_image=False,
                    world_points1=points_chunk,
                )
                warped_images.append(rgb.to(output_device))
                warped_masks.append(coverage.to(output_device))
                warped_labels.append(splatted_label.to(output_device))
        rgb = torch.cat(warped_images, dim=0).reshape(batch, target_frames, buffers, channels, height, width)
        coverage = torch.cat(warped_masks, dim=0).reshape(batch, target_frames, buffers, 1, height, width)
        splatted_label = torch.cat(warped_labels, dim=0).reshape(batch, target_frames, buffers, 1, height, width)
        return rgb, coverage, splatted_label
