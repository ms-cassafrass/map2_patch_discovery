from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from .config import PatchConfig, SamplingConfig


@dataclass(frozen=True)
class PatchCenter:
    x: int
    y: int
    group: str
    distance_to_mask_px: float
    map2_overlap_fraction: float


def _valid_center_region(mask_shape: tuple[int, int], patch: PatchConfig) -> np.ndarray:
    height, width = mask_shape
    half_h = patch.height_px // 2
    half_w = patch.width_px // 2
    valid = np.zeros(mask_shape, dtype=bool)
    valid[half_h:height - half_h, half_w:width - half_w] = True
    return valid


def _center_grid(mask_shape: tuple[int, int], stride_px: int) -> np.ndarray:
    height, width = mask_shape
    yy, xx = np.mgrid[0:height:stride_px, 0:width:stride_px]
    return np.column_stack([yy.ravel(), xx.ravel()])


def _patch_overlap(mask: np.ndarray, y: int, x: int, patch: PatchConfig) -> float:
    half_h = patch.height_px // 2
    half_w = patch.width_px // 2
    crop = mask[y - half_h:y + half_h, x - half_w:x + half_w]
    if crop.size == 0:
        return 0.0
    return float(np.mean(crop))


def sample_patch_centers(mask: np.ndarray, patch: PatchConfig, sampling: SamplingConfig) -> list[PatchCenter]:
    if mask.ndim != 2:
        raise ValueError(f"MAP2 mask must be 2D for phase 1 sampling, got {mask.ndim}D")

    valid = _valid_center_region(mask.shape, patch)
    inside_dist = distance_transform_edt(mask)
    outside_dist = distance_transform_edt(~mask)
    signed_distance = inside_dist.astype(np.float32)
    signed_distance[~mask] = -outside_dist[~mask].astype(np.float32)

    boundary = np.abs(signed_distance) <= float(sampling.boundary_width_px)
    in_mask = mask & (signed_distance > float(sampling.boundary_width_px))
    near_mask_outside = (~mask) & (outside_dist > float(sampling.boundary_width_px)) & (
        outside_dist <= float(sampling.near_outside_distance_px)
    )
    far_background = (~mask) & (outside_dist >= float(sampling.far_background_min_distance_px))

    group_masks = {
        "in_mask": in_mask & valid,
        "boundary": boundary & valid,
        "near_mask_outside": near_mask_outside & valid,
        "far_background": far_background & valid,
    }

    rng = np.random.default_rng(sampling.random_seed)
    centers: list[PatchCenter] = []
    grid = _center_grid(mask.shape, patch.stride_px)

    for group_name in sampling.groups:
        group_mask = group_masks[group_name]
        group_candidates = [(int(y), int(x)) for y, x in grid if group_mask[int(y), int(x)]]
        if not group_candidates:
            continue
        rng.shuffle(group_candidates)
        selected = group_candidates[: patch.max_patches_per_group]
        for y, x in selected:
            overlap = _patch_overlap(mask, y, x, patch)
            centers.append(
                PatchCenter(
                    x=x,
                    y=y,
                    group=group_name,
                    distance_to_mask_px=float(signed_distance[y, x]),
                    map2_overlap_fraction=overlap,
                )
            )

    return centers
