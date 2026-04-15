from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DatasetConfig, PatchConfig, SampleConfig
from .ome import load_binary_mask, open_ome_image
from .sampling import PatchCenter, sample_patch_centers
from .summaries import summarize_channel_crop


def _resolve_channel_indices(sample: SampleConfig, required_channels: list[str]) -> dict[str, int]:
    name_to_index = {name: idx for idx, name in enumerate(sample.channel_names)}
    missing = [name for name in required_channels if name not in name_to_index]
    if missing:
        raise ValueError(f"Sample {sample.sample_id} missing required channels: {missing}")
    return {name: name_to_index[name] for name in required_channels}


def _compute_z_center(channel_volume: np.ndarray, center: PatchCenter) -> int:
    z_profile = channel_volume[:, center.y, center.x]
    if z_profile.size == 0 or not np.any(np.isfinite(z_profile)):
        return int(channel_volume.shape[0] // 2)
    return int(np.nanargmax(z_profile))


def _crop_zyx(volume: np.ndarray, center: PatchCenter, patch: PatchConfig, z_center: int) -> np.ndarray:
    half_h = patch.height_px // 2
    half_w = patch.width_px // 2
    half_z = patch.z_window // 2

    y0 = center.y - half_h
    y1 = center.y + half_h
    x0 = center.x - half_w
    x1 = center.x + half_w
    z0 = max(0, z_center - half_z)
    z1 = min(volume.shape[0], z_center + half_z + 1)
    crop = volume[z0:z1, y0:y1, x0:x1]
    if crop.shape[0] != patch.z_window:
        pad_before = max(0, half_z - z_center)
        pad_after = max(0, patch.z_window - crop.shape[0] - pad_before)
        crop = np.pad(crop, ((pad_before, pad_after), (0, 0), (0, 0)), mode="edge")
    return crop


def _save_patch_tensor(output_dir: Path, patch_id: str, channels: dict[str, np.ndarray], mask_crop: np.ndarray) -> Path:
    patch_dir = output_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)
    path = patch_dir / f"{patch_id}.npz"
    payload = {f"channel_{name}": array.astype(np.float32) for name, array in channels.items()}
    payload["map2_mask"] = mask_crop.astype(np.uint8)
    np.savez_compressed(path, **payload)
    return path


def extract_sample_patches(config: DatasetConfig, sample: SampleConfig) -> pd.DataFrame:
    image = open_ome_image(sample.image_path, channel_names=sample.channel_names)
    mask = load_binary_mask(sample.mask_path)
    centers = sample_patch_centers(mask=mask, patch=config.patch, sampling=config.sampling)
    channel_indices = _resolve_channel_indices(sample, config.cohort.required_channels)

    output_dir = config.output_dir.resolve()
    records: list[dict[str, object]] = []
    map2_volume = image.get_zyx(channel_indices["MAP2"])
    half_h = config.patch.height_px // 2
    half_w = config.patch.width_px // 2

    for index, center in enumerate(centers):
        patch_id = f"{sample.sample_id}_{center.group}_{index:05d}"
        z_center = _compute_z_center(map2_volume, center)
        channel_crops: dict[str, np.ndarray] = {}
        for channel_name, channel_index in channel_indices.items():
            channel_volume = image.get_zyx(channel_index)
            channel_crops[channel_name] = _crop_zyx(channel_volume, center, config.patch, z_center)

        mask_crop = mask[center.y - half_h:center.y + half_h, center.x - half_w:center.x + half_w]
        patch_path = _save_patch_tensor(output_dir=output_dir, patch_id=patch_id, channels=channel_crops, mask_crop=mask_crop)

        record = {
            "patch_id": patch_id,
            "sample_id": sample.sample_id,
            "condition": sample.condition,
            "channel_schema": config.cohort.channel_schema,
            "patch_group": center.group,
            "experiment_date": sample.experiment_date,
            "coverslip": sample.coverslip,
            "culture_batch": sample.culture_batch,
            "field_of_view": sample.field_of_view,
            "cell_id": sample.cell_id,
            "dendrite_id": sample.dendrite_id,
            "source_image": str(sample.image_path),
            "mask_image": str(sample.mask_path),
            "patch_path": str(patch_path),
            "x": center.x,
            "y": center.y,
            "z_center": z_center,
            "pixel_size_xy_um": image.pixel_size_xy_um,
            "pixel_size_z_um": image.pixel_size_z_um,
            "patch_width_px": config.patch.width_px,
            "patch_height_px": config.patch.height_px,
            "z_window": config.patch.z_window,
            "map2_overlap_fraction": center.map2_overlap_fraction,
            "distance_to_mask_px": center.distance_to_mask_px,
            "image_loader_lazy": image.is_lazy,
        }
        for channel_name, crop in channel_crops.items():
            summary = summarize_channel_crop(crop)
            for key, value in summary.items():
                record[f"{channel_name.lower()}_{key}"] = value

        records.append(record)

    return pd.DataFrame.from_records(records)
