from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from .config import DatasetConfig, PatchConfig, SampleConfig
from .ome import load_binary_mask, open_ome_image
from .sampling import PatchCenter, sample_patch_centers
from .summaries import summarize_channel_crop


PROGRESS_EVERY_PATCHES = 25


def _resolve_channel_indices(sample: SampleConfig, required_channels: list[str]) -> dict[str, int]:
    name_to_index = {name: idx for idx, name in enumerate(sample.channel_names)}
    missing = [name for name in required_channels if name not in name_to_index]
    if missing:
        raise ValueError(f"Sample {sample.sample_id} missing required channels: {missing}")
    return {name: name_to_index[name] for name in required_channels}


def _compute_z_centers(channel_volume: np.ndarray, centers: list[PatchCenter]) -> np.ndarray:
    if not centers:
        return np.zeros((0,), dtype=np.int32)
    ys = np.array([center.y for center in centers], dtype=np.intp)
    xs = np.array([center.x for center in centers], dtype=np.intp)
    profiles = np.asarray(channel_volume[:, ys, xs], dtype=np.float32)
    if profiles.ndim != 2:
        raise ValueError(f"Unexpected MAP2 profile array shape: {profiles.shape}")
    finite_any = np.any(np.isfinite(profiles), axis=0)
    filled = np.where(np.isfinite(profiles), profiles, -np.inf)
    z_centers = np.argmax(filled, axis=0).astype(np.int32, copy=False)
    z_centers[~finite_any] = int(channel_volume.shape[0] // 2)
    return z_centers


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


def _save_patch_shard(
    output_dir: Path,
    sample_id: str,
    group_name: str,
    channels: dict[str, list[np.ndarray]],
    mask_crops: list[np.ndarray],
    patch_ids: list[str],
    save_compressed: bool,
) -> Path:
    shard_dir = output_dir / "patches"
    shard_dir.mkdir(parents=True, exist_ok=True)
    path = shard_dir / f"{sample_id}_{group_name}.npz"
    payload: dict[str, np.ndarray] = {
        "patch_ids": np.asarray(patch_ids),
        "map2_mask": np.stack([np.asarray(crop, dtype=np.uint8) for crop in mask_crops], axis=0),
    }
    for name, arrays in channels.items():
        payload[f"channel_{name}"] = np.stack([np.asarray(array, dtype=np.float32) for array in arrays], axis=0)
    if save_compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)
    return path


def _format_group_counts(centers: list[PatchCenter]) -> str:
    counts: dict[str, int] = {}
    for center in centers:
        counts[center.group] = counts.get(center.group, 0) + 1
    if not counts:
        return "no patch centers"
    ordered = [f"{group}={count}" for group, count in sorted(counts.items())]
    return ", ".join(ordered)


def _short_name(path: Path) -> str:
    return path.name


def _shard_key(config: DatasetConfig, group_name: str) -> str:
    return group_name if config.patch.shard_by_group else "all_patches"


def _cache_dir(output_dir: Path, sample_id: str) -> Path:
    return output_dir / "cache" / sample_id


def _cache_array_path(output_dir: Path, sample_id: str, stem: str) -> Path:
    return _cache_dir(output_dir, sample_id) / f"{stem}.npy"


def _load_or_build_sample_cache(
    output_dir: Path,
    sample: SampleConfig,
    image,
    mask: np.ndarray,
    channel_indices: dict[str, int],
    use_sample_cache: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if not use_sample_cache:
        channel_volumes = {
            channel_name: image.get_zyx(channel_index)
            for channel_name, channel_index in channel_indices.items()
        }
        return channel_volumes, np.asarray(mask, dtype=bool)

    cache_root = _cache_dir(output_dir, sample.sample_id)
    cache_root.mkdir(parents=True, exist_ok=True)

    mask_cache_path = _cache_array_path(output_dir, sample.sample_id, "map2_mask_2d")
    if mask_cache_path.exists():
        cached_mask = np.load(mask_cache_path, mmap_mode="r")
    else:
        np.save(mask_cache_path, np.asarray(mask, dtype=np.uint8))
        cached_mask = np.load(mask_cache_path, mmap_mode="r")

    channel_volumes: dict[str, np.ndarray] = {}
    for channel_name, channel_index in channel_indices.items():
        cache_path = _cache_array_path(output_dir, sample.sample_id, f"channel_{channel_name}")
        if cache_path.exists():
            channel_volumes[channel_name] = np.load(cache_path, mmap_mode="r")
            continue
        volume = image.get_zyx(channel_index)
        np.save(cache_path, np.asarray(volume, dtype=np.float32))
        channel_volumes[channel_name] = np.load(cache_path, mmap_mode="r")

    return channel_volumes, np.asarray(cached_mask, dtype=bool)


def extract_sample_patches(config: DatasetConfig, sample: SampleConfig) -> pd.DataFrame:
    t_sample_start = perf_counter()
    t0 = perf_counter()
    image = open_ome_image(sample.image_path, channel_names=sample.channel_names)
    t_image = perf_counter() - t0

    t0 = perf_counter()
    mask = load_binary_mask(sample.mask_path)
    t_mask = perf_counter() - t0

    t0 = perf_counter()
    centers = sample_patch_centers(mask=mask, patch=config.patch, sampling=config.sampling)
    t_sampling = perf_counter() - t0
    channel_indices = _resolve_channel_indices(sample, config.cohort.required_channels)
    print(
        f"[sample] {sample.sample_id} | {sample.condition} | "
        f"{'lazy' if image.is_lazy else 'eager'} | centers={len(centers)} | {_format_group_counts(centers)}",
        flush=True,
    )
    print(f"  image={_short_name(sample.image_path)}", flush=True)
    print(f"  mask={_short_name(sample.mask_path)}", flush=True)
    print(f"  times open={t_image:.1f}s mask={t_mask:.1f}s sample={t_sampling:.1f}s", flush=True)

    output_dir = config.output_dir.resolve()
    records: list[dict[str, object]] = []
    print(
        f"  cache={'on' if config.patch.use_sample_cache else 'off'} | "
        f"patch_compression={'on' if config.patch.save_compressed else 'off'} | "
        f"shard_by_group={'on' if config.patch.shard_by_group else 'off'}",
        flush=True,
    )
    print("  loading channel volumes...", flush=True)
    t0 = perf_counter()
    channel_volumes, mask = _load_or_build_sample_cache(
        output_dir=output_dir,
        sample=sample,
        image=image,
        mask=mask,
        channel_indices=channel_indices,
        use_sample_cache=config.patch.use_sample_cache,
    )
    map2_volume = channel_volumes["MAP2"]
    t_cache = perf_counter() - t0
    print(f"  channel volumes ready | cache_load={t_cache:.1f}s", flush=True)
    half_h = config.patch.height_px // 2
    half_w = config.patch.width_px // 2
    t0 = perf_counter()
    z_centers = _compute_z_centers(map2_volume, centers)
    t_z = perf_counter() - t0
    print(f"  z-centers ready | compute={t_z:.1f}s", flush=True)

    total = len(centers)
    shard_channels: dict[str, dict[str, list[np.ndarray]]] = {}
    shard_masks: dict[str, list[np.ndarray]] = {}
    shard_patch_ids: dict[str, list[str]] = {}
    shard_paths: dict[str, Path] = {}
    t_crop_total = 0.0
    for index, center in enumerate(centers):
        patch_id = f"{sample.sample_id}_{center.group}_{index:05d}"
        z_center = int(z_centers[index])
        t_crop0 = perf_counter()
        channel_crops: dict[str, np.ndarray] = {}
        for channel_name, channel_volume in channel_volumes.items():
            channel_crops[channel_name] = _crop_zyx(channel_volume, center, config.patch, z_center)

        mask_crop = mask[center.y - half_h:center.y + half_h, center.x - half_w:center.x + half_w]
        t_crop_total += perf_counter() - t_crop0

        shard_key = _shard_key(config, center.group)
        group_channels = shard_channels.setdefault(
            shard_key,
            {channel_name: [] for channel_name in channel_crops},
        )
        for channel_name, crop in channel_crops.items():
            group_channels[channel_name].append(crop)
        shard_masks.setdefault(shard_key, []).append(mask_crop)
        shard_patch_ids.setdefault(shard_key, []).append(patch_id)

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
            "patch_path": None,
            "shard_path": None,
            "shard_index": len(shard_patch_ids[shard_key]) - 1,
            "shard_group": shard_key,
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
        done = index + 1
        if done == 1 or done == total or done % PROGRESS_EVERY_PATCHES == 0:
            print(f"  patches {done}/{total} | group={center.group}", flush=True)

    t0 = perf_counter()
    for group_name, channels_by_name in shard_channels.items():
        shard_path = _save_patch_shard(
            output_dir=output_dir,
            sample_id=sample.sample_id,
            group_name=group_name,
            channels=channels_by_name,
            mask_crops=shard_masks[group_name],
            patch_ids=shard_patch_ids[group_name],
            save_compressed=config.patch.save_compressed,
        )
        shard_paths[group_name] = shard_path
    t_write = perf_counter() - t0

    for record in records:
        group_name = str(record["shard_group"])
        shard_path = shard_paths[group_name]
        record["patch_path"] = str(shard_path)
        record["shard_path"] = str(shard_path)

    print(
        f"  times z={t_z:.1f}s crop={t_crop_total:.1f}s write={t_write:.1f}s total={perf_counter() - t_sample_start:.1f}s",
        flush=True,
    )
    print(f"[sample done] {sample.sample_id} | patches={len(records)}", flush=True)
    return pd.DataFrame.from_records(records)
