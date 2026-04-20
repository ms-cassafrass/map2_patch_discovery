from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
try:
    from skimage.feature import blob_log
except ImportError:  # pragma: no cover
    blob_log = None

from .config import DatasetConfig, PatchConfig, SampleConfig
from .ome import load_binary_mask, open_ome_image
from .sampling import PatchCenter, classify_patch_center, compute_signed_distance, sample_patch_centers
from .summaries import summarize_channel_crop


PROGRESS_EVERY_PATCHES = 25


def _resolve_channel_indices(sample: SampleConfig, required_channels: list[str]) -> dict[str, int]:
    name_to_index = {name: idx for idx, name in enumerate(sample.channel_names)}
    missing = [name for name in required_channels if name not in name_to_index]
    if missing:
        raise ValueError(f"Sample {sample.sample_id} missing required channels: {missing}")
    return {name: name_to_index[name] for name in required_channels}


def _normalize_zero_one(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def _resolve_blob_detection_channels(config: DatasetConfig, sample: SampleConfig) -> list[str]:
    if config.sampling.blob_channels is not None:
        channels = [str(channel) for channel in config.sampling.blob_channels]
    else:
        channels = [str(channel) for channel in config.cohort.required_channels if str(channel) != "MAP2"]
    return [channel for channel in channels if channel in sample.channel_names and channel != "MAP2"]


def _detect_blob_patch_centers(
    *,
    config: DatasetConfig,
    sample: SampleConfig,
    channel_volumes: dict[str, np.ndarray],
    mask: np.ndarray,
) -> list[PatchCenter]:
    if blob_log is None:
        raise ValueError("scikit-image is required for sampling.mode=detected_blobs")

    detection_channels = _resolve_blob_detection_channels(config=config, sample=sample)
    if not detection_channels:
        raise ValueError(f"Sample {sample.sample_id} has no non-MAP2 channels available for blob detection.")

    signed_distance, outside_distance = compute_signed_distance(mask)
    seen: set[tuple[str, int, int]] = set()
    centers: list[PatchCenter] = []

    for channel_name in detection_channels:
        projection = np.max(np.asarray(channel_volumes[channel_name], dtype=np.float32), axis=0)
        norm = _normalize_zero_one(projection)
        if not np.any(norm > 0):
            continue
        min_dim = min(norm.shape)
        max_sigma = min(float(config.sampling.blob_max_sigma_px), max(float(config.sampling.blob_min_sigma_px) + 1e-3, float(min_dim) / 4.0))
        blobs = blob_log(
            norm,
            min_sigma=float(config.sampling.blob_min_sigma_px),
            max_sigma=max_sigma,
            num_sigma=int(config.sampling.blob_num_sigma),
            threshold=float(config.sampling.blob_threshold),
            overlap=float(config.sampling.blob_overlap),
            exclude_border=False,
        )
        if len(blobs) == 0:
            print(f"  blob detect {channel_name}: 0 centers", flush=True)
            continue

        kept_for_channel = 0
        for y, x, sigma in np.asarray(blobs, dtype=np.float32):
            yi = int(np.round(float(y)))
            xi = int(np.round(float(x)))
            key = (channel_name, yi, xi)
            if key in seen:
                continue
            seen.add(key)
            center = classify_patch_center(
                mask=mask,
                patch=config.patch,
                sampling=config.sampling,
                y=yi,
                x=xi,
                detection_channel=channel_name,
                detected_radius_px=float(np.sqrt(2.0) * float(sigma)),
                signed_distance=signed_distance,
                outside_distance=outside_distance,
            )
            if center is None:
                continue
            centers.append(center)
            kept_for_channel += 1
        print(f"  blob detect {channel_name}: kept={kept_for_channel}", flush=True)

    return centers


def _compute_z_centers(channel_volumes: dict[str, np.ndarray], centers: list[PatchCenter]) -> np.ndarray:
    if not centers:
        return np.zeros((0,), dtype=np.int32)

    default_channel = "MAP2" if "MAP2" in channel_volumes else next(iter(channel_volumes))
    grouped_indices: dict[str, list[int]] = {}
    for idx, center in enumerate(centers):
        source_channel = center.detection_channel or default_channel
        if source_channel not in channel_volumes:
            source_channel = default_channel
        grouped_indices.setdefault(source_channel, []).append(idx)

    z_centers = np.zeros((len(centers),), dtype=np.int32)
    for channel_name, indices in grouped_indices.items():
        volume = channel_volumes[channel_name]
        ys = np.array([centers[idx].y for idx in indices], dtype=np.intp)
        xs = np.array([centers[idx].x for idx in indices], dtype=np.intp)
        profiles = np.asarray(volume[:, ys, xs], dtype=np.float32)
        if profiles.ndim != 2:
            raise ValueError(f"Unexpected profile array shape for {channel_name}: {profiles.shape}")
        finite_any = np.any(np.isfinite(profiles), axis=0)
        filled = np.where(np.isfinite(profiles), profiles, -np.inf)
        group_centers = np.argmax(filled, axis=0).astype(np.int32, copy=False)
        group_centers[~finite_any] = int(volume.shape[0] // 2)
        z_centers[np.asarray(indices, dtype=np.intp)] = group_centers
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


def _cache_metadata_path(output_dir: Path, sample_id: str) -> Path:
    return _cache_dir(output_dir, sample_id) / "cache_metadata.json"


def _path_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_sample_cache_metadata(
    sample: SampleConfig,
    channel_indices: dict[str, int],
) -> dict[str, object]:
    return {
        "sample_id": sample.sample_id,
        "image": _path_fingerprint(sample.image_path),
        "mask": _path_fingerprint(sample.mask_path),
        "channel_names": [str(name) for name in sample.channel_names],
        "channel_indices": {str(name): int(index) for name, index in sorted(channel_indices.items())},
    }


def _read_sample_cache_metadata(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cache_is_current(
    output_dir: Path,
    sample: SampleConfig,
    channel_indices: dict[str, int],
) -> bool:
    metadata_path = _cache_metadata_path(output_dir, sample.sample_id)
    cached_metadata = _read_sample_cache_metadata(metadata_path)
    current_metadata = _build_sample_cache_metadata(sample=sample, channel_indices=channel_indices)
    if cached_metadata != current_metadata:
        return False

    required_paths = [_cache_array_path(output_dir, sample.sample_id, "map2_mask_2d")]
    required_paths.extend(
        _cache_array_path(output_dir, sample.sample_id, f"channel_{channel_name}")
        for channel_name in channel_indices
    )
    return all(path.exists() for path in required_paths)


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
    metadata_path = _cache_metadata_path(output_dir, sample.sample_id)
    cache_is_current = _cache_is_current(
        output_dir=output_dir,
        sample=sample,
        channel_indices=channel_indices,
    )
    if not cache_is_current:
        print(f"  sample cache invalidated | sample={sample.sample_id}", flush=True)

    mask_cache_path = _cache_array_path(output_dir, sample.sample_id, "map2_mask_2d")
    if cache_is_current and mask_cache_path.exists():
        cached_mask = np.load(mask_cache_path, mmap_mode="r")
    else:
        np.save(mask_cache_path, np.asarray(mask, dtype=np.uint8))
        cached_mask = np.load(mask_cache_path, mmap_mode="r")

    channel_volumes: dict[str, np.ndarray] = {}
    for channel_name, channel_index in channel_indices.items():
        cache_path = _cache_array_path(output_dir, sample.sample_id, f"channel_{channel_name}")
        if cache_is_current and cache_path.exists():
            channel_volumes[channel_name] = np.load(cache_path, mmap_mode="r")
            continue
        volume = image.get_zyx(channel_index)
        np.save(cache_path, np.asarray(volume, dtype=np.float32))
        channel_volumes[channel_name] = np.load(cache_path, mmap_mode="r")

    metadata = _build_sample_cache_metadata(sample=sample, channel_indices=channel_indices)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")

    return channel_volumes, np.asarray(cached_mask, dtype=bool)


def extract_sample_patches(config: DatasetConfig, sample: SampleConfig) -> pd.DataFrame:
    t_sample_start = perf_counter()
    t0 = perf_counter()
    image = open_ome_image(sample.image_path, channel_names=sample.channel_names)
    t_image = perf_counter() - t0

    t0 = perf_counter()
    mask = load_binary_mask(sample.mask_path)
    t_mask = perf_counter() - t0

    channel_indices = _resolve_channel_indices(sample, config.cohort.required_channels)
    print(f"  image={_short_name(sample.image_path)}", flush=True)
    print(f"  mask={_short_name(sample.mask_path)}", flush=True)
    print(f"  times open={t_image:.1f}s mask={t_mask:.1f}s", flush=True)

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
    t_cache = perf_counter() - t0
    print(f"  channel volumes ready | cache_load={t_cache:.1f}s", flush=True)
    t0 = perf_counter()
    if config.sampling.mode == "detected_blobs":
        centers = _detect_blob_patch_centers(
            config=config,
            sample=sample,
            channel_volumes=channel_volumes,
            mask=mask,
        )
    else:
        centers = sample_patch_centers(mask=mask, patch=config.patch, sampling=config.sampling)
    t_sampling = perf_counter() - t0
    print(
        f"[sample] {sample.sample_id} | {sample.condition} | "
        f"{'lazy' if image.is_lazy else 'eager'} | mode={config.sampling.mode} | "
        f"centers={len(centers)} | {_format_group_counts(centers)}",
        flush=True,
    )
    print(f"  sampling={t_sampling:.1f}s", flush=True)
    half_h = config.patch.height_px // 2
    half_w = config.patch.width_px // 2
    t0 = perf_counter()
    z_centers = _compute_z_centers(channel_volumes, centers)
    t_z = perf_counter() - t0
    print(f"  z-centers ready | compute={t_z:.1f}s", flush=True)

    total = len(centers)
    shard_channels: dict[str, dict[str, list[np.ndarray]]] = {}
    shard_masks: dict[str, list[np.ndarray]] = {}
    shard_patch_ids: dict[str, list[str]] = {}
    shard_paths: dict[str, Path] = {}
    t_crop_total = 0.0
    for index, center in enumerate(centers):
        if center.detection_channel:
            patch_id = f"{sample.sample_id}_{center.group}_{center.detection_channel.lower()}_{index:05d}"
        else:
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
            "sampling_mode": config.sampling.mode,
            "detection_channel": center.detection_channel,
            "detected_radius_px": center.detected_radius_px,
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
            detail = center.detection_channel if center.detection_channel is not None else center.group
            print(f"  patches {done}/{total} | group={center.group} | source={detail}", flush=True)

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
