from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import DatasetConfig, SampleConfig
from .patches import extract_sample_patches


def _sample_manifest_metadata_path(sample_manifest_path: Path) -> Path:
    return sample_manifest_path.with_suffix(".metadata.json")


def _path_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _build_sample_manifest_metadata(config: DatasetConfig, sample: SampleConfig) -> dict[str, object]:
    return {
        "dataset_name": config.dataset_name,
        "sample_id": sample.sample_id,
        "source_image": _path_fingerprint(sample.image_path),
        "mask_image": _path_fingerprint(sample.mask_path),
        "channel_names": [str(name) for name in sample.channel_names],
        "patch": {
            "width_px": int(config.patch.width_px),
            "height_px": int(config.patch.height_px),
            "z_window": int(config.patch.z_window),
            "stride_px": int(config.patch.stride_px),
            "max_patches_per_group": int(config.patch.max_patches_per_group),
            "save_compressed": bool(config.patch.save_compressed),
            "shard_by_group": bool(config.patch.shard_by_group),
        },
        "sampling": {
            "mode": str(config.sampling.mode),
            "groups": [str(group) for group in config.sampling.groups],
            "boundary_width_px": int(config.sampling.boundary_width_px),
            "near_outside_distance_px": int(config.sampling.near_outside_distance_px),
            "far_background_min_distance_px": int(config.sampling.far_background_min_distance_px),
            "random_seed": int(config.sampling.random_seed),
            "blob_channels": (
                None
                if config.sampling.blob_channels is None
                else [str(name) for name in config.sampling.blob_channels]
            ),
            "blob_min_sigma_px": float(config.sampling.blob_min_sigma_px),
            "blob_max_sigma_px": float(config.sampling.blob_max_sigma_px),
            "blob_num_sigma": int(config.sampling.blob_num_sigma),
            "blob_threshold": float(config.sampling.blob_threshold),
            "blob_overlap": float(config.sampling.blob_overlap),
        },
        "cohort": {
            "channel_schema": str(config.cohort.channel_schema),
            "required_channels": [str(name) for name in config.cohort.required_channels],
        },
    }


def _read_sample_manifest_metadata(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _sample_manifest_is_current(
    *,
    config: DatasetConfig,
    sample: SampleConfig,
    sample_manifest_path: Path,
) -> bool:
    metadata_path = _sample_manifest_metadata_path(sample_manifest_path)
    cached_metadata = _read_sample_manifest_metadata(metadata_path)
    current_metadata = _build_sample_manifest_metadata(config=config, sample=sample)
    if cached_metadata != current_metadata:
        return False
    if not sample_manifest_path.exists():
        return False

    try:
        manifest = pd.read_csv(sample_manifest_path)
    except Exception:
        return False

    shard_columns = [column for column in ("shard_path", "patch_path") if column in manifest.columns]
    if not shard_columns:
        return False

    referenced_paths: set[Path] = set()
    for column in shard_columns:
        for value in manifest[column].dropna().astype(str):
            if not value:
                continue
            referenced_paths.add(Path(value).resolve())
    return all(path.exists() for path in referenced_paths)


def run_patch_extraction(config: DatasetConfig) -> Path:
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    sample_manifests_dir = manifests_dir / "samples"
    sample_manifests_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {config.dataset_name} | samples={len(config.samples)}", flush=True)
    try:
        short_output = output_dir.relative_to(Path.cwd())
    except ValueError:
        short_output = output_dir
    print(f"[out] {short_output}", flush=True)

    all_frames: list[pd.DataFrame] = []
    for sample_index, sample in enumerate(config.samples, start=1):
        print(f"[sample {sample_index}/{len(config.samples)}]", flush=True)
        sample_manifest_path = sample_manifests_dir / f"{sample.sample_id}.csv"
        metadata_path = _sample_manifest_metadata_path(sample_manifest_path)
        if config.patch.resume and _sample_manifest_is_current(
            config=config,
            sample=sample,
            sample_manifest_path=sample_manifest_path,
        ):
            print(f"[resume] {sample.sample_id} | using {sample_manifest_path.name}", flush=True)
            frame = pd.read_csv(sample_manifest_path)
        else:
            if config.patch.resume and sample_manifest_path.exists():
                print(f"[resume invalidated] {sample.sample_id} | rebuilding sample outputs", flush=True)
            frame = extract_sample_patches(config=config, sample=sample)
            frame.to_csv(sample_manifest_path, index=False)
            metadata = _build_sample_manifest_metadata(config=config, sample=sample)
            metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
            print(f"[sample manifest] {sample_manifest_path.name} | rows={len(frame)}", flush=True)
        all_frames.append(frame)

    manifest = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    manifest_path = manifests_dir / f"{config.dataset_name}_patch_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    try:
        short_manifest = manifest_path.relative_to(Path.cwd())
    except ValueError:
        short_manifest = manifest_path
    print(f"[manifest] {short_manifest} | rows={len(manifest)}", flush=True)
    return manifest_path
