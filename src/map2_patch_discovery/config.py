from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


VALID_SAMPLING_GROUPS = {
    "in_mask",
    "boundary",
    "near_mask_outside",
    "far_background",
}
VALID_SAMPLING_MODES = {
    "grid_map2",
    "detected_blobs",
}


@dataclass(frozen=True)
class PatchConfig:
    width_px: int
    height_px: int
    z_window: int
    stride_px: int
    max_patches_per_group: int
    save_compressed: bool
    resume: bool
    use_sample_cache: bool
    shard_by_group: bool


@dataclass(frozen=True)
class SamplingConfig:
    mode: str
    groups: list[str]
    boundary_width_px: int
    near_outside_distance_px: int
    far_background_min_distance_px: int
    random_seed: int
    blob_channels: list[str] | None = None
    blob_min_sigma_px: float = 0.75
    blob_max_sigma_px: float = 3.5
    blob_num_sigma: int = 6
    blob_threshold: float = 0.03
    blob_overlap: float = 0.5


@dataclass(frozen=True)
class CohortConfig:
    channel_schema: str
    required_channels: list[str]


@dataclass(frozen=True)
class SampleConfig:
    sample_id: str
    condition: str
    experiment_date: str
    coverslip: str
    culture_batch: str
    field_of_view: str
    image_path: Path
    mask_path: Path
    channel_names: list[str]
    cell_id: str | None = None
    dendrite_id: str | None = None


@dataclass(frozen=True)
class DatasetConfig:
    dataset_name: str
    output_dir: Path
    patch: PatchConfig
    sampling: SamplingConfig
    cohort: CohortConfig
    samples: list[SampleConfig]


def _as_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _require_keys(data: dict[str, Any], keys: list[str], section: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing required keys in {section}: {missing}")


def load_dataset_config(config_path: str | Path) -> DatasetConfig:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)

    if not isinstance(raw, dict):
        raise ValueError("Top-level dataset config must be a mapping")

    _require_keys(raw, ["dataset_name", "output_dir", "patch", "sampling", "cohort", "samples"], "root")

    patch_raw = raw["patch"]
    sampling_raw = raw["sampling"]
    cohort_raw = raw["cohort"]
    samples_raw = raw["samples"]

    _require_keys(
        patch_raw,
        ["width_px", "height_px", "z_window", "stride_px", "max_patches_per_group"],
        "patch",
    )
    _require_keys(
        sampling_raw,
        ["groups", "boundary_width_px", "near_outside_distance_px", "far_background_min_distance_px", "random_seed"],
        "sampling",
    )
    _require_keys(cohort_raw, ["channel_schema", "required_channels"], "cohort")

    base_dir = config_path.parent

    patch = PatchConfig(
        width_px=int(patch_raw["width_px"]),
        height_px=int(patch_raw["height_px"]),
        z_window=int(patch_raw["z_window"]),
        stride_px=int(patch_raw["stride_px"]),
        max_patches_per_group=int(patch_raw["max_patches_per_group"]),
        save_compressed=bool(patch_raw.get("save_compressed", False)),
        resume=bool(patch_raw.get("resume", True)),
        use_sample_cache=bool(patch_raw.get("use_sample_cache", True)),
        shard_by_group=bool(patch_raw.get("shard_by_group", True)),
    )
    sampling = SamplingConfig(
        mode=str(sampling_raw.get("mode", "grid_map2")).lower(),
        groups=[str(group) for group in sampling_raw["groups"]],
        boundary_width_px=int(sampling_raw["boundary_width_px"]),
        near_outside_distance_px=int(sampling_raw["near_outside_distance_px"]),
        far_background_min_distance_px=int(sampling_raw["far_background_min_distance_px"]),
        random_seed=int(sampling_raw["random_seed"]),
        blob_channels=(
            None
            if sampling_raw.get("blob_channels") is None
            else [str(name) for name in sampling_raw.get("blob_channels", [])]
        ),
        blob_min_sigma_px=float(sampling_raw.get("blob_min_sigma_px", 0.75)),
        blob_max_sigma_px=float(sampling_raw.get("blob_max_sigma_px", 3.5)),
        blob_num_sigma=int(sampling_raw.get("blob_num_sigma", 6)),
        blob_threshold=float(sampling_raw.get("blob_threshold", 0.03)),
        blob_overlap=float(sampling_raw.get("blob_overlap", 0.5)),
    )
    cohort = CohortConfig(
        channel_schema=str(cohort_raw["channel_schema"]),
        required_channels=[str(name) for name in cohort_raw["required_channels"]],
    )

    samples: list[SampleConfig] = []
    for idx, sample_raw in enumerate(samples_raw):
        _require_keys(
            sample_raw,
            [
                "sample_id",
                "condition",
                "experiment_date",
                "coverslip",
                "culture_batch",
                "field_of_view",
                "image_path",
                "mask_path",
                "channel_names",
            ],
            f"samples[{idx}]",
        )
        sample = SampleConfig(
            sample_id=str(sample_raw["sample_id"]),
            condition=str(sample_raw["condition"]),
            experiment_date=str(sample_raw["experiment_date"]),
            coverslip=str(sample_raw["coverslip"]),
            culture_batch=str(sample_raw["culture_batch"]),
            field_of_view=str(sample_raw["field_of_view"]),
            image_path=_as_path(base_dir, str(sample_raw["image_path"])),
            mask_path=_as_path(base_dir, str(sample_raw["mask_path"])),
            channel_names=[str(name) for name in sample_raw["channel_names"]],
            cell_id=None if sample_raw.get("cell_id") is None else str(sample_raw["cell_id"]),
            dendrite_id=None if sample_raw.get("dendrite_id") is None else str(sample_raw["dendrite_id"]),
        )
        samples.append(sample)

    config = DatasetConfig(
        dataset_name=str(raw["dataset_name"]),
        output_dir=_as_path(base_dir, str(raw["output_dir"])),
        patch=patch,
        sampling=sampling,
        cohort=cohort,
        samples=samples,
    )
    validate_dataset_config(config)
    return config


def validate_dataset_config(config: DatasetConfig) -> None:
    if config.patch.width_px <= 0 or config.patch.height_px <= 0:
        raise ValueError("Patch width and height must be positive")
    if config.patch.width_px % 2 != 0 or config.patch.height_px % 2 != 0:
        raise ValueError("Patch width and height must be even for centered extraction")
    if config.patch.z_window <= 0:
        raise ValueError("Patch z_window must be positive")
    if config.patch.stride_px <= 0:
        raise ValueError("Patch stride must be positive")
    if config.patch.max_patches_per_group <= 0:
        raise ValueError("max_patches_per_group must be positive")
    if config.patch.z_window % 2 == 0:
        raise ValueError("Patch z_window must be odd for centered extraction")
    if not config.cohort.required_channels:
        raise ValueError("At least one required channel must be defined")
    if not config.sampling.groups:
        raise ValueError("At least one sampling group must be defined")
    if config.sampling.mode not in VALID_SAMPLING_MODES:
        valid_modes = ", ".join(sorted(VALID_SAMPLING_MODES))
        raise ValueError(f"Invalid sampling.mode: {config.sampling.mode}. Valid modes: {valid_modes}")
    invalid_groups = [group for group in config.sampling.groups if group not in VALID_SAMPLING_GROUPS]
    if invalid_groups:
        valid_groups = ", ".join(sorted(VALID_SAMPLING_GROUPS))
        raise ValueError(f"Invalid sampling.groups values: {invalid_groups}. Valid groups: {valid_groups}")
    if config.sampling.blob_min_sigma_px <= 0:
        raise ValueError("sampling.blob_min_sigma_px must be positive")
    if config.sampling.blob_max_sigma_px <= config.sampling.blob_min_sigma_px:
        raise ValueError("sampling.blob_max_sigma_px must be greater than sampling.blob_min_sigma_px")
    if config.sampling.blob_num_sigma <= 0:
        raise ValueError("sampling.blob_num_sigma must be positive")
    if config.sampling.blob_threshold <= 0:
        raise ValueError("sampling.blob_threshold must be positive")
    if not 0 <= config.sampling.blob_overlap <= 1:
        raise ValueError("sampling.blob_overlap must be between 0 and 1")
    if not config.samples:
        raise ValueError("At least one sample must be defined")

    seen_sample_ids: set[str] = set()
    for sample in config.samples:
        if sample.sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id: {sample.sample_id}")
        seen_sample_ids.add(sample.sample_id)

        if not sample.image_path.exists():
            raise ValueError(f"Image file not found: {sample.image_path}")
        if not sample.mask_path.exists():
            raise ValueError(f"Mask file not found: {sample.mask_path}")
        if len(sample.channel_names) != len(set(sample.channel_names)):
            raise ValueError(f"Duplicate channel names in sample {sample.sample_id}: {sample.channel_names}")

        missing = [name for name in config.cohort.required_channels if name not in sample.channel_names]
        if missing:
            raise ValueError(f"Sample {sample.sample_id} missing required channels: {missing}")
        if config.sampling.blob_channels is not None:
            invalid_blob_channels = [name for name in config.sampling.blob_channels if name not in sample.channel_names]
            if invalid_blob_channels:
                raise ValueError(
                    f"Sample {sample.sample_id} missing sampling.blob_channels: {invalid_blob_channels}"
                )
