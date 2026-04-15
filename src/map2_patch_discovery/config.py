from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PatchConfig:
    width_px: int
    height_px: int
    z_window: int
    stride_px: int
    max_patches_per_group: int


@dataclass(frozen=True)
class SamplingConfig:
    groups: list[str]
    boundary_width_px: int
    near_outside_distance_px: int
    far_background_min_distance_px: int
    random_seed: int


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

    _require_keys(patch_raw, ["width_px", "height_px", "z_window", "stride_px", "max_patches_per_group"], "patch")
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
    )
    sampling = SamplingConfig(
        groups=[str(group) for group in sampling_raw["groups"]],
        boundary_width_px=int(sampling_raw["boundary_width_px"]),
        near_outside_distance_px=int(sampling_raw["near_outside_distance_px"]),
        far_background_min_distance_px=int(sampling_raw["far_background_min_distance_px"]),
        random_seed=int(sampling_raw["random_seed"]),
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
    if not config.cohort.required_channels:
        raise ValueError("At least one required channel must be defined")
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
