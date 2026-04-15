from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DatasetConfig
from .patches import extract_sample_patches


def run_patch_extraction(config: DatasetConfig) -> Path:
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifests_dir = output_dir / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)

    all_frames: list[pd.DataFrame] = []
    for sample in config.samples:
        frame = extract_sample_patches(config=config, sample=sample)
        all_frames.append(frame)

    manifest = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    manifest_path = manifests_dir / f"{config.dataset_name}_patch_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    return manifest_path
