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
        if config.patch.resume and sample_manifest_path.exists():
            print(f"[resume] {sample.sample_id} | using {sample_manifest_path.name}", flush=True)
            frame = pd.read_csv(sample_manifest_path)
        else:
            frame = extract_sample_patches(config=config, sample=sample)
            frame.to_csv(sample_manifest_path, index=False)
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
