from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_dataset_config
from .pipeline import run_patch_extraction
from .run_log import write_run_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract MAP2-relative local patches from OME-TIFF datasets.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and inputs without extracting patches")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_dataset_config(Path(args.config))
    print(f"[config] {config_path.name}", flush=True)
    if args.dry_run:
        print(
            f"[ok] {config.dataset_name} | samples={len(config.samples)} | schema={config.cohort.channel_schema} | "
            f"resume={'on' if config.patch.resume else 'off'} | "
            f"sample_cache={'on' if config.patch.use_sample_cache else 'off'} | "
            f"patch_compression={'on' if config.patch.save_compressed else 'off'} | "
            f"shard_by_group={'on' if config.patch.shard_by_group else 'off'}",
        )
        return
    metadata_path = write_run_metadata(
        config.output_dir,
        pipeline_name="Patch Extraction",
        command_argv=[sys.executable, "-m", "map2_patch_discovery.cli", *sys.argv[1:]],
        config_path=config_path,
        extra_lines=[
            f"dataset_name: `{config.dataset_name}`",
            f"samples: `{len(config.samples)}`",
            f"channel_schema: `{config.cohort.channel_schema}`",
        ],
    )
    print(f"[run metadata] {metadata_path}", flush=True)
    manifest_path = run_patch_extraction(config)
    print(f"[done] {manifest_path}")


if __name__ == "__main__":
    main()
