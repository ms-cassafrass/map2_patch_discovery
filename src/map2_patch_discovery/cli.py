from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_dataset_config
from .pipeline import run_patch_extraction


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract MAP2-relative local patches from OME-TIFF datasets.")
    parser.add_argument("--config", required=True, help="Path to dataset YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate config and inputs without extracting patches")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_dataset_config(Path(args.config))
    if args.dry_run:
        print(f"Config validated successfully for dataset: {config.dataset_name}")
        print(f"Samples: {len(config.samples)}")
        print(f"Channel schema: {config.cohort.channel_schema}")
        return
    manifest_path = run_patch_extraction(config)
    print(f"Patch extraction complete: {manifest_path}")


if __name__ == "__main__":
    main()
