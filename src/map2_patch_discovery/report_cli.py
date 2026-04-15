from __future__ import annotations

import argparse
from pathlib import Path

from .latent_report import run_latent_report
from .report_config import load_latent_report_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline latent reporting on extracted MAP2-relative patches.")
    parser.add_argument("--config", required=True, help="Path to latent report YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate the report config without running analysis")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_latent_report_config(Path(args.config))
    if args.dry_run:
        print(f"Latent report config validated: {config.manifest_path}")
        print(f"Channels: {', '.join(config.features.channels)}")
        return
    output_dir = run_latent_report(config)
    print(f"Latent report complete: {output_dir}")


if __name__ == "__main__":
    main()
