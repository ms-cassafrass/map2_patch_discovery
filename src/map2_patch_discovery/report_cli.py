from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .report_config import load_latent_report_config
from .run_log import write_run_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline latent reporting on extracted MAP2-relative patches.")
    parser.add_argument("--config", required=True, help="Path to latent report YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Validate the report config without running analysis")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_latent_report_config(Path(args.config))
    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str((repo_root / ".mplconfig").resolve()))
    from .latent_report import run_latent_report
    if args.dry_run:
        print(f"Latent report config validated: {config.manifest_path}")
        print(f"Channels: {', '.join(config.features.channels)}")
        return
    metadata_path = write_run_metadata(
        config.output_dir,
        pipeline_name="Latent Report",
        command_argv=[sys.executable, "-m", "map2_patch_discovery.report_cli", *sys.argv[1:]],
        config_path=config_path,
        extra_lines=[
            f"manifest_path: `{config.manifest_path}`",
            f"channels: `{', '.join(config.features.channels)}`",
            (
                f"feature_variance_filter: cluster `{config.features.feature_variance_cluster}` from "
                f"`{config.features.feature_variance_csv}`"
                if config.features.feature_variance_csv is not None
                else "feature_variance_filter: `none`"
            ),
            f"clustering: `{config.clustering.method}` with `{config.clustering.n_clusters}` clusters",
        ],
    )
    print(f"[run metadata] {metadata_path}")
    output_dir = run_latent_report(config)
    print(f"Latent report complete: {output_dir}")


if __name__ == "__main__":
    main()
