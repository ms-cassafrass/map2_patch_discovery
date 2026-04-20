from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .report_config import load_latent_report_config
from .run_log import write_run_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute full PCA variance thresholds from a latent-report config using cached engineered features."
    )
    parser.add_argument("--config", required=True, help="Path to latent report YAML config")
    parser.add_argument(
        "--output-dir",
        help="Optional output directory. Defaults to <latent output dir>/pca_variance_analysis",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate config and show intended output path")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_latent_report_config(config_path)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (config.output_dir.resolve() / "pca_variance_analysis")

    repo_root = Path(__file__).resolve().parents[2]
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", str((repo_root / ".mplconfig").resolve()))

    if args.dry_run:
        print(f"PCA variance config validated: {config.manifest_path}")
        print(f"Channels: {', '.join(config.features.channels)}")
        print(f"Output dir: {output_dir}")
        return

    from .pca_variance import run_pca_variance_analysis

    metadata_path = write_run_metadata(
        output_dir,
        pipeline_name="PCA Variance Threshold Analysis",
        command_argv=[sys.executable, "-m", "map2_patch_discovery.pca_variance_cli", *sys.argv[1:]],
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
            f"scaler: `{config.preprocessing.scaler}`",
            f"map2_feature_policy: `{config.features.map2_feature_policy}`",
        ],
    )
    print(f"[run metadata] {metadata_path}")
    result_dir = run_pca_variance_analysis(config=config, output_dir=output_dir)
    print(f"PCA variance analysis complete: {result_dir}")


if __name__ == "__main__":
    main()
