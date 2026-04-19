from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .feature_variance_analysis import load_feature_separation_table, run_feature_variance_analysis
from .run_log import write_run_metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone clustering on feature between/within variance metrics."
    )
    parser.add_argument(
        "--separation-csv",
        required=True,
        help="Path to existing feature_cluster_separation.csv or feature_cluster_separation_with_categories.csv",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for feature variance cluster outputs. Defaults to the separation CSV parent directory.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=None,
        help="Optional fixed number of feature-variance clusters. Default uses min(4, n_features).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate the input table without running analysis")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    separation_csv = Path(args.separation_csv).resolve()
    separation_df = load_feature_separation_table(separation_csv)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else separation_csv.parent

    if args.dry_run:
        print(f"Feature separation table validated: {separation_csv}")
        print(f"Rows: {len(separation_df)}")
        print(f"Output dir: {output_dir}")
        return

    metadata_path = write_run_metadata(
        output_dir,
        pipeline_name="Feature Variance Cluster Analysis",
        command_argv=[sys.executable, "-m", "map2_patch_discovery.feature_variance_cli", *sys.argv[1:]],
        config_path=separation_csv,
        extra_lines=[
            f"input_table: `{separation_csv}`",
            f"rows: `{len(separation_df)}`",
            f"n_clusters: `{args.n_clusters if args.n_clusters is not None else 'auto'}`",
        ],
    )
    print(f"[run metadata] {metadata_path}")

    result_dir = run_feature_variance_analysis(
        separation_df=separation_df,
        output_dir=output_dir,
        n_clusters=args.n_clusters,
    )
    print(f"Feature variance cluster analysis complete: {result_dir}")


if __name__ == "__main__":
    main()
