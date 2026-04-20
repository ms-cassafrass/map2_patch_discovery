from __future__ import annotations

import argparse
from pathlib import Path

from .pc_overlay import (
    create_cluster_component_overlay,
    create_cluster_patch_center_overlay,
    create_cluster_puncta_candidate_overlay,
    create_confidence_filtered_cluster_overlay,
    create_principal_component_overlay,
    create_segmented_blob_overlay,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render one sample as a channel max projection with MAP2 mask and either PCA-score or cluster overlay."
    )
    parser.add_argument("--report-csv", required=True, help="Path to patch_latent_report.csv")
    parser.add_argument("--channel", required=True, help="Raw image channel to display, e.g. FLAG, HA, SHANK2")
    parser.add_argument(
        "--overlay",
        choices=("pc", "cluster", "patch-centers", "puncta-candidates", "confidence-filtered", "segmented-blobs"),
        default="pc",
        help="Overlay mode (default: pc)",
    )
    parser.add_argument("--pc", default="pca_1", help="Principal-component column to overlay (default: pca_1)")
    parser.add_argument("--cluster-column", default="cluster_id", help="Cluster column to overlay in cluster mode")
    parser.add_argument("--candidate-column", default=None, help="Optional puncta-candidate score column; defaults to <channel>_log_puncta_dominance")
    parser.add_argument("--candidate-quantile", type=float, default=0.9, help="Quantile threshold for puncta-candidate overlay (default: 0.9)")
    parser.add_argument("--confidence-threshold", type=float, default=0.55, help="Vote-confidence threshold for confidence-filtered overlay (default: 0.55)")
    parser.add_argument("--min-sigma", type=float, default=0.75, help="Minimum LoG sigma for segmented blob overlay (default: 0.75)")
    parser.add_argument("--max-sigma", type=float, default=3.5, help="Maximum LoG sigma for segmented blob overlay (default: 3.5)")
    parser.add_argument("--num-sigma", type=int, default=6, help="Number of LoG scales for segmented blob overlay (default: 6)")
    parser.add_argument("--blob-threshold", type=float, default=0.03, help="Detection threshold for segmented blob overlay (default: 0.03)")
    parser.add_argument("--blob-overlap", type=float, default=0.5, help="Blob overlap tolerance for segmented blob overlay (default: 0.5)")
    parser.add_argument("--condition", default="positive", help="Condition to choose the example sample from")
    parser.add_argument("--sample-id", default=None, help="Optional explicit sample_id")
    parser.add_argument("--output", default=None, help="Optional output PNG path")
    args = parser.parse_args()

    if args.overlay == "cluster":
        result = create_cluster_component_overlay(
            report_csv=Path(args.report_csv),
            channel=args.channel,
            cluster_column=args.cluster_column,
            condition=args.condition,
            sample_id=args.sample_id,
            output_path=(None if args.output is None else Path(args.output)),
        )
    elif args.overlay == "patch-centers":
        result = create_cluster_patch_center_overlay(
            report_csv=Path(args.report_csv),
            channel=args.channel,
            cluster_column=args.cluster_column,
            condition=args.condition,
            sample_id=args.sample_id,
            output_path=(None if args.output is None else Path(args.output)),
        )
    elif args.overlay == "puncta-candidates":
        result = create_cluster_puncta_candidate_overlay(
            report_csv=Path(args.report_csv),
            channel=args.channel,
            cluster_column=args.cluster_column,
            candidate_column=args.candidate_column,
            candidate_quantile=args.candidate_quantile,
            condition=args.condition,
            sample_id=args.sample_id,
            output_path=(None if args.output is None else Path(args.output)),
        )
    elif args.overlay == "confidence-filtered":
        result = create_confidence_filtered_cluster_overlay(
            report_csv=Path(args.report_csv),
            channel=args.channel,
            cluster_column=args.cluster_column,
            confidence_threshold=args.confidence_threshold,
            condition=args.condition,
            sample_id=args.sample_id,
            output_path=(None if args.output is None else Path(args.output)),
        )
    elif args.overlay == "segmented-blobs":
        result = create_segmented_blob_overlay(
            report_csv=Path(args.report_csv),
            channel=args.channel,
            condition=args.condition,
            sample_id=args.sample_id,
            min_sigma=args.min_sigma,
            max_sigma=args.max_sigma,
            num_sigma=args.num_sigma,
            blob_threshold=args.blob_threshold,
            overlap=args.blob_overlap,
            output_path=(None if args.output is None else Path(args.output)),
        )
    else:
        result = create_principal_component_overlay(
            report_csv=Path(args.report_csv),
            channel=args.channel,
            pc_column=args.pc,
            condition=args.condition,
            sample_id=args.sample_id,
            output_path=(None if args.output is None else Path(args.output)),
        )
    print(
        f"Overlay complete: {result.output_path} | sample={result.sample_id} | "
        f"channel={result.channel} | field={result.overlay_field} | patches={result.patch_count}"
    )


if __name__ == "__main__":
    main()
