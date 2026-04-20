from __future__ import annotations

import argparse
from pathlib import Path

from .mask_export import MaskExportConfig, export_map2_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export a usable MAP2 analysis mask using the v5.2 connectivity-mask logic.")
    parser.add_argument("input", help="Input OME-TIFF image path")
    parser.add_argument("-o", "--output-dir", required=True, help="Directory to write the analysis mask")
    parser.add_argument("--channel", type=int, default=0, help="Channel index for MAP2")
    parser.add_argument("--save-debug-png", action=argparse.BooleanOptionalAction, default=True, help="Save a preview PNG")
    parser.add_argument("--mip-mask-method", choices=("threshold", "hysteresis"), default="hysteresis")
    parser.add_argument("--mip-mask-high-percentile", type=float, default=99.5)
    parser.add_argument("--mip-mask-low-percentile", type=float, default=82.0)
    parser.add_argument("--mip-mask-smooth-sigma", type=float, default=1.0)
    parser.add_argument("--mip-mask-halo-px", type=int, default=0)
    parser.add_argument("--mip-confidence-sigma", type=float, default=2.5)
    parser.add_argument("--mip-confidence-floor", type=float, default=0.03)
    parser.add_argument("--connectivity-seed-percentile", type=float, default=98.5)
    parser.add_argument("--connectivity-support-percentile", type=float, default=82.0)
    parser.add_argument("--connectivity-thin-seed-percentile", type=float, default=96.0)
    parser.add_argument("--connectivity-thin-seed-max-width-px", type=float, default=2.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = MaskExportConfig(
        input_path=Path(args.input).resolve(),
        output_dir=Path(args.output_dir).resolve(),
        channel=int(args.channel),
        save_debug_png=bool(args.save_debug_png),
        mip_mask_method=str(args.mip_mask_method),
        mip_mask_high_percentile=float(args.mip_mask_high_percentile),
        mip_mask_low_percentile=float(args.mip_mask_low_percentile),
        mip_mask_smooth_sigma=float(args.mip_mask_smooth_sigma),
        mip_mask_halo_px=int(args.mip_mask_halo_px),
        mip_confidence_sigma=float(args.mip_confidence_sigma),
        mip_confidence_floor=float(args.mip_confidence_floor),
        connectivity_seed_percentile=float(args.connectivity_seed_percentile),
        connectivity_support_percentile=float(args.connectivity_support_percentile),
        connectivity_thin_seed_percentile=float(args.connectivity_thin_seed_percentile),
        connectivity_thin_seed_max_width_px=float(args.connectivity_thin_seed_max_width_px),
    )
    output_path = export_map2_mask(config)
    print(f"Saved analysis mask: {output_path}")


if __name__ == "__main__":
    main()
