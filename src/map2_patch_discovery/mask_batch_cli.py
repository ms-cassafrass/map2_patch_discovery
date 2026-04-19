from __future__ import annotations

import argparse
from pathlib import Path

from .mask_export import MaskExportConfig, export_map2_mask


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate MAP2 analysis masks sequentially for OME-TIFF images that are missing masks."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input OME-TIFF images")
    parser.add_argument("--mask-root", required=True, help="Root directory where per-image mask folders live")
    parser.add_argument("--channel", type=int, default=0, help="Channel index for MAP2")
    parser.add_argument("--time", type=int, default=0, help="Time index")
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
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of missing images to process")
    parser.add_argument("--dry-run", action="store_true", help="List missing masks without generating them")
    return parser


def _missing_images(input_dir: Path, mask_root: Path) -> list[Path]:
    images = sorted(input_dir.glob("*.ome.tif"))
    missing: list[Path] = []
    for image in images:
        stem = image.name[:-len(".ome.tif")]
        mask_path = mask_root / stem / f"{stem}.ome_analysismask.ome.tif"
        if not mask_path.exists():
            missing.append(image)
    return missing


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    mask_root = Path(args.mask_root).resolve()
    missing = _missing_images(input_dir=input_dir, mask_root=mask_root)
    if args.limit is not None:
        missing = missing[: max(0, int(args.limit))]

    print(f"[mask batch] input_dir={input_dir}")
    print(f"[mask batch] mask_root={mask_root}")
    print(f"[mask batch] missing_masks={len(missing)}")
    if args.dry_run:
        for image in missing:
            print(image)
        return

    for index, image in enumerate(missing, start=1):
        stem = image.name[:-len(".ome.tif")]
        output_dir = mask_root / stem
        print(f"[mask {index}/{len(missing)}] {image.name}")
        config = MaskExportConfig(
            input_path=image,
            output_dir=output_dir,
            channel=int(args.channel),
            time=int(args.time),
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
        print(f"[saved] {output_path}")


if __name__ == "__main__":
    main()
