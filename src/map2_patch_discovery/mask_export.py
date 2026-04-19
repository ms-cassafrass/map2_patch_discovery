from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_propagation, distance_transform_edt, gaussian_filter, generate_binary_structure, maximum_filter

from .ome import open_ome_image


@dataclass(frozen=True)
class MaskExportConfig:
    input_path: Path
    output_dir: Path
    channel: int = 0
    time: int = 0
    save_debug_png: bool = True
    mip_mask_method: str = "hysteresis"
    mip_mask_high_percentile: float = 99.5
    mip_mask_low_percentile: float = 82.0
    mip_mask_smooth_sigma: float = 1.0
    mip_mask_halo_px: int = 0
    mip_confidence_sigma: float = 2.5
    mip_confidence_floor: float = 0.03
    connectivity_seed_percentile: float = 98.5
    connectivity_support_percentile: float = 82.0
    connectivity_thin_seed_percentile: float = 96.0
    connectivity_thin_seed_max_width_px: float = 2.0


def max_project(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    return np.max(arr, axis=0) if arr.ndim == 3 else arr


def percentile_normalize(arr: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    if not np.any(finite):
        return np.zeros_like(arr, dtype=np.float32)
    vals = arr[finite]
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo:
        out = np.zeros_like(arr, dtype=np.float32)
        out[finite] = 1.0
        return out
    out = np.zeros_like(arr, dtype=np.float32)
    out[finite] = (arr[finite] - lo) / (hi - lo)
    return np.clip(out, 0.0, 1.0)


def otsu_threshold(img: np.ndarray) -> float:
    arr = np.asarray(img, dtype=np.float32)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 0.0
    hist, bin_edges = np.histogram(finite, bins=256)
    hist = hist.astype(np.float64)
    prob = hist / max(1.0, hist.sum())
    omega = np.cumsum(prob)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    idx = int(np.argmax(sigma_b2))
    return float(bin_centers[idx])


def build_mip_mask(
    raw_3d: np.ndarray,
    method: str,
    high_percentile: float,
    low_percentile: float,
    smooth_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    mip = max_project(raw_3d).astype(np.float32, copy=False)
    work = gaussian_filter(mip, sigma=float(max(0.0, smooth_sigma))) if smooth_sigma > 0 else mip
    if method == "threshold":
        thr = float(np.percentile(work, float(np.clip(high_percentile, 0.0, 100.0))))
        seed_mask = work >= thr
        return np.asarray(seed_mask, dtype=bool), np.asarray(seed_mask, dtype=bool)

    high_thr = float(np.percentile(work, float(np.clip(high_percentile, 0.0, 100.0))))
    low_thr = float(np.percentile(work, float(np.clip(low_percentile, 0.0, 100.0))))
    if low_thr > high_thr:
        low_thr = high_thr
    seeds = work >= high_thr
    grow = work >= low_thr
    if np.any(seeds):
        structure = generate_binary_structure(2, 2)
        mask = binary_propagation(seeds, structure=structure, mask=grow)
        return np.asarray(seeds, dtype=bool), np.asarray(mask, dtype=bool)

    seed_mask = work >= otsu_threshold(work)
    return np.asarray(seed_mask, dtype=bool), np.asarray(seed_mask, dtype=bool)


def build_mip_mask_halo(mask_2d: np.ndarray, halo_px: int) -> np.ndarray:
    halo_px = max(0, int(halo_px))
    base = np.asarray(mask_2d, dtype=bool)
    if halo_px == 0:
        return base
    structure = generate_binary_structure(2, 2)
    from scipy.ndimage import binary_dilation
    return np.asarray(binary_dilation(base, structure=structure, iterations=halo_px), dtype=bool)


def build_mip_confidence(mask_2d: np.ndarray | None, sigma: float, floor: float) -> np.ndarray | None:
    if mask_2d is None:
        return None
    mask = np.asarray(mask_2d, dtype=np.float32)
    sigma = float(max(0.0, sigma))
    floor = float(np.clip(floor, 0.0, 1.0))
    conf = gaussian_filter(mask, sigma=sigma) if sigma > 0 else mask
    conf = np.asarray(conf, dtype=np.float32)
    cmin = float(np.min(conf))
    cmax = float(np.max(conf))
    if cmax > cmin:
        conf = (conf - cmin) / (cmax - cmin)
    else:
        conf = np.zeros_like(conf, dtype=np.float32)
    return floor + ((1.0 - floor) * conf)


def broadcast_confidence(confidence_2d: np.ndarray | None, shape: tuple[int, ...]) -> np.ndarray | None:
    if confidence_2d is None:
        return None
    return np.broadcast_to(np.asarray(confidence_2d, dtype=np.float32), shape)


def blend_confidence_into_image(source_image: np.ndarray, confidence_3d: np.ndarray | None, floor: float) -> np.ndarray:
    source = np.asarray(source_image, dtype=np.float32)
    if confidence_3d is None:
        return source
    floor = float(np.clip(floor, 0.0, 1.0))
    confidence = np.asarray(confidence_3d, dtype=np.float32)
    weights = floor + ((1.0 - floor) * confidence)
    return source * weights.astype(np.float32, copy=False)


def build_connectivity_mask(guided_input: np.ndarray, config: MaskExportConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norm = percentile_normalize(guided_input)
    seed_thr = float(np.percentile(norm, float(np.clip(config.connectivity_seed_percentile, 0.0, 100.0))))
    support_thr = float(np.percentile(norm, float(np.clip(config.connectivity_support_percentile, 0.0, 100.0))))
    thin_seed_thr = float(np.percentile(norm, float(np.clip(config.connectivity_thin_seed_percentile, 0.0, 100.0))))

    field = np.asarray(norm, dtype=np.float32)
    support_mask = norm >= support_thr
    width_map_3d = distance_transform_edt(support_mask).astype(np.float32, copy=False)
    strong_seed = norm >= seed_thr
    thin_seed_base = (norm >= thin_seed_thr) & (width_map_3d <= float(config.connectivity_thin_seed_max_width_px))
    thin_peak_mask = norm == maximum_filter(norm, size=(1, 3, 3))
    thin_seed = thin_seed_base & thin_peak_mask
    if not np.any(thin_seed) and np.any(thin_seed_base):
        thin_seed = thin_seed_base

    seed = np.asarray(strong_seed | thin_seed, dtype=bool)
    support_mask = np.asarray(support_mask | seed, dtype=bool)
    structure = generate_binary_structure(field.ndim, 1)
    connectivity_mask = binary_propagation(seed, structure=structure, mask=support_mask)
    return seed, field, np.asarray(connectivity_mask, dtype=bool)


def _save_mask(mask_2d: np.ndarray, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_uint8 = np.asarray(mask_2d, dtype=bool).astype(np.uint8) * 255
    tifffile.imwrite(
        str(output_path),
        mask_uint8,
        ome=True,
        metadata={"axes": "YX"},
        photometric="minisblack",
        compression=None,
    )
    return output_path


def _save_debug_png(mask_2d: np.ndarray, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(mask_2d, cmap="gray", vmin=0, vmax=1)
    ax.set_title("MAP2 Connectivity Mask")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def export_map2_mask(config: MaskExportConfig) -> Path:
    image = open_ome_image(config.input_path)
    raw_working = image.get_zyx(int(config.channel)).astype(np.float32, copy=True)

    _seed_2d, mip_mask = build_mip_mask(
        raw_working,
        method=config.mip_mask_method,
        high_percentile=float(config.mip_mask_high_percentile),
        low_percentile=float(config.mip_mask_low_percentile),
        smooth_sigma=float(config.mip_mask_smooth_sigma),
    )
    guidance_source = build_mip_mask_halo(mip_mask, halo_px=int(config.mip_mask_halo_px))
    confidence_2d = build_mip_confidence(
        guidance_source,
        sigma=float(config.mip_confidence_sigma),
        floor=float(config.mip_confidence_floor),
    )
    confidence_3d = broadcast_confidence(confidence_2d, tuple(raw_working.shape))
    guided_input = blend_confidence_into_image(
        raw_working,
        confidence_3d,
        floor=float(config.mip_confidence_floor),
    )
    _seed_3d, _field, connectivity_mask_3d = build_connectivity_mask(guided_input, config)
    mask_2d = np.asarray(np.max(connectivity_mask_3d, axis=0), dtype=bool)

    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{config.input_path.stem}_analysismask.ome.tif"
    _save_mask(mask_2d, output_path)

    if config.save_debug_png:
        _save_debug_png(mask_2d, output_dir / f"{config.input_path.stem}_analysismask_preview.png")

    return output_path
