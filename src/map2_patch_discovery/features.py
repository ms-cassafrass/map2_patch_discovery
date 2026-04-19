from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.ndimage import binary_erosion, distance_transform_edt, gaussian_filter, gaussian_laplace, label, sobel
from scipy.signal import find_peaks
from scipy.stats import kurtosis, skew

try:  # pragma: no cover - optional dependency
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
except ImportError:  # pragma: no cover
    graycomatrix = None
    graycoprops = None
    local_binary_pattern = None

try:  # pragma: no cover - optional dependency
    from skimage.morphology import convex_hull_image
except ImportError:  # pragma: no cover
    convex_hull_image = None

try:  # pragma: no cover - optional dependency
    import pywt
except ImportError:  # pragma: no cover
    pywt = None


CHANNEL_PAIRS = (
    ("FLAG", "HA"),
    ("FLAG", "SHANK2"),
    ("HA", "SHANK2"),
    ("MAP2", "FLAG"),
    ("MAP2", "HA"),
    ("MAP2", "SHANK2"),
)


def load_patch_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_patch_payload(row: pd.Series | object) -> dict[str, np.ndarray]:
    path = getattr(row, "shard_path", None)
    index = getattr(row, "shard_index", None)
    if path is None or (isinstance(path, float) and pd.isna(path)):
        path = getattr(row, "patch_path")
    payload = load_patch_npz(path)
    if index is None or (isinstance(index, float) and pd.isna(index)):
        return payload
    return slice_patch_payload(payload, int(index))


def slice_patch_payload(payload: dict[str, np.ndarray], patch_index: int) -> dict[str, np.ndarray]:
    sliced: dict[str, np.ndarray] = {}
    for key, value in payload.items():
        if key == "patch_ids":
            sliced[key] = np.asarray(value[patch_index])
        elif isinstance(value, np.ndarray) and value.ndim >= 1:
            sliced[key] = np.asarray(value[patch_index])
        else:
            sliced[key] = value
    return sliced


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator == 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    finite = np.isfinite(a) & np.isfinite(b)
    if np.sum(finite) < 3:
        return 0.0
    a = a[finite]
    b = b[finite]
    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _normalize_zero_one(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def _quantize_image(arr: np.ndarray, levels: int = 16) -> np.ndarray:
    scaled = _normalize_zero_one(arr)
    return np.clip(np.floor(scaled * (levels - 1)), 0, levels - 1).astype(np.uint8)


def _center_and_surround_masks(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = shape
    y0 = h // 4
    y1 = h - y0
    x0 = w // 4
    x1 = w - x0
    center = np.zeros(shape, dtype=bool)
    center[y0:y1, x0:x1] = True
    surround = ~center
    return center, surround


def _bright_threshold(img: np.ndarray) -> float:
    img = np.asarray(img, dtype=np.float32)
    return float(max(np.percentile(img, 97.0), float(np.mean(img) + np.std(img))))


def _component_geometry(binary: np.ndarray, intensity_image: np.ndarray) -> dict[str, float]:
    binary = np.asarray(binary, dtype=bool)
    if not np.any(binary):
        return {
            "component_count": 0.0,
            "largest_area": 0.0,
            "largest_area_fraction": 0.0,
            "eccentricity": 0.0,
            "circularity": 0.0,
            "compactness": 0.0,
            "solidity": 0.0,
            "elongation": 0.0,
            "radial_symmetry": 0.0,
            "microstructure_density": 0.0,
            "dominant_object_fraction": 0.0,
        }

    labels, n = label(binary)
    areas = np.bincount(labels.ravel())[1:]
    largest_label = int(np.argmax(areas) + 1)
    largest = labels == largest_label
    largest_area = float(np.sum(largest))
    total_area = float(np.sum(binary))
    ys, xs = np.nonzero(largest)
    cy = float(np.mean(ys))
    cx = float(np.mean(xs))
    y_centered = ys - cy
    x_centered = xs - cx
    if ys.size >= 2:
        cov = np.cov(np.vstack([y_centered, x_centered]))
        evals = np.sort(np.maximum(np.linalg.eigvalsh(cov), 0.0))
        major = float(np.sqrt(evals[-1])) if evals.size else 0.0
        minor = float(np.sqrt(evals[0])) if evals.size else 0.0
        elongation = _safe_ratio(major, minor)
        eccentricity = float(np.sqrt(max(0.0, 1.0 - _safe_ratio(minor**2, major**2)))) if major > 0 else 0.0
    else:
        elongation = 0.0
        eccentricity = 0.0

    perimeter = float(np.sum(binary ^ binary_erosion(binary)))
    circularity = float((4.0 * np.pi * total_area) / max(perimeter**2, 1.0))
    compactness = circularity

    if convex_hull_image is not None:
        hull_area = float(np.sum(convex_hull_image(largest)))
    else:
        y0, y1 = int(np.min(ys)), int(np.max(ys)) + 1
        x0, x1 = int(np.min(xs)), int(np.max(xs)) + 1
        hull_area = float(max((y1 - y0) * (x1 - x0), 1))
    solidity = _safe_ratio(largest_area, hull_area)

    img = np.asarray(intensity_image, dtype=np.float32)
    coords_y, coords_x = np.indices(img.shape, dtype=np.float32)
    distances = np.sqrt((coords_y - cy) ** 2 + (coords_x - cx) ** 2)
    radial_weight = 1.0 - _normalize_zero_one(distances)
    radial_symmetry = max(0.0, _safe_corr(img * largest.astype(np.float32), radial_weight * largest.astype(np.float32)))

    return {
        "component_count": float(n),
        "largest_area": largest_area,
        "largest_area_fraction": _safe_ratio(largest_area, float(binary.size)),
        "eccentricity": float(eccentricity),
        "circularity": float(circularity),
        "compactness": float(compactness),
        "solidity": float(solidity),
        "elongation": float(elongation),
        "radial_symmetry": float(radial_symmetry),
        "microstructure_density": _safe_ratio(float(n), float(binary.size)),
        "dominant_object_fraction": _safe_ratio(largest_area, total_area),
    }


def _glcm_features(img: np.ndarray) -> dict[str, float]:
    if graycomatrix is None or graycoprops is None:
        return {
            "glcm_contrast": 0.0,
            "glcm_homogeneity": 0.0,
            "glcm_energy": 0.0,
            "glcm_correlation": 0.0,
        }
    quantized = _quantize_image(img, levels=16)
    glcm = graycomatrix(quantized, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=16, symmetric=True, normed=True)
    return {
        "glcm_contrast": float(np.mean(graycoprops(glcm, "contrast"))),
        "glcm_homogeneity": float(np.mean(graycoprops(glcm, "homogeneity"))),
        "glcm_energy": float(np.mean(graycoprops(glcm, "energy"))),
        "glcm_correlation": float(np.mean(graycoprops(glcm, "correlation"))),
    }


def _lbp_features(img: np.ndarray) -> dict[str, float]:
    if local_binary_pattern is None:
        return {"lbp_entropy": 0.0, "lbp_uniform_fraction": 0.0}
    lbp = local_binary_pattern(_normalize_zero_one(img), P=8, R=1.0, method="uniform")
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(12), density=True)
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    entropy = float(-np.sum(hist * np.log2(hist))) if hist.size else 0.0
    uniform_fraction = float(np.mean(lbp < 9))
    return {"lbp_entropy": entropy, "lbp_uniform_fraction": uniform_fraction}


def _wavelet_like_features(img: np.ndarray) -> dict[str, float]:
    img = np.asarray(img, dtype=np.float32)
    if pywt is not None:
        coeffs = pywt.wavedec2(img, wavelet="haar", level=2)
        result: dict[str, float] = {}
        for idx, detail in enumerate(coeffs[1:], start=1):
            cH, cV, cD = detail
            result[f"wavelet_energy_scale_{idx}"] = float(np.mean(cH**2) + np.mean(cV**2) + np.mean(cD**2))
        if "wavelet_energy_scale_1" not in result:
            result["wavelet_energy_scale_1"] = 0.0
        if "wavelet_energy_scale_2" not in result:
            result["wavelet_energy_scale_2"] = 0.0
        return result

    blur1 = gaussian_filter(img, sigma=1.0)
    blur2 = gaussian_filter(img, sigma=2.0)
    blur4 = gaussian_filter(img, sigma=4.0)
    return {
        "wavelet_energy_scale_1": float(np.mean((img - blur1) ** 2)),
        "wavelet_energy_scale_2": float(np.mean((blur1 - blur2) ** 2)),
        "wavelet_energy_scale_3": float(np.mean((blur2 - blur4) ** 2)),
    }


def _z_profile_features(crop: np.ndarray) -> dict[str, float]:
    profile = np.asarray(crop, dtype=np.float32).mean(axis=(1, 2))
    if profile.size == 0:
        return {
            "z_peak": 0.0,
            "z_std": 0.0,
            "z_width_halfmax": 0.0,
            "z_slices_above_halfmax": 0.0,
            "z_center_of_mass": 0.0,
            "z_skewness": 0.0,
            "z_kurtosis": 0.0,
            "z_peak_count": 0.0,
            "z_multi_peak_score": 0.0,
            "z_peak_symmetry": 0.0,
        }
    peak_index = int(np.argmax(profile))
    peak_value = float(profile[peak_index])
    halfmax = 0.5 * peak_value
    above = profile >= halfmax
    peak_count = len(find_peaks(profile, prominence=max(peak_value * 0.1, 1e-6))[0])
    z_axis = np.arange(profile.size, dtype=np.float32)
    center_of_mass = float(np.sum(z_axis * profile) / max(np.sum(profile), 1e-6))
    left = profile[:peak_index]
    right = profile[peak_index + 1 :]
    compare_len = min(len(left), len(right))
    if compare_len > 0:
        left_cmp = left[-compare_len:]
        right_cmp = right[:compare_len]
        symmetry = 1.0 - _safe_ratio(float(np.mean(np.abs(left_cmp - right_cmp))), peak_value if peak_value > 0 else 1.0)
    else:
        symmetry = 0.0
    return {
        "z_peak": float(peak_index),
        "z_std": float(np.std(profile)),
        "z_width_halfmax": float(np.sum(above)),
        "z_slices_above_halfmax": float(np.sum(above)),
        "z_center_of_mass": center_of_mass,
        "z_skewness": float(skew(profile)) if profile.size >= 3 else 0.0,
        "z_kurtosis": float(kurtosis(profile)) if profile.size >= 4 else 0.0,
        "z_peak_count": float(peak_count),
        "z_multi_peak_score": float(max(0, peak_count - 1)),
        "z_peak_symmetry": float(symmetry),
    }


def _center_of_mass_offset(a: np.ndarray, b: np.ndarray) -> float:
    def weighted_com(img: np.ndarray) -> tuple[float, float]:
        img = np.asarray(img, dtype=np.float32)
        total = float(np.sum(img))
        if total <= 0:
            return 0.0, 0.0
        yy, xx = np.indices(img.shape, dtype=np.float32)
        return float(np.sum(yy * img) / total), float(np.sum(xx * img) / total)

    ay, ax = weighted_com(a)
    by, bx = weighted_com(b)
    return float(np.sqrt((ay - by) ** 2 + (ax - bx) ** 2))


def _manders_like(a: np.ndarray, b: np.ndarray, a_thr: float, b_thr: float) -> tuple[float, float]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_mask = a >= a_thr
    b_mask = b >= b_thr
    m1 = _safe_ratio(float(np.sum(a[a_mask & b_mask])), float(np.sum(a[a_mask])))
    m2 = _safe_ratio(float(np.sum(b[a_mask & b_mask])), float(np.sum(b[b_mask])))
    return m1, m2


def _channel_feature_block(channel: str, crop: np.ndarray, mask: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
    crop = np.asarray(crop, dtype=np.float32)
    max_proj = np.max(crop, axis=0)
    mean_proj = np.mean(crop, axis=0)
    bright_thr = _bright_threshold(max_proj)
    bright_mask = max_proj >= bright_thr
    inside_vals = mean_proj[mask] if np.any(mask) else mean_proj.ravel()
    outside_vals = mean_proj[~mask] if np.any(~mask) else mean_proj.ravel()
    center_mask, surround_mask = _center_and_surround_masks(mean_proj.shape)
    center_vals = mean_proj[center_mask]
    surround_vals = mean_proj[surround_mask]
    grad_y = sobel(mean_proj, axis=0)
    grad_x = sobel(mean_proj, axis=1)
    grad_mag = np.sqrt(grad_y**2 + grad_x**2)
    high_freq = mean_proj - gaussian_filter(mean_proj, sigma=1.0)
    hist, _ = np.histogram(_normalize_zero_one(mean_proj), bins=32, range=(0.0, 1.0), density=True)
    hist = hist.astype(np.float64)
    hist = hist[hist > 0]
    intensity_entropy = float(-np.sum(hist * np.log2(hist))) if hist.size else 0.0
    component_stats = _component_geometry(bright_mask, max_proj)
    glcm = _glcm_features(mean_proj)
    lbp = _lbp_features(mean_proj)
    wavelet = _wavelet_like_features(mean_proj)
    z_features = _z_profile_features(crop)
    inside_bright_fraction = float(np.mean(bright_mask[mask])) if np.any(mask) else 0.0
    outside_bright_fraction = float(np.mean(bright_mask[~mask])) if np.any(~mask) else 0.0

    feature_block = {
        f"{channel.lower()}_proj_mean": float(np.mean(mean_proj)),
        f"{channel.lower()}_proj_max": float(np.max(max_proj)),
        f"{channel.lower()}_proj_std": float(np.std(mean_proj)),
        f"{channel.lower()}_inside_mean": float(np.mean(inside_vals)),
        f"{channel.lower()}_outside_mean": float(np.mean(outside_vals)),
        f"{channel.lower()}_inside_outside_ratio": _safe_ratio(float(np.mean(inside_vals)), float(np.mean(outside_vals)) if np.size(outside_vals) else 0.0),
        f"{channel.lower()}_local_contrast": _safe_ratio(float(np.std(mean_proj)), float(np.mean(mean_proj)) + 1e-6),
        f"{channel.lower()}_dynamic_range": float(np.max(mean_proj) - np.min(mean_proj)),
        f"{channel.lower()}_bright_pixel_fraction": float(np.mean(bright_mask)),
        f"{channel.lower()}_top_percentile_intensity": float(np.mean(max_proj[max_proj >= np.percentile(max_proj, 99.0)])),
        f"{channel.lower()}_intensity_entropy": intensity_entropy,
        f"{channel.lower()}_center_surround_diff": float(np.mean(center_vals) - np.mean(surround_vals)),
        f"{channel.lower()}_center_surround_ratio": _safe_ratio(float(np.mean(center_vals)), float(np.mean(surround_vals))),
        f"{channel.lower()}_center_mean": float(np.mean(center_vals)),
        f"{channel.lower()}_surround_mean": float(np.mean(surround_vals)),
        f"{channel.lower()}_estimated_punctum_area": float(np.sum(bright_mask)),
        f"{channel.lower()}_log_response": float(np.max(np.abs(gaussian_laplace(mean_proj, sigma=1.0)))),
        f"{channel.lower()}_dog_response": float(np.max(np.abs(gaussian_filter(mean_proj, 1.0) - gaussian_filter(mean_proj, 2.0)))),
        f"{channel.lower()}_gradient_mean": float(np.mean(grad_mag)),
        f"{channel.lower()}_gradient_std": float(np.std(grad_mag)),
        f"{channel.lower()}_high_frequency_power": float(np.mean(high_freq**2)),
        f"{channel.lower()}_inside_bright_fraction": inside_bright_fraction,
        f"{channel.lower()}_outside_bright_fraction": outside_bright_fraction,
        f"{channel.lower()}_inside_bright_outside_bright_ratio": _safe_ratio(inside_bright_fraction, outside_bright_fraction),
        f"{channel.lower()}_local_background_estimate": float(np.median(surround_vals)),
        f"{channel.lower()}_local_snr": _safe_ratio(float(np.mean(center_vals) - np.mean(surround_vals)), float(np.std(surround_vals)) + 1e-6),
        f"{channel.lower()}_neighborhood_variance": float(np.var(mean_proj)),
    }
    for key, value in component_stats.items():
        feature_block[f"{channel.lower()}_{key}"] = float(value)
    for key, value in glcm.items():
        feature_block[f"{channel.lower()}_{key}"] = float(value)
    for key, value in lbp.items():
        feature_block[f"{channel.lower()}_{key}"] = float(value)
    for key, value in wavelet.items():
        feature_block[f"{channel.lower()}_{key}"] = float(value)
    for key, value in z_features.items():
        feature_block[f"{channel.lower()}_{key}"] = float(value)

    stats_block = {
        "mean_proj": mean_proj,
        "max_proj": max_proj,
        "bright_mask": bright_mask,
        "bright_threshold": bright_thr,
        "component_compactness": float(component_stats["compactness"]),
        "log_response": float(feature_block[f"{channel.lower()}_log_response"]),
        "mean_intensity": float(np.mean(mean_proj)),
        "sum_intensity": float(np.sum(crop)),
    }
    return feature_block, stats_block


def _map2_spatial_features(map2_mean_proj: np.ndarray, mask: np.ndarray, center_y: int, center_x: int, distance_to_mask_px: float) -> dict[str, float]:
    center_intensity = float(map2_mean_proj[center_y, center_x])
    thickness = 0.0
    if mask[center_y, center_x]:
        thickness = float(distance_transform_edt(mask)[center_y, center_x] * 2.0)
    return {
        "map2_mask_fraction": float(np.mean(mask)),
        "distance_to_mask_boundary_px": float(distance_to_mask_px),
        "center_of_patch_map2_intensity": center_intensity,
        "map2_local_thickness_proxy": thickness,
    }


def _cross_channel_features(channel_stats: dict[str, dict[str, float | np.ndarray]], mask: np.ndarray) -> dict[str, float]:
    features: dict[str, float] = {}
    mask = np.asarray(mask, dtype=bool)
    for a, b in CHANNEL_PAIRS:
        if a not in channel_stats or b not in channel_stats:
            continue
        a_mean = np.asarray(channel_stats[a]["mean_proj"], dtype=np.float32)
        b_mean = np.asarray(channel_stats[b]["mean_proj"], dtype=np.float32)
        a_max = np.asarray(channel_stats[a]["max_proj"], dtype=np.float32)
        b_max = np.asarray(channel_stats[b]["max_proj"], dtype=np.float32)
        a_bright = np.asarray(channel_stats[a]["bright_mask"], dtype=bool)
        b_bright = np.asarray(channel_stats[b]["bright_mask"], dtype=bool)

        pair = f"{a.lower()}_{b.lower()}"
        features[f"{pair}_pixel_corr"] = _safe_corr(a_mean, b_mean)
        features[f"{pair}_pixel_corr_in_mask"] = _safe_corr(a_mean[mask], b_mean[mask]) if np.any(mask) else 0.0
        intersect = float(np.sum(a_bright & b_bright))
        union = float(np.sum(a_bright | b_bright))
        features[f"{pair}_bright_overlap_jaccard"] = _safe_ratio(intersect, union)
        features[f"{pair}_bright_overlap_coef"] = _safe_ratio(intersect, min(float(np.sum(a_bright)), float(np.sum(b_bright))))
        m1, m2 = _manders_like(
            a_max,
            b_max,
            float(channel_stats[a]["bright_threshold"]),
            float(channel_stats[b]["bright_threshold"]),
        )
        features[f"{pair}_manders_m1"] = m1
        features[f"{pair}_manders_m2"] = m2
        features[f"{pair}_com_offset"] = _center_of_mass_offset(a_max, b_max)
        features[f"{pair}_compactness_ratio"] = _safe_ratio(
            float(channel_stats[a]["component_compactness"]),
            float(channel_stats[b]["component_compactness"]),
        )
        features[f"{pair}_spotness_ratio"] = _safe_ratio(
            float(channel_stats[a]["log_response"]),
            float(channel_stats[b]["log_response"]),
        )
        features[f"{pair}_mean_ratio"] = _safe_ratio(
            float(channel_stats[a]["mean_intensity"]),
            float(channel_stats[b]["mean_intensity"]),
        )
    return features


def extract_engineered_features(manifest: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    shard_cache: dict[str, dict[str, np.ndarray]] = {}
    total = len(manifest)
    start = perf_counter()
    print(f"[features] start | patches={total} | channels={','.join(channels)}")
    for idx, row in enumerate(manifest.itertuples(index=False), start=1):
        shard_path = getattr(row, "shard_path", None)
        shard_index = getattr(row, "shard_index", None)
        if shard_path is not None and not (isinstance(shard_path, float) and pd.isna(shard_path)):
            shard_key = str(shard_path)
            if shard_key not in shard_cache:
                shard_cache[shard_key] = load_patch_npz(shard_key)
            payload = slice_patch_payload(shard_cache[shard_key], int(shard_index))
        else:
            payload = load_patch_payload(row)

        mask = np.asarray(payload["map2_mask"], dtype=bool)
        flat_record: dict[str, object] = {"patch_id": row.patch_id}
        channel_stats: dict[str, dict[str, float | np.ndarray]] = {}

        for channel in channels:
            key = f"channel_{channel}"
            if key not in payload:
                source_path = getattr(row, "shard_path", None) or row.patch_path
                raise ValueError(f"Patch source {source_path} missing expected key {key}")
            crop = np.asarray(payload[key], dtype=np.float32)
            feature_block, stats_block = _channel_feature_block(channel=channel, crop=crop, mask=mask)
            flat_record.update(feature_block)
            channel_stats[channel] = stats_block

        map2_mean_proj = np.asarray(channel_stats["MAP2"]["mean_proj"], dtype=np.float32)
        center_y = map2_mean_proj.shape[0] // 2
        center_x = map2_mean_proj.shape[1] // 2
        flat_record.update(
            _map2_spatial_features(
                map2_mean_proj=map2_mean_proj,
                mask=mask,
                center_y=center_y,
                center_x=center_x,
                distance_to_mask_px=float(getattr(row, "distance_to_mask_px", 0.0)),
            )
        )
        flat_record.update(_cross_channel_features(channel_stats=channel_stats, mask=mask))

        channel_means = {channel: float(channel_stats[channel]["mean_intensity"]) for channel in channel_stats}
        channel_sums = {channel: float(channel_stats[channel]["sum_intensity"]) for channel in channel_stats}
        flat_record["flag_to_ha_mean_ratio"] = _safe_ratio(channel_means.get("FLAG", 0.0), channel_means.get("HA", 0.0))
        flat_record["ha_to_flag_mean_ratio"] = _safe_ratio(channel_means.get("HA", 0.0), channel_means.get("FLAG", 0.0))
        flat_record["flag_ha_mean_sum"] = float(channel_means.get("FLAG", 0.0) + channel_means.get("HA", 0.0))
        flat_record["flag_ha_mean_product"] = float(channel_means.get("FLAG", 0.0) * channel_means.get("HA", 0.0))
        flat_record["flag_ha_mean_absdiff"] = float(abs(channel_means.get("FLAG", 0.0) - channel_means.get("HA", 0.0)))
        flat_record["shank2_to_map2_mean_ratio"] = _safe_ratio(channel_means.get("SHANK2", 0.0), channel_means.get("MAP2", 0.0))
        flat_record["flag_plus_ha_to_shank2_sum_ratio"] = _safe_ratio(
            channel_sums.get("FLAG", 0.0) + channel_sums.get("HA", 0.0),
            channel_sums.get("SHANK2", 0.0),
        )
        records.append(flat_record)
        if idx == 1 or idx % 1000 == 0 or idx == total:
            elapsed = perf_counter() - start
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = (total - idx) / rate if rate > 0 else 0.0
            print(
                f"[features] {idx}/{total} | elapsed={elapsed:.1f}s | "
                f"rate={rate:.1f} patches/s | eta={eta/60:.1f}m"
            )

    elapsed = perf_counter() - start
    print(f"[features] done | patches={total} | elapsed={elapsed:.1f}s")
    return pd.DataFrame.from_records(records)
