from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label, maximum_filter
from sklearn.decomposition import PCA

from .features import CHANNEL_PAIRS, load_patch_npz, load_patch_payload, slice_patch_payload
from .latent_report import (
    _apply_map2_feature_policy,
    _audit_and_filter_feature_columns,
    _categorize_feature_family,
    _load_or_compute_engineered_features,
)
from .preprocessing import scale_feature_matrix
from .report_config import LatentReportConfig


OBJECT_DISTANCE_THRESHOLDS_PX = (3.0, 5.0)
PROGRESS_EVERY = 1000


def _bright_threshold(img: np.ndarray) -> float:
    img = np.asarray(img, dtype=np.float32)
    return float(max(np.percentile(img, 97.0), float(np.mean(img) + np.std(img))))


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator == 0.0 or not np.isfinite(denominator):
        return 0.0
    return float(numerator / denominator)


def _component_areas_and_centroids(binary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels, n = label(np.asarray(binary, dtype=bool))
    if n == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    areas: list[float] = []
    centroids: list[tuple[float, float]] = []
    for component_id in range(1, n + 1):
        ys, xs = np.nonzero(labels == component_id)
        if ys.size == 0:
            continue
        areas.append(float(ys.size))
        centroids.append((float(np.mean(ys)), float(np.mean(xs))))
    return np.asarray(areas, dtype=np.float32), np.asarray(centroids, dtype=np.float32)


def _pairwise_distances(points_a: np.ndarray, points_b: np.ndarray) -> np.ndarray:
    if points_a.size == 0 or points_b.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    diffs = points_a[:, None, :] - points_b[None, :, :]
    return np.sqrt(np.sum(diffs**2, axis=2, dtype=np.float32)).astype(np.float32, copy=False)


def _self_nearest_distances(points: np.ndarray) -> np.ndarray:
    if len(points) <= 1:
        return np.zeros((0,), dtype=np.float32)
    dists = _pairwise_distances(points, points)
    np.fill_diagonal(dists, np.inf)
    return np.min(dists, axis=1)


def _map2_centerline(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return np.zeros_like(mask, dtype=bool)
    dist = distance_transform_edt(mask).astype(np.float32, copy=False)
    local_max = dist == maximum_filter(dist, size=3)
    centerline = mask & local_max & (dist > 0)
    if np.any(centerline):
        return centerline
    max_dist = float(np.max(dist))
    if max_dist <= 0.0:
        return mask
    return mask & (dist >= (0.9 * max_dist))


def _mask_orientation(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ys, xs = np.nonzero(np.asarray(mask, dtype=bool))
    if ys.size < 2:
        centroid = np.asarray([mask.shape[0] / 2.0, mask.shape[1] / 2.0], dtype=np.float32)
        major = np.asarray([1.0, 0.0], dtype=np.float32)
        minor = np.asarray([0.0, 1.0], dtype=np.float32)
        evals = np.asarray([0.0, 0.0], dtype=np.float32)
        return centroid, major, minor, evals

    coords = np.column_stack([ys, xs]).astype(np.float32)
    centroid = np.mean(coords, axis=0)
    centered = coords - centroid
    cov = np.cov(centered.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.asarray(evals[order], dtype=np.float32)
    evecs = np.asarray(evecs[:, order], dtype=np.float32)
    major = evecs[:, 0]
    minor = evecs[:, 1] if evecs.shape[1] > 1 else np.asarray([major[1], -major[0]], dtype=np.float32)
    return centroid.astype(np.float32), major.astype(np.float32), minor.astype(np.float32), evals


def _posthoc_map2_features(mask: np.ndarray) -> dict[str, float]:
    mask = np.asarray(mask, dtype=bool)
    centerline = _map2_centerline(mask)
    centroid, major, minor, evals = _mask_orientation(mask)
    angle = float(np.arctan2(float(major[0]), float(major[1])))
    anisotropy = 0.0
    if evals.size >= 2 and float(evals[0]) > 0.0:
        anisotropy = 1.0 - _safe_ratio(float(evals[1]), float(evals[0]))

    return {
        "posthoc_map2_centerline_fraction": float(np.mean(centerline)),
        "posthoc_map2_centerline_length_px": float(np.sum(centerline)),
        "posthoc_map2_orientation_angle_rad": angle,
        "posthoc_map2_orientation_anisotropy": float(anisotropy),
        "posthoc_map2_centroid_y": float(centroid[0]),
        "posthoc_map2_centroid_x": float(centroid[1]),
        "posthoc_map2_major_axis_y": float(major[0]),
        "posthoc_map2_major_axis_x": float(major[1]),
        "posthoc_map2_minor_axis_y": float(minor[0]),
        "posthoc_map2_minor_axis_x": float(minor[1]),
    }


def _channel_object_features(
    channel: str,
    crop: np.ndarray,
    mask: np.ndarray,
    centerline: np.ndarray,
    centroid: np.ndarray,
    major_axis: np.ndarray,
    minor_axis: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    max_proj = np.max(np.asarray(crop, dtype=np.float32), axis=0)
    bright = max_proj >= _bright_threshold(max_proj)
    areas, object_centroids = _component_areas_and_centroids(bright)
    nn = _self_nearest_distances(object_centroids)

    centerline_dist = distance_transform_edt(~np.asarray(centerline, dtype=bool)).astype(np.float32, copy=False)
    mask_dist = distance_transform_edt(np.asarray(mask, dtype=bool)).astype(np.float32, copy=False)

    if len(object_centroids) > 0:
        obj_y = np.clip(np.rint(object_centroids[:, 0]).astype(np.intp), 0, centerline_dist.shape[0] - 1)
        obj_x = np.clip(np.rint(object_centroids[:, 1]).astype(np.intp), 0, centerline_dist.shape[1] - 1)
        dist_to_centerline = centerline_dist[obj_y, obj_x].astype(np.float32, copy=False)
        local_half_width = mask_dist[obj_y, obj_x].astype(np.float32, copy=False)
        dist_to_centerline_norm = np.divide(
            dist_to_centerline,
            np.maximum(local_half_width, np.float32(1e-6)),
            out=np.zeros_like(dist_to_centerline, dtype=np.float32),
            where=np.isfinite(local_half_width),
        )
        centered = object_centroids - centroid[None, :]
        longitudinal = centered @ major_axis
        transverse = centered @ minor_axis
    else:
        dist_to_centerline = np.zeros((0,), dtype=np.float32)
        dist_to_centerline_norm = np.zeros((0,), dtype=np.float32)
        longitudinal = np.zeros((0,), dtype=np.float32)
        transverse = np.zeros((0,), dtype=np.float32)

    prefix = channel.lower()
    features = {
        f"posthoc_{prefix}_object_count": float(len(object_centroids)),
        f"posthoc_{prefix}_object_density": _safe_ratio(float(len(object_centroids)), float(max_proj.size)),
        f"posthoc_{prefix}_object_area_mean": float(np.mean(areas)) if areas.size else 0.0,
        f"posthoc_{prefix}_object_area_median": float(np.median(areas)) if areas.size else 0.0,
        f"posthoc_{prefix}_object_nn_distance_mean_px": float(np.mean(nn)) if nn.size else 0.0,
        f"posthoc_{prefix}_object_nn_distance_median_px": float(np.median(nn)) if nn.size else 0.0,
        f"posthoc_{prefix}_object_centerline_distance_mean_px": float(np.mean(dist_to_centerline)) if dist_to_centerline.size else 0.0,
        f"posthoc_{prefix}_object_centerline_distance_median_px": float(np.median(dist_to_centerline)) if dist_to_centerline.size else 0.0,
        f"posthoc_{prefix}_object_centerline_distance_mean_norm": float(np.mean(dist_to_centerline_norm)) if dist_to_centerline_norm.size else 0.0,
        f"posthoc_{prefix}_object_longitudinal_std_px": float(np.std(longitudinal)) if longitudinal.size else 0.0,
        f"posthoc_{prefix}_object_transverse_std_px": float(np.std(transverse)) if transverse.size else 0.0,
        f"posthoc_{prefix}_object_transverse_abs_mean_px": float(np.mean(np.abs(transverse))) if transverse.size else 0.0,
    }
    return features, {
        "centroids": object_centroids,
        "count": np.asarray([len(object_centroids)], dtype=np.float32),
    }


def _pair_object_features(channel_a: str, channel_b: str, stats: dict[str, dict[str, np.ndarray]]) -> dict[str, float]:
    if channel_a not in stats or channel_b not in stats:
        return {}
    points_a = np.asarray(stats[channel_a]["centroids"], dtype=np.float32)
    points_b = np.asarray(stats[channel_b]["centroids"], dtype=np.float32)
    pair_name = f"{channel_a.lower()}_{channel_b.lower()}"
    if len(points_a) == 0 or len(points_b) == 0:
        base = {
            f"posthoc_{pair_name}_object_nn_mean_px": 0.0,
            f"posthoc_{pair_name}_object_nn_median_px": 0.0,
            f"posthoc_{pair_name}_object_count_ratio": _safe_ratio(float(len(points_a)), float(len(points_b))),
        }
        for threshold in OBJECT_DISTANCE_THRESHOLDS_PX:
            label_suffix = str(int(threshold))
            base[f"posthoc_{pair_name}_object_within_{label_suffix}px_fraction"] = 0.0
        return base

    dists = _pairwise_distances(points_a, points_b)
    nearest_ab = np.min(dists, axis=1)
    nearest_ba = np.min(dists, axis=0)
    nearest_all = np.concatenate([nearest_ab, nearest_ba]).astype(np.float32, copy=False)
    features = {
        f"posthoc_{pair_name}_object_nn_mean_px": float(np.mean(nearest_all)),
        f"posthoc_{pair_name}_object_nn_median_px": float(np.median(nearest_all)),
        f"posthoc_{pair_name}_object_count_ratio": _safe_ratio(float(len(points_a)), float(len(points_b))),
    }
    for threshold in OBJECT_DISTANCE_THRESHOLDS_PX:
        label_suffix = str(int(threshold))
        frac_ab = float(np.mean(nearest_ab <= threshold)) if nearest_ab.size else 0.0
        frac_ba = float(np.mean(nearest_ba <= threshold)) if nearest_ba.size else 0.0
        features[f"posthoc_{pair_name}_object_within_{label_suffix}px_fraction"] = 0.5 * (frac_ab + frac_ba)
    return features


def extract_posthoc_features(manifest: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    shard_cache: dict[str, dict[str, np.ndarray]] = {}
    total = len(manifest)
    start = perf_counter()
    print(f"[posthoc] start | patches={total} | channels={','.join(channels)}")

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
        centerline = _map2_centerline(mask)
        centroid, major_axis, minor_axis, _ = _mask_orientation(mask)

        record: dict[str, object] = {"patch_id": row.patch_id}
        record.update(_posthoc_map2_features(mask))
        object_stats: dict[str, dict[str, np.ndarray]] = {}
        for channel in channels:
            key = f"channel_{channel}"
            if key not in payload:
                continue
            features, stats = _channel_object_features(
                channel=channel,
                crop=np.asarray(payload[key], dtype=np.float32),
                mask=mask,
                centerline=centerline,
                centroid=centroid,
                major_axis=major_axis,
                minor_axis=minor_axis,
            )
            record.update(features)
            object_stats[channel] = stats

        for channel_a, channel_b in CHANNEL_PAIRS:
            record.update(_pair_object_features(channel_a, channel_b, object_stats))
        records.append(record)

        if idx == 1 or idx % PROGRESS_EVERY == 0 or idx == total:
            elapsed = perf_counter() - start
            rate = idx / elapsed if elapsed > 0 else 0.0
            eta = (total - idx) / rate if rate > 0 else 0.0
            print(f"[posthoc] {idx}/{total} | elapsed={elapsed:.1f}s | rate={rate:.1f} patches/s | eta={eta/60:.1f}m")

    print(f"[posthoc] done | patches={total} | elapsed={perf_counter()-start:.1f}s")
    return pd.DataFrame.from_records(records)


def _prepare_selected_features(config: LatentReportConfig, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, object], dict[str, int | bool | str]]:
    manifest = pd.read_csv(config.manifest_path)
    feature_df, feature_cache_summary = _load_or_compute_engineered_features(
        manifest=manifest,
        manifest_path=config.manifest_path,
        channels=config.features.channels,
    )
    report_df = manifest.merge(feature_df, on="patch_id", how="inner")
    feature_columns = [column for column in feature_df.columns if column != "patch_id"]
    if config.features.feature_variance_csv is not None:
        variance_df = pd.read_csv(config.features.feature_variance_csv)
        if "feature" not in variance_df.columns or "feature_variance_cluster" not in variance_df.columns:
            raise ValueError("Feature variance CSV must contain 'feature' and 'feature_variance_cluster' columns")
        selected_features = variance_df.loc[
            variance_df["feature_variance_cluster"] == config.features.feature_variance_cluster,
            "feature",
        ].astype(str).tolist()
        feature_columns = [column for column in feature_columns if column in set(selected_features)]
        if not feature_columns:
            raise ValueError(
                f"No engineered features matched feature_variance_cluster={config.features.feature_variance_cluster} "
                f"from {config.features.feature_variance_csv}"
            )

    feature_columns, _, _ = _apply_map2_feature_policy(
        feature_columns=feature_columns,
        policy=config.features.map2_feature_policy,
        output_dir=output_dir,
    )
    if not feature_columns:
        raise ValueError("All candidate features were excluded by the MAP2 feature policy.")

    feature_columns, audit_summary, _ = _audit_and_filter_feature_columns(
        report_df=report_df,
        feature_columns=feature_columns,
        output_dir=output_dir,
    )
    if not feature_columns:
        raise ValueError("All candidate features were dropped by the variance audit.")
    return manifest, report_df, feature_columns, feature_cache_summary, audit_summary


def _feature_family(feature: str) -> str:
    feature_lower = str(feature).lower()
    if feature_lower.startswith("posthoc_map2_"):
        return "Posthoc Dendrite Geometry"
    if feature_lower.startswith("posthoc_") and "_object_" in feature_lower:
        return "Posthoc Object Geometry"
    return _categorize_feature_family(str(feature))


def _save_variance_weighted_contribution_report(
    *,
    output_dir: Path,
    pca: PCA,
    feature_columns: list[str],
) -> pd.DataFrame:
    explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    loadings = np.asarray(pca.components_, dtype=np.float64).T
    pc_columns = [f"PC{i+1}" for i in range(loadings.shape[1])]

    weighted_abs = np.sum(np.abs(loadings) * explained[None, :], axis=1)
    weighted_sq = np.sum((loadings**2) * explained[None, :], axis=1)
    dominant_pc = np.argmax(np.abs(loadings), axis=1) + 1 if loadings.size else np.zeros((len(feature_columns),), dtype=np.int32)

    df = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_family": [_feature_family(feature) for feature in feature_columns],
            "variance_weighted_abs_loading": weighted_abs,
            "variance_weighted_squared_loading": weighted_sq,
            "dominant_pc": dominant_pc,
        }
    )
    for idx, column in enumerate(pc_columns):
        df[column] = loadings[:, idx]
        df[f"{column}_abs"] = np.abs(loadings[:, idx])
        df[f"{column}_weighted_abs"] = np.abs(loadings[:, idx]) * explained[idx]
    df = df.sort_values("variance_weighted_squared_loading", ascending=False).reset_index(drop=True)
    df.to_csv(output_dir / "variance_weighted_pca_feature_contributions.csv", index=False)

    summary = (
        df.groupby("feature_family", dropna=False)
        .agg(
            feature_count=("feature", "size"),
            family_weighted_abs_sum=("variance_weighted_abs_loading", "sum"),
            family_weighted_sq_sum=("variance_weighted_squared_loading", "sum"),
        )
        .reset_index()
        .sort_values("family_weighted_sq_sum", ascending=False)
    )
    summary.to_csv(output_dir / "variance_weighted_pca_feature_family_summary.csv", index=False)

    top = df.head(min(25, len(df)))
    fig_h = max(6, 0.28 * len(top))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(top["feature"][::-1], top["variance_weighted_squared_loading"][::-1], color="#4C78A8")
    ax.set_xlabel("Variance-Weighted Squared Loading")
    ax.set_title("Top PCA Variance Contributors")
    fig.tight_layout()
    fig.savefig(output_dir / "variance_weighted_pca_feature_contributions.png", dpi=150)
    plt.close(fig)

    return df


def run_posthoc_feature_analysis(
    config: LatentReportConfig,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    overall_start = perf_counter()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (config.output_dir.resolve() / "posthoc_feature_analysis")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[posthoc-analysis] output_dir={output_dir}")

    stage_start = perf_counter()
    manifest, report_df, feature_columns, feature_cache_summary, audit_summary = _prepare_selected_features(config, output_dir)
    print(
        f"[posthoc-analysis] selected existing features | rows={len(report_df)} | kept={len(feature_columns)} "
        f"| elapsed={perf_counter()-stage_start:.1f}s"
    )

    stage_start = perf_counter()
    posthoc_df = extract_posthoc_features(manifest=manifest, channels=config.features.channels)
    report_df = report_df.merge(posthoc_df, on="patch_id", how="inner")
    posthoc_columns = [column for column in posthoc_df.columns if column != "patch_id"]
    combined_columns = feature_columns + posthoc_columns
    combined_columns, _, _ = _apply_map2_feature_policy(
        feature_columns=combined_columns,
        policy=config.features.map2_feature_policy,
        output_dir=output_dir,
    )
    combined_columns, combined_audit_summary, combined_audit_df = _audit_and_filter_feature_columns(
        report_df=report_df,
        feature_columns=combined_columns,
        output_dir=output_dir,
    )
    if not combined_columns:
        raise ValueError("All combined features were dropped by the variance audit.")
    print(
        f"[posthoc-analysis] posthoc features ready | added={len(posthoc_columns)} | combined_kept={len(combined_columns)} "
        f"| elapsed={perf_counter()-stage_start:.1f}s"
    )

    stage_start = perf_counter()
    feature_matrix = report_df[combined_columns].to_numpy(dtype=np.float64)
    scaled = scale_feature_matrix(feature_matrix, config.preprocessing.scaler)
    max_components = min(scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=max_components, random_state=config.clustering.random_seed)
    pca.fit(scaled)
    print(
        f"[posthoc-analysis] PCA fit complete | scaler={config.preprocessing.scaler} "
        f"| components={max_components} | elapsed={perf_counter()-stage_start:.1f}s"
    )

    posthoc_df.to_csv(output_dir / "posthoc_features.csv", index=False)
    pd.DataFrame({"feature": feature_columns}).to_csv(output_dir / "existing_selected_feature_columns.csv", index=False)
    pd.DataFrame({"feature": posthoc_columns}).to_csv(output_dir / "posthoc_feature_columns.csv", index=False)
    pd.DataFrame({"feature": combined_columns}).to_csv(output_dir / "combined_selected_feature_columns.csv", index=False)
    combined_audit_df.to_csv(output_dir / "combined_feature_audit.csv", index=False)
    contribution_df = _save_variance_weighted_contribution_report(
        output_dir=output_dir,
        pca=pca,
        feature_columns=combined_columns,
    )

    overview = pd.DataFrame(
        [
            {
                **feature_cache_summary,
                **audit_summary,
                **combined_audit_summary,
                "scaler": config.preprocessing.scaler,
                "manifest_rows": int(len(manifest)),
                "existing_selected_feature_count": int(len(feature_columns)),
                "posthoc_feature_count": int(len(posthoc_columns)),
                "combined_selected_feature_count": int(len(combined_columns)),
                "full_pca_component_count": int(max_components),
            }
        ]
    )
    overview.to_csv(output_dir / "posthoc_analysis_overview.csv", index=False)

    top = contribution_df.head(min(20, len(contribution_df)))
    lines = [
        "# Posthoc Feature Analysis Summary",
        "",
        f"- manifest_rows: `{len(manifest)}`",
        f"- scaler: `{config.preprocessing.scaler}`",
        f"- existing_selected_feature_count: `{len(feature_columns)}`",
        f"- posthoc_feature_count: `{len(posthoc_columns)}`",
        f"- combined_selected_feature_count: `{len(combined_columns)}`",
        "",
        "## Top Variance-Weighted Contributors",
        "",
    ]
    for row in top.itertuples(index=False):
        lines.append(
            f"- `{row.feature}` | family=`{row.feature_family}` | "
            f"score=`{float(row.variance_weighted_squared_loading):.6f}` | dominant_pc=`PC{int(row.dominant_pc)}`"
        )
    (output_dir / "posthoc_feature_analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[posthoc-analysis] done | elapsed={perf_counter()-overall_start:.1f}s")
    return output_dir
