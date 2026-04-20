from __future__ import annotations

import os
import json
import hashlib
import re
from inspect import getsource
from time import perf_counter
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault(
    "MPLCONFIGDIR",
    str((Path(__file__).resolve().parents[2] / ".mplconfig").resolve()),
)

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgba
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from . import features as features_module
from .feature_variance_analysis import run_feature_variance_analysis
from .features import extract_engineered_features, load_patch_npz, load_patch_payload, slice_patch_payload
from .preprocessing import scale_feature_matrix
from .report_config import LatentReportConfig


NEAR_CONSTANT_STD_THRESHOLD = 1e-8
NEAR_CONSTANT_RANGE_THRESHOLD = 1e-8


def _categorize_feature_family(feature: str) -> str:
    feature = feature.lower()
    cross_channel_tokens = (
        "_flag_ha_",
        "_flag_shank2_",
        "_ha_shank2_",
        "_map2_flag_",
        "_map2_ha_",
        "_map2_shank2_",
        "flag_to_ha",
        "ha_to_flag",
        "flag_ha_",
        "shank2_to_map2",
        "plus_ha_to_shank2",
        "_pixel_corr",
        "_manders_",
        "_bright_overlap_",
        "_com_offset",
        "_spotness_ratio",
        "_compactness_ratio",
        "_mean_ratio",
    )
    map2_spatial_tokens = (
        "_inside_mean",
        "_outside_mean",
        "_inside_outside_ratio",
        "_inside_bright_fraction",
        "_outside_bright_fraction",
        "_inside_bright_outside_bright_ratio",
        "map2_mask_fraction",
        "distance_to_mask_boundary_px",
        "center_of_patch_map2_intensity",
        "map2_local_thickness_proxy",
        "_center_surround_",
    )
    z_profile_tokens = (
        "_z_peak",
        "_z_std",
        "_z_width_halfmax",
        "_z_slices_above_halfmax",
        "_z_center_of_mass",
        "_z_skewness",
        "_z_kurtosis",
        "_z_peak_count",
        "_z_multi_peak_score",
        "_z_peak_symmetry",
    )
    spotness_tokens = (
        "_estimated_punctum_area",
        "_component_count",
        "_largest_area",
        "_largest_area_fraction",
        "_eccentricity",
        "_circularity",
        "_compactness",
        "_boundary_irregularity",
        "_elongation",
        "_roundness",
        "_radial_symmetry",
        "_log_response",
        "_log_puncta_",
        "_dog_response",
        "_dominant_object_fraction",
    )
    if any(token in feature for token in cross_channel_tokens):
        return "Cross-Channel Overlap"
    if any(token in feature for token in map2_spatial_tokens):
        return "MAP2-Aware Spatial"
    if any(token in feature for token in z_profile_tokens):
        return "Z-Profile"
    if any(token in feature for token in spotness_tokens):
        return "Spotness / Morphology"
    return "Other"


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def _dataset_cache_dir(manifest_path: Path) -> Path:
    manifest_path = manifest_path.resolve()
    return manifest_path.parent.parent / "engineered_feature_cache"


def _path_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _code_fingerprint() -> dict[str, str]:
    digest = hashlib.sha256()
    digest.update(getsource(extract_engineered_features).encode("utf-8"))
    module_path = Path(features_module.__file__).resolve()
    digest.update(module_path.read_bytes())
    return {
        "extract_engineered_features_sha256": digest.hexdigest(),
        "features_module_path": str(module_path),
    }


def _manifest_cache_inputs(manifest: pd.DataFrame, manifest_path: Path, channels: list[str]) -> dict[str, object]:
    shard_entries: list[dict[str, object]] = []
    if "shard_path" in manifest.columns:
        shard_paths = sorted({str(path) for path in manifest["shard_path"].dropna().astype(str) if str(path)})
        shard_entries = [_path_fingerprint(Path(path)) for path in shard_paths]

    return {
        "channels": [str(channel) for channel in channels],
        "manifest": _path_fingerprint(manifest_path),
        "rows": int(len(manifest)),
        "patch_ids_sha256": hashlib.sha256(
            "\n".join(manifest["patch_id"].astype(str).tolist()).encode("utf-8")
        ).hexdigest(),
        "shards": shard_entries,
        "code": _code_fingerprint(),
    }


def _manifest_cache_fingerprint_from_inputs(cache_inputs: dict[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(cache_inputs, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _effective_engineered_feature_channels(channels: list[str], map2_feature_policy: str) -> list[str]:
    effective = [str(channel) for channel in channels]
    if str(map2_feature_policy).lower() in {"prior_only", "exclude_all_map2", "mask_internal_only"}:
        effective = [channel for channel in effective if channel != "MAP2"]
    return effective


def _display_channels(channels: list[str], map2_feature_policy: str) -> list[str]:
    display = [str(channel) for channel in channels]
    if str(map2_feature_policy).lower() == "mask_internal_only":
        display = [channel for channel in display if channel != "MAP2"]
    return display


def _apply_channel_scope_filter(
    *,
    feature_columns: list[str],
    allowed_channels: list[str],
    output_dir: Path,
) -> tuple[list[str], dict[str, int | str], pd.DataFrame]:
    normalized_allowed = [str(channel).upper() for channel in allowed_channels if str(channel).upper() != "MAP2"]
    if not normalized_allowed:
        raise ValueError("Channel scope filter requires at least one non-MAP2 allowed channel.")

    channel_tokens = ("MAP2", "FLAG", "HA", "SHANK2")
    channel_pattern = re.compile(r"(?<![a-z0-9])(map2|flag|ha|shank2)(?![a-z0-9])")
    allowed_tokens = {channel.lower() for channel in normalized_allowed}
    audit_rows: list[dict[str, object]] = []
    kept_columns: list[str] = []

    for feature in feature_columns:
        feature_lower = str(feature).lower()
        referenced_channels = [match.group(1).upper() for match in channel_pattern.finditer(feature_lower)]
        referenced_token_set = {token.lower() for token in referenced_channels}
        if not referenced_token_set:
            keep = False
            policy_reason = "excluded_non_channel_scoped"
        elif referenced_token_set.issubset(allowed_tokens):
            keep = True
            policy_reason = "kept"
            kept_columns.append(feature)
        else:
            keep = False
            policy_reason = "excluded_out_of_scope_channel"

        audit_rows.append(
            {
                "feature": feature,
                "allowed_channels": ",".join(normalized_allowed),
                "referenced_channels": ",".join(referenced_channels),
                "kept": keep,
                "policy_reason": policy_reason,
            }
        )

    audit_df = pd.DataFrame.from_records(audit_rows).sort_values(["kept", "feature"], ascending=[True, True])
    audit_df.to_csv(output_dir / "channel_scope_audit.csv", index=False)

    total = len(feature_columns)
    kept = len(kept_columns)
    excluded = total - kept
    excluded_out_of_scope = int((audit_df["policy_reason"] == "excluded_out_of_scope_channel").sum())
    excluded_non_channel_scoped = int((audit_df["policy_reason"] == "excluded_non_channel_scoped").sum())
    summary = {
        "channel_scope_allowed_channels": ",".join(normalized_allowed),
        "channel_scope_total_features": total,
        "channel_scope_kept_features": kept,
        "channel_scope_excluded_features": excluded,
        "channel_scope_excluded_out_of_scope_channel": excluded_out_of_scope,
        "channel_scope_excluded_non_channel_scoped": excluded_non_channel_scoped,
    }
    summary_lines = [
        "# Channel Scope Filter",
        "",
        f"- allowed_channels: `{', '.join(normalized_allowed)}`",
        f"- total_candidate_features: `{total}`",
        f"- kept_after_channel_scope: `{kept}`",
        f"- excluded_after_channel_scope: `{excluded}`",
        f"- excluded_out_of_scope_channel: `{excluded_out_of_scope}`",
        f"- excluded_non_channel_scoped: `{excluded_non_channel_scoped}`",
    ]
    (output_dir / "channel_scope_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return kept_columns, summary, audit_df


def _load_or_compute_engineered_features(
    *,
    manifest: pd.DataFrame,
    manifest_path: Path,
    channels: list[str],
    map2_feature_policy: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cache_dir = _dataset_cache_dir(manifest_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_dir / "engineered_features_metadata.json"
    parquet_path = cache_dir / "engineered_features.parquet"
    csv_path = cache_dir / "engineered_features.csv"

    effective_channels = _effective_engineered_feature_channels(
        channels=channels,
        map2_feature_policy=map2_feature_policy,
    )
    cache_inputs = _manifest_cache_inputs(manifest=manifest, manifest_path=manifest_path, channels=effective_channels)
    cache_key = _manifest_cache_fingerprint_from_inputs(cache_inputs)
    expected_patch_ids = manifest["patch_id"].astype(str).tolist()

    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = None
        if isinstance(metadata, dict):
            fmt = str(metadata.get("format", "")).lower()
            cached_key = str(metadata.get("cache_key", ""))
            cached_rows = int(metadata.get("rows", -1))
            cached_channels = [str(v) for v in metadata.get("channels", [])]
            cached_inputs = metadata.get("cache_inputs")
            cache_file = parquet_path if fmt == "parquet" else csv_path if fmt == "csv" else None
            if (
                cache_file is not None
                and cache_file.exists()
                and cached_key == cache_key
                and cached_rows == len(manifest)
                and cached_channels == [str(v) for v in effective_channels]
                and cached_inputs == cache_inputs
            ):
                start = perf_counter()
                if fmt == "parquet":
                    feature_df = pd.read_parquet(cache_file)
                else:
                    feature_df = pd.read_csv(cache_file)
                if feature_df["patch_id"].astype(str).tolist() == expected_patch_ids:
                    elapsed = perf_counter() - start
                    print(
                        f"[latent] engineered feature cache hit | format={fmt} | rows={len(feature_df)} "
                        f"| elapsed={elapsed:.1f}s"
                    )
                    return feature_df, {
                        "cache_status": "hit",
                        "cache_format": fmt,
                        "cache_dir": str(cache_dir),
                        "cache_key": cache_key,
                    }
                print("[latent] engineered feature cache mismatch on patch ordering; recomputing features.")

    start = perf_counter()
    feature_df = extract_engineered_features(manifest=manifest, channels=effective_channels)
    elapsed = perf_counter() - start
    fmt = "parquet"
    try:
        feature_df.to_parquet(parquet_path, index=False)
        if csv_path.exists():
            csv_path.unlink()
    except Exception:
        fmt = "csv"
        feature_df.to_csv(csv_path, index=False)
        if parquet_path.exists():
            parquet_path.unlink()

    metadata = {
        "cache_key": cache_key,
        "cache_inputs": cache_inputs,
        "manifest_path": str(manifest_path.resolve()),
        "channels": [str(v) for v in effective_channels],
        "rows": int(len(feature_df)),
        "columns": [str(v) for v in feature_df.columns.tolist()],
        "format": fmt,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(
        f"[latent] engineered feature cache saved | format={fmt} | rows={len(feature_df)} "
        f"| elapsed={elapsed:.1f}s | dir={cache_dir}"
    )
    return feature_df, {
        "cache_status": "miss",
        "cache_format": fmt,
        "cache_dir": str(cache_dir),
        "cache_key": cache_key,
    }


def _apply_map2_feature_policy(
    *,
    feature_columns: list[str],
    policy: str,
    output_dir: Path,
) -> tuple[list[str], dict[str, int | str], pd.DataFrame]:
    policy = policy.lower()
    allowed_map2_context = {
        "map2_mask_fraction",
        "distance_to_mask_boundary_px",
        "map2_local_thickness_proxy",
    }
    map2_spatial_tokens = (
        "_inside_mean",
        "_outside_mean",
        "_inside_outside_ratio",
        "_inside_bright_fraction",
        "_outside_bright_fraction",
        "_inside_bright_outside_bright_ratio",
        "map2_mask_fraction",
        "distance_to_mask_boundary_px",
        "center_of_patch_map2_intensity",
        "map2_local_thickness_proxy",
        "_center_surround_",
    )
    mask_context_tokens = map2_spatial_tokens + ("_pixel_corr_in_mask",)
    audit_rows: list[dict[str, object]] = []
    kept_columns: list[str] = []

    for feature in feature_columns:
        feature_lower = feature.lower()
        contains_map2 = "map2" in feature_lower
        is_map2_spatial = any(token in feature_lower for token in map2_spatial_tokens)
        is_mask_context = any(token in feature_lower for token in mask_context_tokens)
        is_posthoc_map2_dendrite = feature_lower.startswith("posthoc_map2_")
        keep = True
        policy_reason = "kept"

        if policy == "prior_only":
            if feature_lower in allowed_map2_context:
                keep = True
                policy_reason = "allowed_map2_prior_context"
            elif contains_map2:
                keep = False
                policy_reason = "excluded_map2_as_signal"
        elif policy == "exclude_spatial":
            if is_map2_spatial:
                keep = False
                policy_reason = "excluded_map2_spatial"
        elif policy == "exclude_spatial_and_dendrite":
            if is_map2_spatial:
                keep = False
                policy_reason = "excluded_map2_spatial"
            elif is_posthoc_map2_dendrite:
                keep = False
                policy_reason = "excluded_posthoc_map2_dendrite"
        elif policy == "exclude_all_map2":
            if contains_map2:
                keep = False
                policy_reason = "excluded_all_map2_signal"
            elif is_map2_spatial:
                keep = False
                policy_reason = "excluded_map2_spatial"
            elif is_posthoc_map2_dendrite:
                keep = False
                policy_reason = "excluded_posthoc_map2_dendrite"
        elif policy == "mask_internal_only":
            if contains_map2:
                keep = False
                policy_reason = "excluded_all_map2_signal"
            elif is_mask_context:
                keep = False
                policy_reason = "excluded_mask_context"
            elif is_posthoc_map2_dendrite:
                keep = False
                policy_reason = "excluded_posthoc_map2_dendrite"

        if keep:
            kept_columns.append(feature)

        audit_rows.append(
            {
                "feature": feature,
                "contains_map2_token": contains_map2,
                "is_map2_spatial": is_map2_spatial,
                "is_mask_context": is_mask_context,
                "is_posthoc_map2_dendrite": is_posthoc_map2_dendrite,
                "policy": policy,
                "kept": keep,
                "policy_reason": policy_reason,
            }
        )

    audit_df = pd.DataFrame.from_records(audit_rows).sort_values(["kept", "feature"], ascending=[True, True])
    audit_df.to_csv(output_dir / "map2_feature_policy_audit.csv", index=False)

    total = len(feature_columns)
    kept = len(kept_columns)
    excluded = total - kept
    excluded_map2_signal = int((audit_df["policy_reason"] == "excluded_map2_as_signal").sum())
    excluded_all_map2_signal = int((audit_df["policy_reason"] == "excluded_all_map2_signal").sum())
    excluded_map2_spatial = int((audit_df["policy_reason"] == "excluded_map2_spatial").sum())
    excluded_mask_context = int((audit_df["policy_reason"] == "excluded_mask_context").sum())
    excluded_posthoc_map2_dendrite = int((audit_df["policy_reason"] == "excluded_posthoc_map2_dendrite").sum())
    summary = {
        "map2_feature_policy": policy,
        "map2_policy_total_features": total,
        "map2_policy_kept_features": kept,
        "map2_policy_excluded_features": excluded,
        "map2_policy_excluded_map2_as_signal": excluded_map2_signal,
        "map2_policy_excluded_all_map2_signal": excluded_all_map2_signal,
        "map2_policy_excluded_map2_spatial": excluded_map2_spatial,
        "map2_policy_excluded_mask_context": excluded_mask_context,
        "map2_policy_excluded_posthoc_map2_dendrite": excluded_posthoc_map2_dendrite,
    }
    summary_lines = [
        "# MAP2 Feature Policy",
        "",
        f"- policy: `{policy}`",
        f"- total_candidate_features: `{total}`",
        f"- kept_after_policy: `{kept}`",
        f"- excluded_after_policy: `{excluded}`",
        f"- excluded_map2_as_signal: `{excluded_map2_signal}`",
        f"- excluded_all_map2_signal: `{excluded_all_map2_signal}`",
        f"- excluded_map2_spatial: `{excluded_map2_spatial}`",
        f"- excluded_mask_context: `{excluded_mask_context}`",
        f"- excluded_posthoc_map2_dendrite: `{excluded_posthoc_map2_dendrite}`",
    ]
    if policy == "prior_only":
        summary_lines.extend(
            [
                "",
                "## Prior-Only Rules",
                "",
                "- kept MAP2-derived context only:",
                "  - `map2_mask_fraction`",
                "  - `distance_to_mask_boundary_px`",
                "  - `map2_local_thickness_proxy`",
                "- excluded all other features containing `map2` in the feature name",
            ]
        )
    elif policy == "exclude_spatial":
        summary_lines.extend(
            [
                "",
                "## Exclude-Spatial Rules",
                "",
                "- excluded the MAP2-aware spatial family:",
                "  - inside/outside features",
                "  - center/surround features",
                "  - `map2_mask_fraction`",
                "  - `distance_to_mask_boundary_px`",
                "  - `center_of_patch_map2_intensity`",
                "  - `map2_local_thickness_proxy`",
                "- kept other MAP2-derived features such as intensity, texture, morphology, and z-profile features",
            ]
        )
    elif policy == "exclude_spatial_and_dendrite":
        summary_lines.extend(
            [
                "",
                "## Exclude-Spatial-And-Dendrite Rules",
                "",
                "- excluded the MAP2-aware spatial family:",
                "  - inside/outside features",
                "  - center/surround features",
                "  - `map2_mask_fraction`",
                "  - `distance_to_mask_boundary_px`",
                "  - `center_of_patch_map2_intensity`",
                "  - `map2_local_thickness_proxy`",
                "- excluded post-hoc MAP2 dendrite geometry features:",
                "  - `posthoc_map2_*`",
                "- kept other MAP2-derived features such as intensity, texture, morphology, and z-profile features",
            ]
        )
    elif policy == "exclude_all_map2":
        summary_lines.extend(
            [
                "",
                "## Exclude-All-MAP2 Rules",
                "",
                "- excluded all features containing `map2` in the feature name",
                "- excluded the MAP2-aware spatial family even when the feature name does not itself contain `map2`:",
                "  - inside/outside features",
                "  - center/surround features",
                "- excluded post-hoc MAP2 dendrite geometry features:",
                "  - `posthoc_map2_*`",
                "- the MAP2 mask can still be used internally for patch extraction and mask-aware calculations, but no MAP2-derived signal features are kept in the modeling feature set",
            ]
        )
    elif policy == "mask_internal_only":
        summary_lines.extend(
            [
                "",
                "## Mask-Internal-Only Rules",
                "",
                "- excluded all features containing `map2` in the feature name",
                "- excluded all mask-context analysis features:",
                "  - inside/outside features",
                "  - center/surround features",
                "  - `map2_mask_fraction`",
                "  - `distance_to_mask_boundary_px`",
                "  - `center_of_patch_map2_intensity`",
                "  - `map2_local_thickness_proxy`",
                "  - `*_pixel_corr_in_mask`",
                "- excluded post-hoc MAP2 dendrite geometry features:",
                "  - `posthoc_map2_*`",
                "- the MAP2 mask is retained only for internal sampling/extraction logic and is not represented in the modeling feature set",
            ]
        )
    (output_dir / "map2_feature_policy_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(
        f"[latent] MAP2 feature policy | policy={policy} | kept={kept}/{total} "
        f"| excluded_map2_as_signal={excluded_map2_signal} | excluded_all_map2_signal={excluded_all_map2_signal} "
        f"| excluded_map2_spatial={excluded_map2_spatial} | excluded_mask_context={excluded_mask_context} "
        f"| excluded_posthoc_map2_dendrite={excluded_posthoc_map2_dendrite}"
    )
    return kept_columns, summary, audit_df


def _fit_clusters(embedding: np.ndarray, method: str, n_clusters: int, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    embedding = np.asarray(embedding, dtype=np.float64)
    n_clusters = max(2, min(int(n_clusters), embedding.shape[0]))
    if method == "gmm":
        for reg_covar in (1e-6, 1e-5, 1e-4, 1e-3):
            try:
                model = GaussianMixture(
                    n_components=n_clusters,
                    random_state=random_seed,
                    reg_covar=reg_covar,
                )
                labels = model.fit_predict(embedding)
                centers = model.means_
                return labels.astype(int), centers
            except ValueError:
                continue
        print(
            "[clustering] GMM failed after regularization retries; "
            "falling back to kmeans for this run."
        )
    model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = model.fit_predict(embedding)
    return labels.astype(int), model.cluster_centers_


def _audit_and_filter_feature_columns(
    *,
    report_df: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> tuple[list[str], dict[str, int | bool | str], pd.DataFrame]:
    audit_rows: list[dict[str, object]] = []
    kept_columns: list[str] = []

    for feature in feature_columns:
        values = report_df[feature].to_numpy(dtype=np.float64)
        finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            mean = std = min_value = max_value = value_range = 0.0
            n_unique = 0
        else:
            mean = float(np.mean(finite_values))
            std = float(np.std(finite_values))
            min_value = float(np.min(finite_values))
            max_value = float(np.max(finite_values))
            value_range = float(max_value - min_value)
            n_unique = int(pd.Series(finite_values).nunique(dropna=True))

        is_all_zero = finite_values.size > 0 and np.allclose(finite_values, 0.0, atol=1e-12, rtol=0.0)
        is_constant = n_unique <= 1
        is_near_constant = (not is_constant) and (
            std <= NEAR_CONSTANT_STD_THRESHOLD or value_range <= NEAR_CONSTANT_RANGE_THRESHOLD
        )

        if is_all_zero:
            drop_reason = "all_zero"
        elif is_constant:
            drop_reason = "constant"
        elif is_near_constant:
            drop_reason = "near_constant"
        else:
            drop_reason = ""
            kept_columns.append(feature)

        audit_rows.append(
            {
                "feature": feature,
                "mean": mean,
                "std": std,
                "min": min_value,
                "max": max_value,
                "range": value_range,
                "n_unique": n_unique,
                "is_all_zero": is_all_zero,
                "is_constant": is_constant,
                "is_near_constant": is_near_constant,
                "drop_reason": drop_reason,
                "kept": drop_reason == "",
            }
        )

    audit_df = pd.DataFrame.from_records(audit_rows).sort_values(["kept", "std", "range"], ascending=[True, True, True])
    audit_df.to_csv(output_dir / "feature_variance_audit.csv", index=False)

    total = len(feature_columns)
    kept = len(kept_columns)
    dropped = total - kept
    all_zero = int((audit_df["drop_reason"] == "all_zero").sum())
    constant = int((audit_df["drop_reason"] == "constant").sum())
    near_constant = int((audit_df["drop_reason"] == "near_constant").sum())
    low_information = kept < 2 or kept < min(10, total) or (total > 0 and kept / total < 0.25)

    summary_lines = [
        "# Feature Variance Audit",
        "",
        f"- total_features: `{total}`",
        f"- kept_features: `{kept}`",
        f"- dropped_features: `{dropped}`",
        f"- dropped_all_zero: `{all_zero}`",
        f"- dropped_constant: `{constant}`",
        f"- dropped_near_constant: `{near_constant}`",
        f"- low_information_subset: `{low_information}`",
    ]
    if low_information:
        summary_lines.extend(
            [
                "",
                "## Warning",
                "",
                "This feature subset appears low-information after dropping constant and near-constant features.",
                "Interpret PCA, clustering, and separation outputs cautiously for this run.",
            ]
        )
    (output_dir / "feature_variance_audit_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    status = (
        f"[feature audit] kept={kept}/{total}, dropped={dropped} "
        f"(all_zero={all_zero}, constant={constant}, near_constant={near_constant})"
    )
    print(status)
    if low_information:
        print(
            "[feature audit warning] Remaining feature subset is low-information; "
            "downstream latent structure may be unstable or hard to interpret."
        )

    return kept_columns, {
        "total_features": total,
        "kept_features": kept,
        "dropped_features": dropped,
        "dropped_all_zero": all_zero,
        "dropped_constant": constant,
        "dropped_near_constant": near_constant,
        "low_information_subset": low_information,
    }, audit_df


def _select_representatives(embedding: np.ndarray, labels: np.ndarray, centers: np.ndarray, k: int) -> dict[int, list[int]]:
    representatives: dict[int, list[int]] = {}
    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embedding = embedding[cluster_indices]
        center = centers[int(cluster_id)]
        distances = np.linalg.norm(cluster_embedding - center, axis=1)
        ordered = cluster_indices[np.argsort(distances)]
        representatives[int(cluster_id)] = ordered[:k].tolist()
    return representatives


def _select_extremes(embedding: np.ndarray, labels: np.ndarray, centers: np.ndarray, k: int) -> dict[int, list[int]]:
    extremes: dict[int, list[int]] = {}
    for cluster_id in np.unique(labels):
        cluster_indices = np.where(labels == cluster_id)[0]
        cluster_embedding = embedding[cluster_indices]
        center = centers[int(cluster_id)]
        distances = np.linalg.norm(cluster_embedding - center, axis=1)
        ordered = cluster_indices[np.argsort(distances)[::-1]]
        extremes[int(cluster_id)] = ordered[:k].tolist()
    return extremes


def _save_pca_plot(output_dir: Path, report_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        report_df["pca_1"],
        report_df["pca_2"],
        c=report_df["cluster_id"],
        cmap="tab10",
        s=18,
        alpha=0.8,
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Latent Patch Clusters")
    fig.colorbar(scatter, ax=ax, label="cluster_id")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_clusters.png", dpi=150)
    plt.close(fig)


def _layer_alpha(index: int, total: int, *, bottom_alpha: float = 0.7, top_alpha: float = 0.35) -> float:
    total = max(int(total), 1)
    if total == 1:
        return float(bottom_alpha)
    frac = float(index) / float(total - 1)
    return float(bottom_alpha + frac * (top_alpha - bottom_alpha))


def _ordered_condition_levels(levels: list[str]) -> list[str]:
    ordered = []
    if "positive" in levels:
        ordered.append("positive")
    remaining = sorted(level for level in levels if level not in {"positive", "negative"})
    ordered.extend(remaining)
    if "negative" in levels:
        ordered.append("negative")
    return ordered


def _scatter_category_layer(
    ax,
    x,
    y,
    *,
    color: str,
    label: str,
    layer_index: int,
    layer_total: int,
    size: float = 10.0,
) -> None:
    edge_alpha = _layer_alpha(layer_index, layer_total, bottom_alpha=0.85, top_alpha=0.55)
    face_alpha = _layer_alpha(layer_index, layer_total, bottom_alpha=0.22, top_alpha=0.08)
    ax.scatter(
        x,
        y,
        s=size,
        label=label,
        facecolors=to_rgba(color, face_alpha),
        edgecolors=to_rgba(color, edge_alpha),
        linewidths=0.35,
        rasterized=True,
    )


def _save_latent_structure_interpretation(output_dir: Path, report_df: pd.DataFrame) -> None:
    interpretation_dir = output_dir / "latent_structure_interpretation"
    interpretation_dir.mkdir(parents=True, exist_ok=True)

    eps = 1e-6
    total_patches = len(report_df)
    condition_levels = _ordered_condition_levels(report_df["condition"].dropna().astype(str).unique().tolist())
    patch_group_levels = sorted(report_df["patch_group"].dropna().astype(str).unique().tolist())

    overall_condition_counts = report_df["condition"].value_counts(dropna=False)
    overall_condition_frac = (overall_condition_counts / max(1, total_patches)).to_dict()
    overall_patch_group_counts = report_df["patch_group"].value_counts(dropna=False)
    overall_patch_group_frac = (overall_patch_group_counts / max(1, total_patches)).to_dict()

    cluster_condition_rows: list[dict[str, object]] = []
    cluster_patch_group_rows: list[dict[str, object]] = []
    cluster_summary_rows: list[dict[str, object]] = []

    for cluster_id, cluster_df in report_df.groupby("cluster_id", dropna=False):
        cluster_size = len(cluster_df)
        condition_counts = cluster_df["condition"].value_counts(dropna=False)
        condition_frac = (condition_counts / max(1, cluster_size)).to_dict()
        patch_group_counts = cluster_df["patch_group"].value_counts(dropna=False)
        patch_group_frac = (patch_group_counts / max(1, cluster_size)).to_dict()

        for condition in condition_levels:
            frac = float(condition_frac.get(condition, 0.0))
            baseline = float(overall_condition_frac.get(condition, 0.0))
            cluster_condition_rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_size,
                    "condition": condition,
                    "count": int(condition_counts.get(condition, 0)),
                    "fraction_within_cluster": frac,
                    "global_fraction": baseline,
                    "log2_enrichment_vs_global": float(np.log2((frac + eps) / (baseline + eps))),
                }
            )

        for patch_group in patch_group_levels:
            frac = float(patch_group_frac.get(patch_group, 0.0))
            baseline = float(overall_patch_group_frac.get(patch_group, 0.0))
            cluster_patch_group_rows.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_size,
                    "patch_group": patch_group,
                    "count": int(patch_group_counts.get(patch_group, 0)),
                    "fraction_within_cluster": frac,
                    "global_fraction": baseline,
                    "log2_enrichment_vs_global": float(np.log2((frac + eps) / (baseline + eps))),
                }
            )

        positive_fraction = float(condition_frac.get("positive", 0.0))
        negative_fraction = float(condition_frac.get("negative", 0.0))
        if 0.4 <= positive_fraction <= 0.6 and 0.4 <= negative_fraction <= 0.6:
            condition_status = "mixed"
        elif positive_fraction > negative_fraction:
            condition_status = "positive_enriched"
        elif negative_fraction > positive_fraction:
            condition_status = "negative_enriched"
        else:
            condition_status = "mixed"
        dominant_patch_group = max(patch_group_frac.items(), key=lambda item: item[1])[0] if patch_group_frac else "none"
        cluster_summary_rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": cluster_size,
                "positive_fraction": positive_fraction,
                "negative_fraction": negative_fraction,
                "condition_status": condition_status,
                "dominant_patch_group": dominant_patch_group,
                "dominant_patch_group_fraction": float(patch_group_frac.get(dominant_patch_group, 0.0)),
            }
        )

    cluster_condition_df = pd.DataFrame.from_records(cluster_condition_rows)
    cluster_patch_group_df = pd.DataFrame.from_records(cluster_patch_group_rows)
    cluster_summary_df = pd.DataFrame.from_records(cluster_summary_rows).sort_values("cluster_id")
    cluster_condition_df.to_csv(interpretation_dir / "cluster_condition_enrichment.csv", index=False)
    cluster_patch_group_df.to_csv(interpretation_dir / "cluster_patchgroup_enrichment.csv", index=False)
    cluster_summary_df.to_csv(interpretation_dir / "cluster_interpretation_summary.csv", index=False)

    composition_df = (
        report_df.groupby(["cluster_id", "condition", "patch_group"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["cluster_id", "condition", "patch_group"])
    )
    composition_df.to_csv(interpretation_dir / "cluster_condition_patchgroup_composition.csv", index=False)

    # PCA scatter colored by condition
    condition_palette = {
        "negative": "#4C78A8",
        "positive": "#E45756",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, condition in enumerate(condition_levels):
        subset = report_df[report_df["condition"].astype(str) == condition]
        if subset.empty:
            continue
        _scatter_category_layer(
            ax,
            subset["pca_1"],
            subset["pca_2"],
            size=9,
            label=condition,
            color=condition_palette.get(condition, "#9D9DA1"),
            layer_index=idx,
            layer_total=len(condition_levels),
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("PCA Scatter Colored by Condition")
    ax.legend(title="Condition")
    fig.tight_layout()
    fig.savefig(interpretation_dir / "pca_scatter_by_condition.png", dpi=150)
    plt.close(fig)

    # PCA scatter colored by patch_group
    patch_group_palette = {
        "in_mask": "#54A24B",
        "boundary": "#F58518",
        "near_mask_outside": "#4C78A8",
        "far_background": "#B279A2",
    }
    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, patch_group in enumerate(patch_group_levels):
        subset = report_df[report_df["patch_group"].astype(str) == patch_group]
        if subset.empty:
            continue
        _scatter_category_layer(
            ax,
            subset["pca_1"],
            subset["pca_2"],
            size=9,
            label=patch_group,
            color=patch_group_palette.get(patch_group, "#9D9DA1"),
            layer_index=idx,
            layer_total=len(patch_group_levels),
        )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("PCA Scatter Colored by Patch Group")
    ax.legend(title="Patch Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(interpretation_dir / "pca_scatter_by_patchgroup.png", dpi=150)
    plt.close(fig)

    # Individual PCA scatter per patch_group, colored by condition
    for patch_group in patch_group_levels:
        subset_group = report_df[report_df["patch_group"].astype(str) == patch_group]
        if subset_group.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        for idx, condition in enumerate(condition_levels):
            subset = subset_group[subset_group["condition"].astype(str) == condition]
            if subset.empty:
                continue
            _scatter_category_layer(
                ax,
                subset["pca_1"],
                subset["pca_2"],
                size=10,
                label=condition,
                color=condition_palette.get(condition, "#9D9DA1"),
                layer_index=idx,
                layer_total=len(condition_levels),
            )
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.set_title(f"PCA Scatter for {patch_group} Colored by Condition")
        ax.legend(title="Condition")
        fig.tight_layout()
        safe_group = str(patch_group).lower().replace(" ", "_").replace("/", "_")
        fig.savefig(interpretation_dir / f"pca_scatter_{safe_group}_by_condition.png", dpi=150)
        plt.close(fig)

    # Cluster-level plots
    condition_pivot = cluster_condition_df.pivot(index="cluster_id", columns="condition", values="fraction_within_cluster").fillna(0.0)
    fig, ax = plt.subplots(figsize=(10, 5))
    bottom = np.zeros(len(condition_pivot), dtype=np.float64)
    for condition in condition_pivot.columns:
        vals = condition_pivot[condition].to_numpy(dtype=np.float64)
        ax.bar(condition_pivot.index.astype(str), vals, bottom=bottom, label=condition)
        bottom += vals
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Fraction Within Cluster")
    ax.set_title("Condition Composition by Cluster")
    ax.legend(title="Condition")
    fig.tight_layout()
    fig.savefig(interpretation_dir / "cluster_condition_composition.png", dpi=150)
    plt.close(fig)

    patch_group_pivot = cluster_patch_group_df.pivot(index="cluster_id", columns="patch_group", values="fraction_within_cluster").fillna(0.0)
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom = np.zeros(len(patch_group_pivot), dtype=np.float64)
    for patch_group in patch_group_pivot.columns:
        vals = patch_group_pivot[patch_group].to_numpy(dtype=np.float64)
        ax.bar(patch_group_pivot.index.astype(str), vals, bottom=bottom, label=patch_group)
        bottom += vals
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Fraction Within Cluster")
    ax.set_title("Patch-Group Composition by Cluster")
    ax.legend(title="Patch Group", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(interpretation_dir / "cluster_patchgroup_composition.png", dpi=150)
    plt.close(fig)

    # PCA-space binning
    bins = 30
    x = report_df["pca_1"].to_numpy(dtype=np.float64)
    y = report_df["pca_2"].to_numpy(dtype=np.float64)
    x_edges = np.linspace(float(np.min(x)), float(np.max(x)), bins + 1)
    y_edges = np.linspace(float(np.min(y)), float(np.max(y)), bins + 1)
    all_hist, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])

    positive_mask = report_df["condition"].astype(str).eq("positive").to_numpy()
    negative_mask = report_df["condition"].astype(str).eq("negative").to_numpy()
    positive_hist, _, _ = np.histogram2d(x[positive_mask], y[positive_mask], bins=[x_edges, y_edges])
    negative_hist, _, _ = np.histogram2d(x[negative_mask], y[negative_mask], bins=[x_edges, y_edges])

    positive_fraction_grid = np.divide(positive_hist, all_hist, out=np.zeros_like(positive_hist, dtype=np.float64), where=all_hist > 0)
    log2_pos_neg = np.log2((positive_hist + eps) / (negative_hist + eps))

    extent = [x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        positive_fraction_grid.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Positive Fraction Across PCA Space")
    fig.colorbar(im, ax=ax, label="Positive Fraction")
    fig.tight_layout()
    fig.savefig(interpretation_dir / "pca_condition_positive_fraction_heatmap.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    vmax = float(np.nanmax(np.abs(log2_pos_neg))) if np.isfinite(log2_pos_neg).any() else 1.0
    if vmax <= 0.0:
        vmax = 1.0
    im = ax.imshow(
        log2_pos_neg.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.set_title("Positive vs Negative Log2 Enrichment Across PCA Space")
    fig.colorbar(im, ax=ax, label="log2(positive / negative)")
    fig.tight_layout()
    fig.savefig(interpretation_dir / "pca_condition_log2_enrichment_heatmap.png", dpi=150)
    plt.close(fig)

    # Patch-group PCA heatmaps
    n_groups = len(patch_group_levels)
    ncols = 2
    nrows = int(np.ceil(max(1, n_groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, max(4, 4 * nrows)))
    axes_arr = np.atleast_1d(axes).reshape(nrows, ncols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for ax, patch_group in zip(axes_arr.ravel(), patch_group_levels):
        ax.axis("on")
        mask = report_df["patch_group"].astype(str).eq(patch_group).to_numpy()
        group_hist, _, _ = np.histogram2d(x[mask], y[mask], bins=[x_edges, y_edges])
        group_fraction_grid = np.divide(group_hist, all_hist, out=np.zeros_like(group_hist, dtype=np.float64), where=all_hist > 0)
        im = ax.imshow(
            group_fraction_grid.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap="magma",
            vmin=0.0,
            vmax=float(np.max(group_fraction_grid)) if np.max(group_fraction_grid) > 0 else 1.0,
        )
        ax.set_title(str(patch_group))
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Patch-Group Fraction Across PCA Space", fontsize=12)
    fig.tight_layout()
    fig.savefig(interpretation_dir / "pca_patchgroup_fraction_heatmaps.png", dpi=150)
    plt.close(fig)

    summary_lines = [
        "# Latent Structure Interpretation",
        "",
        "This folder summarizes how condition and patch_group distribute across the shared latent space.",
        "",
        "Key questions addressed:",
        "- which PCA regions contain more positive patches?",
        "- which clusters are enriched in negatives?",
        "- which clusters are mixed?",
        "- how do condition and patch_group distribute across latent structure?",
    ]
    (interpretation_dir / "README.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def _save_pca_variance_plots(output_dir: Path, pca: PCA) -> None:
    explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float32)
    cumulative = np.cumsum(explained)
    pcs = np.arange(1, explained.size + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(pcs, explained, color="#4C78A8")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Per-PC Variance Contribution")
    ax.set_xticks(pcs)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_explained_variance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pcs, cumulative, marker="o", color="#F58518")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("Cumulative PCA Variance")
    ax.set_xticks(pcs)
    ax.set_ylim(0.0, 1.05)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_cumulative_variance.png", dpi=150)
    plt.close(fig)


def _save_pca_loadings(output_dir: Path, pca: PCA, feature_columns: list[str]) -> pd.DataFrame:
    loading_matrix = np.asarray(pca.components_, dtype=np.float32).T
    loading_df = pd.DataFrame(
        loading_matrix,
        index=feature_columns,
        columns=[f"PC{i+1}" for i in range(loading_matrix.shape[1])],
    ).reset_index(names="feature")
    loading_df.to_csv(output_dir / "pca_loadings.csv", index=False)

    abs_loading_df = loading_df.copy()
    pc_columns = [c for c in abs_loading_df.columns if c.startswith("PC")]
    abs_loading_df[pc_columns] = abs_loading_df[pc_columns].abs()
    abs_loading_df.to_csv(output_dir / "pca_abs_loadings.csv", index=False)

    top_rows: list[dict[str, object]] = []
    for pc in pc_columns:
        top = (
            abs_loading_df[["feature", pc]]
            .sort_values(pc, ascending=False)
            .head(20)
            .rename(columns={pc: "abs_loading"})
        )
        for rank, row in enumerate(top.itertuples(index=False), start=1):
            top_rows.append({"pc": pc, "rank": rank, "feature": row.feature, "abs_loading": row.abs_loading})
    pd.DataFrame.from_records(top_rows).to_csv(output_dir / "pca_top_loadings.csv", index=False)

    heatmap_features = (
        abs_loading_df.assign(max_abs_loading=abs_loading_df[pc_columns].max(axis=1))
        .sort_values("max_abs_loading", ascending=False)
        .head(min(40, len(abs_loading_df)))
    )
    heatmap_values = heatmap_features[pc_columns].to_numpy(dtype=np.float32)
    vmax = float(np.max(heatmap_values)) if heatmap_values.size else 1.0
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    fig_h = max(6, 0.22 * len(heatmap_features))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    im = ax.imshow(
        heatmap_values,
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_xticks(np.arange(len(pc_columns)))
    ax.set_xticklabels(pc_columns)
    ax.set_yticks(np.arange(len(heatmap_features)))
    ax.set_yticklabels(heatmap_features["feature"].tolist(), fontsize=8)
    ax.set_title("PCA Absolute Loadings Heatmap")
    fig.colorbar(im, ax=ax, label="Absolute Loading Magnitude")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_loadings_heatmap.png", dpi=150)
    plt.close(fig)

    return loading_df


def _compute_cluster_separation_scores(report_df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    labels = report_df["cluster_id"].to_numpy()
    unique_labels = np.unique(labels)
    if unique_labels.size <= 1:
        return pd.DataFrame({"feature": feature_columns, "between_cluster_var": 0.0, "within_cluster_var": 0.0, "separation_score": 0.0})

    rows: list[dict[str, float | str]] = []
    for feature in feature_columns:
        values = report_df[feature].to_numpy(dtype=np.float32)
        global_mean = float(np.mean(values))
        between = 0.0
        within = 0.0
        for label in unique_labels:
            cluster_vals = values[labels == label]
            if cluster_vals.size == 0:
                continue
            cluster_mean = float(np.mean(cluster_vals))
            between += float(cluster_vals.size) * (cluster_mean - global_mean) ** 2
            within += float(np.sum((cluster_vals - cluster_mean) ** 2))
        separation = _safe_ratio(between, within + 1e-6)
        rows.append(
            {
                "feature": feature,
                "between_cluster_var": float(between),
                "within_cluster_var": float(within),
                "separation_score": float(separation),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("separation_score", ascending=False).reset_index(drop=True)


def _save_cluster_separation_outputs(output_dir: Path, separation_df: pd.DataFrame) -> None:
    separation_df.to_csv(output_dir / "feature_cluster_separation.csv", index=False)
    separation_df = separation_df.copy()
    separation_df["feature_category"] = separation_df["feature"].map(_categorize_feature_family)
    separation_df.to_csv(output_dir / "feature_cluster_separation_with_categories.csv", index=False)
    category_colors = {
        "Spotness / Morphology": "#4C78A8",
        "MAP2-Aware Spatial": "#54A24B",
        "Z-Profile": "#F58518",
        "Cross-Channel Overlap": "#E45756",
        "Other": "#9D9DA1",
    }
    plot_df = separation_df.iloc[::-1]
    bar_colors = [category_colors.get(category, "#9D9DA1") for category in plot_df["feature_category"]]
    fig_h = max(8, 0.22 * len(plot_df))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(plot_df["feature"], plot_df["separation_score"], color=bar_colors)
    ax.set_xlabel("Between/Within Cluster Variance Ratio")
    ax.set_ylabel("Feature")
    ax.set_title("Features Driving Cluster Separation")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, label=label)
        for label, color in category_colors.items()
        if label in plot_df["feature_category"].values
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, title="Feature Category", loc="lower right")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_cluster_separation_all_features.png", dpi=150)
    plt.close(fig)


def _save_within_family_redundancy_audit(
    *,
    output_dir: Path,
    separation_df: pd.DataFrame,
    loading_df: pd.DataFrame,
) -> None:
    pc_columns = [column for column in loading_df.columns if column.startswith("PC")]
    if not pc_columns:
        return

    loadings = loading_df.copy()
    loadings["feature_category"] = loadings["feature"].map(_categorize_feature_family)
    separation = separation_df.copy()
    separation["feature_category"] = separation["feature"].map(_categorize_feature_family)
    merged = separation.merge(loadings, on=["feature", "feature_category"], how="inner")
    if merged.empty:
        return

    audit_dir = output_dir / "within_family_redundancy_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    pair_rows: list[dict[str, object]] = []
    group_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []

    for family, family_df in merged.groupby("feature_category", dropna=False):
        family_df = family_df.reset_index(drop=True)
        family_df.to_csv(
            audit_dir / f"{str(family).lower().replace(' ', '_').replace('/', '_')}_family_features.csv",
            index=False,
        )
        if len(family_df) < 2:
            continue

        metric_columns = [
            "between_cluster_var",
            "within_cluster_var",
            "separation_score",
            *pc_columns,
        ]
        metric_matrix = family_df[metric_columns].copy()
        metric_matrix["between_cluster_var"] = np.log10(metric_matrix["between_cluster_var"].to_numpy(dtype=np.float64) + 1e-6)
        metric_matrix["within_cluster_var"] = np.log10(metric_matrix["within_cluster_var"].to_numpy(dtype=np.float64) + 1e-6)
        metric_matrix["separation_score"] = np.log10(metric_matrix["separation_score"].to_numpy(dtype=np.float64) + 1e-6)
        metric_matrix[pc_columns] = metric_matrix[pc_columns].abs()
        scaled = StandardScaler().fit_transform(metric_matrix.to_numpy(dtype=np.float64))

        linkage_matrix = linkage(scaled, method="ward")
        family_cluster_count = max(1, min(3, len(family_df) - 1))
        group_labels = fcluster(linkage_matrix, t=family_cluster_count, criterion="maxclust")
        family_df = family_df.copy()
        family_df["family_redundancy_group"] = group_labels.astype(int)

        features = family_df["feature"].tolist()
        variance_profiles = family_df[["between_cluster_var", "within_cluster_var", "separation_score"]].to_numpy(dtype=np.float64)
        loading_profiles = family_df[pc_columns].abs().to_numpy(dtype=np.float64)

        pair_distance_values: list[float] = []
        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                pair_distance_values.append(float(np.linalg.norm(scaled[i] - scaled[j])))
        distance_cutoff = float(np.quantile(pair_distance_values, 0.25)) if pair_distance_values else 0.0

        for i in range(len(features)):
            for j in range(i + 1, len(features)):
                variance_corr = float(np.corrcoef(variance_profiles[i], variance_profiles[j])[0, 1])
                loading_corr = float(np.corrcoef(loading_profiles[i], loading_profiles[j])[0, 1])
                combined_distance = float(np.linalg.norm(scaled[i] - scaled[j]))
                same_group = int(group_labels[i]) == int(group_labels[j])
                true_redundant = (
                    same_group
                    and np.isfinite(variance_corr)
                    and np.isfinite(loading_corr)
                    and variance_corr >= 0.95
                    and loading_corr >= 0.95
                    and combined_distance <= distance_cutoff
                )
                pair_rows.append(
                    {
                        "feature_category": family,
                        "feature_a": features[i],
                        "feature_b": features[j],
                        "variance_profile_corr": variance_corr,
                        "loading_profile_corr": loading_corr,
                        "combined_distance": combined_distance,
                        "distance_cutoff": distance_cutoff,
                        "same_hierarchical_group": same_group,
                        "true_redundant_candidate": true_redundant,
                    }
                )

        for group_id, group_df in family_df.groupby("family_redundancy_group", dropna=False):
            if len(group_df) < 2:
                continue
            group_features = group_df["feature"].tolist()
            group_pair_df = pd.DataFrame(
                [
                    row for row in pair_rows
                    if row["feature_category"] == family
                    and row["feature_a"] in group_features
                    and row["feature_b"] in group_features
                ]
            )
            mean_variance_corr = float(group_pair_df["variance_profile_corr"].mean()) if not group_pair_df.empty else np.nan
            mean_loading_corr = float(group_pair_df["loading_profile_corr"].mean()) if not group_pair_df.empty else np.nan
            mean_distance = float(group_pair_df["combined_distance"].mean()) if not group_pair_df.empty else np.nan
            true_redundant_group = (
                not group_pair_df.empty
                and np.isfinite(mean_variance_corr)
                and np.isfinite(mean_loading_corr)
                and mean_variance_corr >= 0.95
                and mean_loading_corr >= 0.95
                and (group_pair_df["true_redundant_candidate"].all())
            )
            representative = (
                group_df.sort_values(["separation_score", *pc_columns], ascending=[False, *([False] * len(pc_columns))])
                .iloc[0]
            )
            representative_rows.append(
                {
                    "feature_category": family,
                    "family_redundancy_group": int(group_id),
                    "representative_feature": representative["feature"],
                    "representative_separation_score": representative["separation_score"],
                    "group_size": len(group_df),
                    "true_redundant_group": true_redundant_group,
                }
            )
            for _, row in group_df.iterrows():
                group_rows.append(
                    {
                        "feature_category": family,
                        "family_redundancy_group": int(group_id),
                        "feature": row["feature"],
                        "separation_score": row["separation_score"],
                        "mean_variance_profile_corr_with_group": mean_variance_corr,
                        "mean_loading_profile_corr_with_group": mean_loading_corr,
                        "mean_combined_distance_with_group": mean_distance,
                        "true_redundant_group": true_redundant_group,
                    }
                )

    pair_df = pd.DataFrame.from_records(pair_rows)
    group_df = pd.DataFrame.from_records(group_rows)
    representative_df = pd.DataFrame.from_records(representative_rows)
    pair_df.to_csv(audit_dir / "within_family_redundancy_pairs.csv", index=False)
    group_df.to_csv(audit_dir / "within_family_redundancy_groups.csv", index=False)
    representative_df.to_csv(audit_dir / "within_family_redundancy_representatives.csv", index=False)

    true_pair_count = int(pair_df["true_redundant_candidate"].sum()) if not pair_df.empty else 0
    true_group_count = int(group_df["true_redundant_group"].fillna(False).astype(bool).sum()) if not group_df.empty else 0
    summary_lines = [
        "# Within-Family Redundancy Audit",
        "",
        "This audit only flags within-family redundancy candidates when variance behavior, PCA loading behavior,",
        "and hierarchical grouping all agree strongly enough to suggest true redundancy.",
        "",
        f"- feature_families_evaluated: `{merged['feature_category'].nunique()}`",
        f"- pair_candidates_flagged: `{true_pair_count}`",
        f"- group_members_flagged: `{true_group_count}`",
        "",
        "## Outputs",
        "",
        "- `within_family_redundancy_pairs.csv`",
        "- `within_family_redundancy_groups.csv`",
        "- `within_family_redundancy_representatives.csv`",
    ]
    (audit_dir / "within_family_redundancy_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def _compute_channel_display_ranges(report_df: pd.DataFrame, channels: list[str]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    shard_cache: dict[str, dict[str, np.ndarray]] = {}

    if "shard_path" in report_df.columns and report_df["shard_path"].notna().any():
        shard_paths = sorted({str(path) for path in report_df["shard_path"].dropna().unique()})
        for channel in channels:
            key = f"channel_{channel}"
            channel_min = np.inf
            channel_max = -np.inf
            for shard_path in shard_paths:
                if shard_path not in shard_cache:
                    shard_cache[shard_path] = load_patch_npz(shard_path)
                payload = shard_cache[shard_path]
                if key not in payload:
                    continue
                arr = np.asarray(payload[key], dtype=np.float32)
                if arr.ndim == 4:
                    proj = np.max(arr, axis=1)
                elif arr.ndim == 3:
                    proj = arr
                else:
                    continue
                channel_min = min(channel_min, float(np.min(proj)))
                channel_max = max(channel_max, float(np.max(proj)))
            if not np.isfinite(channel_min) or not np.isfinite(channel_max) or channel_max <= channel_min:
                channel_min, channel_max = 0.0, 1.0
            ranges[channel] = (channel_min, channel_max)
        return ranges

    for channel in channels:
        key = f"channel_{channel}"
        channel_min = np.inf
        channel_max = -np.inf
        for row in report_df.itertuples(index=False):
            payload = load_patch_payload(row)
            if key not in payload:
                continue
            proj = np.max(np.asarray(payload[key], dtype=np.float32), axis=0)
            channel_min = min(channel_min, float(np.min(proj)))
            channel_max = max(channel_max, float(np.max(proj)))
        if not np.isfinite(channel_min) or not np.isfinite(channel_max) or channel_max <= channel_min:
            channel_min, channel_max = 0.0, 1.0
        ranges[channel] = (channel_min, channel_max)
    return ranges


def _save_representative_gallery(
    output_dir: Path,
    report_df: pd.DataFrame,
    representatives: dict[int, list[int]],
    channel_display_ranges: dict[str, tuple[float, float]],
    channels: list[str],
    *,
    filename_prefix: str = "cluster",
    title_prefix: str = "Cluster",
) -> None:
    gallery_dir = output_dir / "representatives"
    gallery_dir.mkdir(parents=True, exist_ok=True)
    shard_cache: dict[str, dict[str, np.ndarray]] = {}
    if not channels:
        return

    for cluster_id, indices in representatives.items():
        fig, axes = plt.subplots(len(indices), len(channels), figsize=(3 * len(channels), max(3, len(indices) * 2.5)))
        if len(indices) == 1:
            axes = np.expand_dims(axes, axis=0)
        if len(channels) == 1:
            axes = np.asarray(axes).reshape(len(indices), 1)

        for row_idx, report_index in enumerate(indices):
            row = report_df.iloc[report_index]
            shard_path = row.get("shard_path")
            shard_index = row.get("shard_index")
            if pd.notna(shard_path):
                shard_key = str(shard_path)
                if shard_key not in shard_cache:
                    shard_cache[shard_key] = load_patch_npz(shard_key)
                payload = slice_patch_payload(shard_cache[shard_key], int(shard_index))
            else:
                payload = load_patch_payload(row)
            for col_idx, channel in enumerate(channels):
                ax = axes[row_idx, col_idx]
                key = f"channel_{channel}"
                if key in payload:
                    vmin, vmax = channel_display_ranges.get(channel, (0.0, 1.0))
                    ax.imshow(np.max(payload[key], axis=0), cmap="gray", vmin=vmin, vmax=vmax)
                ax.set_title(channel)
                ax.axis("off")

        fig.suptitle(f"{title_prefix} {cluster_id} representatives", fontsize=12)
        fig.tight_layout()
        fig.savefig(gallery_dir / f"{filename_prefix}_{cluster_id:02d}.png", dpi=150)
        plt.close(fig)


def _load_row_payload(
    row: pd.Series,
    shard_cache: dict[str, dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    shard_path = row.get("shard_path")
    shard_index = row.get("shard_index")
    if pd.notna(shard_path):
        shard_key = str(shard_path)
        if shard_key not in shard_cache:
            shard_cache[shard_key] = load_patch_npz(shard_key)
        return slice_patch_payload(shard_cache[shard_key], int(shard_index))
    return load_patch_payload(row)


def _save_cluster_summary_gallery(
    output_dir: Path,
    report_df: pd.DataFrame,
    channel_display_ranges: dict[str, tuple[float, float]],
    channels: list[str],
) -> None:
    summary_dir = output_dir / "cluster_summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    shard_cache: dict[str, dict[str, np.ndarray]] = {}

    for cluster_id in sorted(report_df["cluster_id"].unique()):
        cluster_rows = report_df[report_df["cluster_id"] == cluster_id]
        if cluster_rows.empty:
            continue

        channel_projections: dict[str, list[np.ndarray]] = {channel: [] for channel in channels}
        for _, row in cluster_rows.iterrows():
            payload = _load_row_payload(row, shard_cache)
            for channel in channels:
                key = f"channel_{channel}"
                if key in payload:
                    channel_projections[channel].append(np.max(np.asarray(payload[key], dtype=np.float32), axis=0))

        fig, axes = plt.subplots(2, len(channels), figsize=(3 * len(channels), 6))
        if len(channels) == 1:
            axes = np.asarray(axes).reshape(2, 1)
        for col_idx, channel in enumerate(channels):
            projections = channel_projections[channel]
            ax_avg = axes[0, col_idx]
            ax_med = axes[1, col_idx]
            vmin, vmax = channel_display_ranges.get(channel, (0.0, 1.0))
            if projections:
                stack = np.stack(projections, axis=0)
                avg_img = np.mean(stack, axis=0)
                med_img = np.median(stack, axis=0)
                ax_avg.imshow(avg_img, cmap="gray", vmin=vmin, vmax=vmax)
                ax_med.imshow(med_img, cmap="gray", vmin=vmin, vmax=vmax)
            ax_avg.set_title(f"{channel} mean")
            ax_med.set_title(f"{channel} median")
            ax_avg.axis("off")
            ax_med.axis("off")

        fig.suptitle(f"Cluster {cluster_id} summary projections", fontsize=12)
        fig.tight_layout()
        fig.savefig(summary_dir / f"cluster_{cluster_id:02d}_summary.png", dpi=150)
        plt.close(fig)


def run_latent_report(config: LatentReportConfig) -> Path:
    overall_start = perf_counter()
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[latent] output_dir={output_dir}")

    stage_start = perf_counter()
    manifest = pd.read_csv(config.manifest_path)
    print(f"[latent] manifest loaded | rows={len(manifest)} | elapsed={perf_counter()-stage_start:.1f}s")
    stage_start = perf_counter()
    feature_df, feature_cache_summary = _load_or_compute_engineered_features(
        manifest=manifest,
        manifest_path=config.manifest_path,
        channels=config.features.channels,
        map2_feature_policy=config.features.map2_feature_policy,
    )
    print(
        f"[latent] features ready | rows={len(feature_df)} | cols={feature_df.shape[1]} "
        f"| cache={feature_cache_summary['cache_status']} | elapsed={perf_counter()-stage_start:.1f}s"
    )
    stage_start = perf_counter()
    report_df = manifest.merge(feature_df, on="patch_id", how="inner")
    print(f"[latent] merged feature table | rows={len(report_df)} | elapsed={perf_counter()-stage_start:.1f}s")
    effective_channels = _effective_engineered_feature_channels(
        channels=config.features.channels,
        map2_feature_policy=config.features.map2_feature_policy,
    )

    feature_columns = [column for column in feature_df.columns if column != "patch_id"]
    if config.features.feature_variance_csv is not None:
        stage_start = perf_counter()
        variance_df = pd.read_csv(config.features.feature_variance_csv)
        if "feature" not in variance_df.columns or "feature_variance_cluster" not in variance_df.columns:
            raise ValueError(
                "Feature variance CSV must contain 'feature' and 'feature_variance_cluster' columns"
            )
        selected_features = variance_df.loc[
            variance_df["feature_variance_cluster"] == config.features.feature_variance_cluster, "feature"
        ].astype(str).tolist()
        feature_columns = [column for column in feature_columns if column in set(selected_features)]
        if not feature_columns:
            raise ValueError(
                f"No engineered features matched feature_variance_cluster={config.features.feature_variance_cluster} "
                f"from {config.features.feature_variance_csv}"
            )
        print(
            f"[latent] feature variance filter | cluster={config.features.feature_variance_cluster} "
            f"| selected={len(feature_columns)} | elapsed={perf_counter()-stage_start:.1f}s"
        )

    stage_start = perf_counter()
    feature_columns, channel_scope_summary, channel_scope_audit_df = _apply_channel_scope_filter(
        feature_columns=feature_columns,
        allowed_channels=effective_channels,
        output_dir=output_dir,
    )
    if not feature_columns:
        raise ValueError(
            "All candidate features were excluded by the channel scope filter before the MAP2 feature policy."
        )
    print(
        f"[latent] channel scope applied | allowed={','.join([str(v) for v in effective_channels if str(v) != 'MAP2'])} "
        f"| kept={len(feature_columns)} | elapsed={perf_counter()-stage_start:.1f}s"
    )

    stage_start = perf_counter()
    feature_columns, map2_policy_summary, map2_policy_audit_df = _apply_map2_feature_policy(
        feature_columns=feature_columns,
        policy=config.features.map2_feature_policy,
        output_dir=output_dir,
    )
    if not feature_columns:
        raise ValueError(
            "All candidate features were excluded by the MAP2 feature policy before the pre-clustering audit."
        )
    print(f"[latent] MAP2 policy applied | kept={len(feature_columns)} | elapsed={perf_counter()-stage_start:.1f}s")

    stage_start = perf_counter()
    feature_columns, audit_summary, audit_df = _audit_and_filter_feature_columns(
        report_df=report_df,
        feature_columns=feature_columns,
        output_dir=output_dir,
    )
    if not feature_columns:
        raise ValueError(
            "All candidate features were dropped by the pre-clustering audit as zero, constant, or near-constant."
        )
    print(f"[latent] feature audit complete | kept={len(feature_columns)} | elapsed={perf_counter()-stage_start:.1f}s")

    stage_start = perf_counter()
    feature_matrix = report_df[feature_columns].to_numpy(dtype=np.float64)
    scaled = scale_feature_matrix(feature_matrix, config.preprocessing.scaler)
    print(
        f"[latent] scaling complete | scaler={config.preprocessing.scaler} "
        f"| matrix_shape={scaled.shape} | elapsed={perf_counter()-stage_start:.1f}s"
    )

    stage_start = perf_counter()
    n_components = min(config.dimensionality_reduction.n_pca_components, scaled.shape[1], max(2, scaled.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=config.clustering.random_seed)
    embedding = pca.fit_transform(scaled)
    print(f"[latent] PCA complete | components={n_components} | elapsed={perf_counter()-stage_start:.1f}s")

    stage_start = perf_counter()
    labels, centers = _fit_clusters(
        embedding=embedding,
        method=config.clustering.method,
        n_clusters=config.clustering.n_clusters,
        random_seed=config.clustering.random_seed,
    )
    print(f"[latent] clustering complete | unique_clusters={len(np.unique(labels))} | elapsed={perf_counter()-stage_start:.1f}s")
    representatives = _select_representatives(
        embedding=embedding,
        labels=labels,
        centers=centers,
        k=config.reporting.representatives_per_cluster,
    )
    extreme_representatives = _select_extremes(
        embedding=embedding,
        labels=labels,
        centers=centers,
        k=min(5, config.reporting.representatives_per_cluster),
    )
    central_representatives = {
        cluster_id: indices[: min(5, len(indices))]
        for cluster_id, indices in representatives.items()
    }
    print(f"[latent] representative selection complete | elapsed={perf_counter()-stage_start:.1f}s")

    stage_start = perf_counter()
    report_df["cluster_id"] = labels
    report_df["pca_1"] = embedding[:, 0]
    report_df["pca_2"] = embedding[:, 1] if embedding.shape[1] > 1 else 0.0

    cluster_summary = (
        report_df.groupby(["cluster_id", "condition", "patch_group"], dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values(["cluster_id", "count"], ascending=[True, False])
    )
    numeric_means = report_df.groupby("cluster_id")[feature_columns].mean().reset_index()
    removed_feature_columns = audit_df.loc[~audit_df["kept"], "feature"].astype(str).tolist()
    separation_df = _compute_cluster_separation_scores(report_df=report_df, feature_columns=feature_columns)
    removed_separation_df = _compute_cluster_separation_scores(report_df=report_df, feature_columns=removed_feature_columns)
    print(f"[latent] summary metrics computed | elapsed={perf_counter()-stage_start:.1f}s")
    stage_start = perf_counter()
    report_df.to_csv(output_dir / "patch_latent_report.csv", index=False)
    cluster_summary.to_csv(output_dir / "cluster_condition_patchgroup_counts.csv", index=False)
    numeric_means.to_csv(output_dir / "cluster_feature_means.csv", index=False)
    pd.DataFrame({"feature": feature_columns}).to_csv(output_dir / "selected_feature_columns.csv", index=False)
    map2_policy_audit_df.loc[~map2_policy_audit_df["kept"], ["feature", "policy_reason"]].to_csv(
        output_dir / "excluded_map2_policy_features.csv",
        index=False,
    )
    overview_summary = {**feature_cache_summary, **map2_policy_summary, **audit_summary}
    pd.DataFrame([overview_summary]).to_csv(output_dir / "feature_variance_audit_overview.csv", index=False)
    print(f"[latent] core tables written | elapsed={perf_counter()-stage_start:.1f}s")
    stage_start = perf_counter()
    _save_pca_variance_plots(output_dir=output_dir, pca=pca)
    loading_df = _save_pca_loadings(output_dir=output_dir, pca=pca, feature_columns=feature_columns)
    print(f"[latent] PCA outputs written | elapsed={perf_counter()-stage_start:.1f}s")
    stage_start = perf_counter()
    feature_variance_dir = output_dir / "feature_variance_analysis"
    feature_variance_dir.mkdir(parents=True, exist_ok=True)
    _save_cluster_separation_outputs(output_dir=feature_variance_dir, separation_df=separation_df)
    _save_within_family_redundancy_audit(
        output_dir=feature_variance_dir,
        separation_df=separation_df,
        loading_df=loading_df,
    )
    if not removed_separation_df.empty:
        removed_separation_df.to_csv(feature_variance_dir / "removed_feature_cluster_separation.csv", index=False)
    run_feature_variance_analysis(
        output_dir=feature_variance_dir,
        separation_df=separation_df,
        removed_separation_df=removed_separation_df,
    )
    print(f"[latent] feature variance analysis complete | elapsed={perf_counter()-stage_start:.1f}s")
    stage_start = perf_counter()
    display_channels = _display_channels(
        channels=config.features.channels,
        map2_feature_policy=config.features.map2_feature_policy,
    )
    channel_display_ranges = _compute_channel_display_ranges(report_df=report_df, channels=display_channels)
    pd.DataFrame.from_records(
        [
            {"channel": channel, "display_min": bounds[0], "display_max": bounds[1]}
            for channel, bounds in channel_display_ranges.items()
        ]
    ).to_csv(output_dir / "channel_display_ranges.csv", index=False)
    print(f"[latent] channel display ranges written | elapsed={perf_counter()-stage_start:.1f}s")

    stage_start = perf_counter()
    representatives_rows: list[dict[str, object]] = []
    for cluster_id, indices in representatives.items():
        for rank, report_index in enumerate(indices, start=1):
            row = report_df.iloc[report_index]
            representatives_rows.append(
                {
                    "cluster_id": cluster_id,
                    "representative_set": "central_ranked",
                    "rank": rank,
                    "patch_id": row["patch_id"],
                    "patch_path": row["patch_path"],
                    "shard_path": row.get("shard_path"),
                    "shard_index": row.get("shard_index"),
                    "condition": row["condition"],
                    "patch_group": row["patch_group"],
                }
            )
    for cluster_id, indices in extreme_representatives.items():
        for rank, report_index in enumerate(indices, start=1):
            row = report_df.iloc[report_index]
            representatives_rows.append(
                {
                    "cluster_id": cluster_id,
                    "representative_set": "extreme",
                    "rank": rank,
                    "patch_id": row["patch_id"],
                    "patch_path": row["patch_path"],
                    "shard_path": row.get("shard_path"),
                    "shard_index": row.get("shard_index"),
                    "condition": row["condition"],
                    "patch_group": row["patch_group"],
                }
            )
    pd.DataFrame.from_records(representatives_rows).to_csv(output_dir / "cluster_representatives.csv", index=False)
    print(f"[latent] representative table written | elapsed={perf_counter()-stage_start:.1f}s")

    stage_start = perf_counter()
    _save_pca_plot(output_dir=output_dir, report_df=report_df)
    _save_latent_structure_interpretation(output_dir=output_dir, report_df=report_df)
    _save_representative_gallery(
        output_dir=output_dir,
        report_df=report_df,
        representatives=central_representatives,
        channel_display_ranges=channel_display_ranges,
        channels=display_channels,
        filename_prefix="cluster_central",
        title_prefix="Cluster",
    )
    _save_representative_gallery(
        output_dir=output_dir,
        report_df=report_df,
        representatives=extreme_representatives,
        channel_display_ranges=channel_display_ranges,
        channels=display_channels,
        filename_prefix="cluster_extreme",
        title_prefix="Cluster",
    )
    print(f"[latent] plots and interpretation outputs written | elapsed={perf_counter()-stage_start:.1f}s")
    print(f"[latent] done | total_elapsed={perf_counter()-overall_start:.1f}s")

    return output_dir
