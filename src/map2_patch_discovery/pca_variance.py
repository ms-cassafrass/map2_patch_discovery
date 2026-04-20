from __future__ import annotations

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from .latent_report import (
    _apply_map2_feature_policy,
    _audit_and_filter_feature_columns,
    _load_or_compute_engineered_features,
)
from .preprocessing import scale_feature_matrix
from .report_config import LatentReportConfig


VARIANCE_THRESHOLDS = (0.90, 0.95, 0.99)


def _dimensions_for_thresholds(cumulative: np.ndarray, thresholds: tuple[float, ...]) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        if cumulative.size == 0:
            dims = 0
            achieved = 0.0
        else:
            index = int(np.searchsorted(cumulative, threshold, side="left"))
            if index >= cumulative.size:
                dims = int(cumulative.size)
                achieved = float(cumulative[-1])
            else:
                dims = index + 1
                achieved = float(cumulative[index])
        rows.append(
            {
                "target_explained_variance_ratio": float(threshold),
                "minimum_dimensions": int(dims),
                "achieved_cumulative_explained_variance_ratio": float(achieved),
            }
        )
    return rows


def _save_variance_curve(output_dir: Path, curve_df: pd.DataFrame, threshold_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        curve_df["component"],
        curve_df["cumulative_explained_variance_ratio"],
        marker="o",
        color="#4C78A8",
    )
    for row in threshold_df.itertuples(index=False):
        threshold = float(row.target_explained_variance_ratio)
        dims = int(row.minimum_dimensions)
        ax.axhline(threshold, color="#E45756", linestyle="--", linewidth=1)
        if dims > 0:
            ax.axvline(dims, color="#E45756", linestyle="--", linewidth=1)
            ax.scatter(
                [dims],
                [float(row.achieved_cumulative_explained_variance_ratio)],
                color="#E45756",
                s=30,
                zorder=3,
            )
            ax.text(
                dims,
                0.02,
                f"{dims}",
                color="#E45756",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_title("Cumulative Explained Variance")
    ax.set_ylim(0.0, 1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "pca_cumulative_variance_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        curve_df["component"],
        curve_df["explained_variance_ratio"],
        marker="o",
        color="#54A24B",
    )
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Per-Component Explained Variance")
    fig.tight_layout()
    fig.savefig(output_dir / "pca_explained_variance_curve.png", dpi=150)
    plt.close(fig)


def run_pca_variance_analysis(
    config: LatentReportConfig,
    *,
    output_dir: str | Path | None = None,
) -> Path:
    overall_start = perf_counter()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (config.output_dir.resolve() / "pca_variance_analysis")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[pca-variance] output_dir={output_dir}")

    stage_start = perf_counter()
    manifest = pd.read_csv(config.manifest_path)
    print(f"[pca-variance] manifest loaded | rows={len(manifest)} | elapsed={perf_counter()-stage_start:.1f}s")

    stage_start = perf_counter()
    feature_df, feature_cache_summary = _load_or_compute_engineered_features(
        manifest=manifest,
        manifest_path=config.manifest_path,
        channels=config.features.channels,
    )
    print(
        f"[pca-variance] features ready | rows={len(feature_df)} | cols={feature_df.shape[1]} "
        f"| cache={feature_cache_summary['cache_status']} | elapsed={perf_counter()-stage_start:.1f}s"
    )

    stage_start = perf_counter()
    report_df = manifest.merge(feature_df, on="patch_id", how="inner")
    print(f"[pca-variance] merged feature table | rows={len(report_df)} | elapsed={perf_counter()-stage_start:.1f}s")

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
        print(
            f"[pca-variance] feature variance filter | cluster={config.features.feature_variance_cluster} "
            f"| selected={len(feature_columns)}"
        )

    feature_columns, map2_policy_summary, _ = _apply_map2_feature_policy(
        feature_columns=feature_columns,
        policy=config.features.map2_feature_policy,
        output_dir=output_dir,
    )
    if not feature_columns:
        raise ValueError("All candidate features were excluded by the MAP2 feature policy.")

    feature_columns, audit_summary, audit_df = _audit_and_filter_feature_columns(
        report_df=report_df,
        feature_columns=feature_columns,
        output_dir=output_dir,
    )
    if not feature_columns:
        raise ValueError("All candidate features were dropped by the variance audit.")
    print(f"[pca-variance] retained features={len(feature_columns)}")

    stage_start = perf_counter()
    feature_matrix = report_df[feature_columns].to_numpy(dtype=np.float64)
    scaled = scale_feature_matrix(feature_matrix, config.preprocessing.scaler)
    max_components = min(scaled.shape[0], scaled.shape[1])
    pca = PCA(n_components=max_components, random_state=config.clustering.random_seed)
    pca.fit(scaled)
    explained = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
    cumulative = np.cumsum(explained)
    print(
        f"[pca-variance] PCA fit complete | components={max_components} "
        f"| elapsed={perf_counter()-stage_start:.1f}s"
    )

    curve_df = pd.DataFrame(
        {
            "component": np.arange(1, explained.size + 1, dtype=np.int32),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance_ratio": cumulative,
        }
    )
    threshold_df = pd.DataFrame.from_records(_dimensions_for_thresholds(cumulative, VARIANCE_THRESHOLDS))
    selected_df = pd.DataFrame({"feature": feature_columns})
    dropped_df = audit_df.loc[~audit_df["kept"], :].copy()
    overview_df = pd.DataFrame(
        [
            {
                **feature_cache_summary,
                **map2_policy_summary,
                **audit_summary,
                "scaler": config.preprocessing.scaler,
                "manifest_rows": int(len(manifest)),
                "retained_feature_count": int(len(feature_columns)),
                "full_pca_component_count": int(max_components),
                "max_cumulative_explained_variance_ratio": float(cumulative[-1]) if cumulative.size else 0.0,
            }
        ]
    )

    curve_df.to_csv(output_dir / "pca_variance_curve.csv", index=False)
    threshold_df.to_csv(output_dir / "pca_variance_thresholds.csv", index=False)
    selected_df.to_csv(output_dir / "selected_feature_columns.csv", index=False)
    dropped_df.to_csv(output_dir / "dropped_feature_columns.csv", index=False)
    overview_df.to_csv(output_dir / "pca_variance_overview.csv", index=False)
    _save_variance_curve(output_dir=output_dir, curve_df=curve_df, threshold_df=threshold_df)

    summary_lines = [
        "# PCA Variance Threshold Summary",
        "",
        f"- manifest_rows: `{len(manifest)}`",
        f"- retained_feature_count: `{len(feature_columns)}`",
        f"- scaler: `{config.preprocessing.scaler}`",
        f"- full_pca_component_count: `{max_components}`",
        "",
        "## Minimum Dimensions",
        "",
    ]
    for row in threshold_df.itertuples(index=False):
        pct = int(round(float(row.target_explained_variance_ratio) * 100))
        summary_lines.append(
            f"- `{pct}%` variance: `{int(row.minimum_dimensions)}` dimensions "
            f"(achieved `{float(row.achieved_cumulative_explained_variance_ratio):.4f}`)"
        )
    (output_dir / "pca_variance_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("[pca-variance] thresholds:")
    for row in threshold_df.itertuples(index=False):
        pct = int(round(float(row.target_explained_variance_ratio) * 100))
        print(
            f"  {pct}% -> dims={int(row.minimum_dimensions)} "
            f"| achieved={float(row.achieved_cumulative_explained_variance_ratio):.4f}"
        )
    print(f"[pca-variance] done | elapsed={perf_counter()-overall_start:.1f}s")
    return output_dir
