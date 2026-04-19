from __future__ import annotations

import os
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
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


REQUIRED_COLUMNS = ("feature", "between_cluster_var", "within_cluster_var", "separation_score")


def _write_interpretation_notes(output_dir: Path) -> None:
    lines = [
        "HOW TO READ THE FEATURE VARIANCE PLOTS",
        "",
        "General idea:",
        "Each point in these plots is a FEATURE, not a patch.",
        "These figures describe how each feature behaves across the patch clusters from the latent run.",
        "",
        "Key metrics:",
        "between_cluster_var = how much the feature differs between cluster means",
        "within_cluster_var = how much the feature still varies inside clusters",
        "separation_score = between / within, so larger means better cluster discrimination",
        "",
        "log versions:",
        "The plots often use log10(...) because the raw variance values span very large numeric ranges.",
        "The log scale makes the relative structure easier to see.",
        "",
        "feature_variance_metric_correlation_heatmap.png:",
        "Shows the Pearson correlation among log_between_cluster_var, log_within_cluster_var, and log_separation_score.",
        "Use this to see whether these metrics are moving together or in opposition overall.",
        "",
        "feature_variance_metric_pairplots.png:",
        "Shows pairwise scatter plots for the variance metrics.",
        "These let you see whether a relationship is tight, nonlinear, diffuse, or split into subgroups.",
        "If color is present, it is showing feature_variance_cluster assignment.",
        "",
        "feature_variance_landscape.png:",
        "x-axis = log10(within_cluster_var)",
        "y-axis = log10(between_cluster_var)",
        "This is a feature-behavior map.",
        "",
        "Interpretation of the landscape:",
        "upper-left = usually strongest discriminative features",
        "lower within + higher between means stable inside clusters and distinct across clusters",
        "",
        "upper-right = separating but noisy features",
        "They differ across clusters but also vary a lot inside clusters.",
        "",
        "lower-left = quiet or weak features",
        "They do not vary much, but they also do not separate clusters much.",
        "",
        "lower-right = usually weak/noisy features",
        "Higher within + lower between means lots of within-cluster spread without much cluster separation.",
        "",
        "feature_variance_cluster_plot.png:",
        "PCA of the retained features using the variance metrics only.",
        "This groups features by similarity in between/within/separation behavior.",
        "It is a clustering of FEATURES, not of patches.",
        "",
        "feature_variance_removed_feature_projection.png:",
        "Shows removed features projected into the same PCA space as the retained-feature variance PCA.",
        "This helps compare where dropped features would have landed relative to retained ones.",
        "",
        "feature_variance_hierarchical_dendrogram.png:",
        "Hierarchical clustering of features using their variance behavior.",
        "Shorter branch distances mean more similar feature behavior.",
        "This is useful for spotting families or neighborhoods of potentially redundant features.",
        "",
        "Silhouette and elbow plots:",
        "These help choose how many feature-variance clusters are reasonable.",
        "Silhouette favors cleaner separation.",
        "Elbow / inertia looks for diminishing returns as k increases.",
        "",
        "Important caution:",
        "These plots do NOT directly tell you biology.",
        "They tell you how useful, stable, noisy, or redundant a feature appears with respect to the current patch clustering.",
    ]
    (output_dir / "HOW_TO_READ_FEATURE_VARIANCE_PLOTS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _save_variance_correlation_outputs(
    *,
    output_dir: Path,
    work: pd.DataFrame,
) -> None:
    metric_columns = [
        "log_between_cluster_var",
        "log_within_cluster_var",
        "log_separation_score",
    ]
    corr_df = work[metric_columns].corr(method="pearson")
    corr_df.to_csv(output_dir / "feature_variance_metric_correlations.csv")

    fig, ax = plt.subplots(figsize=(6, 5))
    corr_values = corr_df.to_numpy(dtype=np.float32)
    im = ax.imshow(corr_values, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(metric_columns)))
    ax.set_yticks(np.arange(len(metric_columns)))
    pretty_labels = ["log between", "log within", "log separation"]
    ax.set_xticklabels(pretty_labels, rotation=30, ha="right")
    ax.set_yticklabels(pretty_labels)
    for i in range(corr_values.shape[0]):
        for j in range(corr_values.shape[1]):
            ax.text(j, i, f"{corr_values[i, j]:.2f}", ha="center", va="center", fontsize=9, color="black")
    ax.set_title("Feature Variance Metric Correlations")
    fig.colorbar(im, ax=ax, label="Pearson r")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_variance_metric_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    pairs = [
        ("log_between_cluster_var", "log_within_cluster_var", "Between vs Within"),
        ("log_between_cluster_var", "log_separation_score", "Between vs Separation"),
        ("log_within_cluster_var", "log_separation_score", "Within vs Separation"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    if "feature_variance_cluster" in work.columns:
        color_values = work["feature_variance_cluster"].to_numpy(dtype=np.int32)
        cmap = "tab10"
    else:
        color_values = "#4C78A8"
        cmap = None
    for ax, (x_col, y_col, title) in zip(axes, pairs):
        scatter = ax.scatter(
            work[x_col],
            work[y_col],
            c=color_values,
            cmap=cmap,
            s=28,
            alpha=0.8,
        )
        ax.set_xlabel(x_col.replace("_", " "))
        ax.set_ylabel(y_col.replace("_", " "))
        ax.set_title(title)
    if cmap is not None:
        fig.colorbar(scatter, ax=axes.ravel().tolist(), label="feature_variance_cluster")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_variance_metric_pairplots.png", dpi=150)
    plt.close(fig)


def _save_k_selection_outputs(
    *,
    output_dir: Path,
    scaled: np.ndarray,
    random_seed: int,
) -> int:
    n_features = scaled.shape[0]
    max_k = min(8, n_features - 1)
    if max_k < 2:
        fallback = pd.DataFrame.from_records(
            [{"k": 1, "silhouette_score": np.nan, "inertia": 0.0, "selected_by_silhouette": True}]
        )
        fallback.to_csv(output_dir / "feature_variance_k_selection.csv", index=False)
        return 1

    rows: list[dict[str, float | int | bool]] = []
    best_k = 2
    best_silhouette = -np.inf
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, random_state=random_seed, n_init=10)
        labels = model.fit_predict(scaled).astype(int)
        silhouette = float(silhouette_score(scaled, labels)) if len(np.unique(labels)) > 1 else np.nan
        inertia = float(model.inertia_)
        rows.append(
            {
                "k": k,
                "silhouette_score": silhouette,
                "inertia": inertia,
                "selected_by_silhouette": False,
            }
        )
        if np.isfinite(silhouette) and silhouette > best_silhouette:
            best_silhouette = silhouette
            best_k = k

    selection_df = pd.DataFrame.from_records(rows)
    selection_df.loc[selection_df["k"] == best_k, "selected_by_silhouette"] = True
    selection_df.to_csv(output_dir / "feature_variance_k_selection.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(selection_df["k"], selection_df["silhouette_score"], marker="o", color="#4C78A8")
    ax.axvline(best_k, color="#E45756", linestyle="--", label=f"best k = {best_k}")
    ax.set_xlabel("k")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette-Based k Selection")
    ax.set_xticks(selection_df["k"].tolist())
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "feature_variance_silhouette_selection.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(selection_df["k"], selection_df["inertia"], marker="o", color="#72B7B2")
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow / Inertia Analysis")
    ax.set_xticks(selection_df["k"].tolist())
    fig.tight_layout()
    fig.savefig(output_dir / "feature_variance_elbow_inertia.png", dpi=150)
    plt.close(fig)

    return best_k


def _save_hierarchical_outputs(
    *,
    output_dir: Path,
    work: pd.DataFrame,
    scaled: np.ndarray,
    n_clusters: int,
) -> None:
    linkage_matrix = linkage(scaled, method="ward")

    n_labels = len(work)
    fig_h = max(8, min(40, 0.22 * n_labels))
    fig_w = 14
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    dendrogram(
        linkage_matrix,
        labels=work["feature"].astype(str).to_list(),
        orientation="right",
        leaf_rotation=0,
        leaf_font_size=8 if n_labels <= 60 else 7 if n_labels <= 120 else 6,
        ax=ax,
    )
    ax.set_title("Hierarchical Clustering of Feature Variance Metrics")
    ax.set_xlabel("Ward Distance")
    ax.set_ylabel("Feature")
    fig.tight_layout(pad=1.2)
    fig.savefig(output_dir / "feature_variance_hierarchical_dendrogram.png", dpi=150)
    plt.close(fig)

    hierarchical_labels = fcluster(linkage_matrix, t=n_clusters, criterion="maxclust") - 1
    hierarchical_df = work.copy()
    hierarchical_df["hierarchical_cluster"] = hierarchical_labels.astype(int)
    hierarchical_df.to_csv(output_dir / "feature_variance_hierarchical_clusters.csv", index=False)

    summary = (
        hierarchical_df.groupby("hierarchical_cluster", dropna=False)
        .agg(
            feature_count=("feature", "size"),
            mean_between_cluster_var=("between_cluster_var", "mean"),
            mean_within_cluster_var=("within_cluster_var", "mean"),
            mean_separation_score=("separation_score", "mean"),
        )
        .reset_index()
        .sort_values("hierarchical_cluster")
    )
    summary.to_csv(output_dir / "feature_variance_hierarchical_summary.csv", index=False)


def load_feature_separation_table(path: str | Path) -> pd.DataFrame:
    path = Path(path).resolve()
    table = pd.read_csv(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ValueError(
            f"Feature separation table is missing required columns: {', '.join(missing)}"
        )
    return table


def run_feature_variance_analysis(
    *,
    separation_df: pd.DataFrame,
    output_dir: str | Path,
    removed_separation_df: pd.DataFrame | None = None,
    random_seed: int = 42,
    n_clusters: int | None = None,
) -> Path:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_interpretation_notes(output_dir)

    work = separation_df.copy()
    for column in ("between_cluster_var", "within_cluster_var", "separation_score"):
        work[f"log_{column}"] = np.log10(np.asarray(work[column], dtype=np.float64) + 1e-6)

    _save_variance_correlation_outputs(output_dir=output_dir, work=work)

    feature_matrix = work[
        [
            "log_between_cluster_var",
            "log_within_cluster_var",
            "log_separation_score",
        ]
    ].to_numpy(dtype=np.float32)

    n_features = feature_matrix.shape[0]
    if n_features == 0:
        return output_dir

    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)
    best_silhouette_k = _save_k_selection_outputs(
        output_dir=output_dir,
        scaled=scaled,
        random_seed=random_seed,
    )

    if n_clusters is None:
        n_clusters = best_silhouette_k
    else:
        n_clusters = max(1, min(int(n_clusters), n_features))

    if n_clusters < 2 or n_features < 2:
        work["feature_variance_cluster"] = 0
        work.to_csv(output_dir / "feature_variance_clusters.csv", index=False)
        return output_dir

    model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = model.fit_predict(scaled).astype(int)
    work["feature_variance_cluster"] = labels
    work.to_csv(output_dir / "feature_variance_clusters.csv", index=False)

    cluster_summary = (
        work.groupby("feature_variance_cluster", dropna=False)
        .agg(
            feature_count=("feature", "size"),
            mean_between_cluster_var=("between_cluster_var", "mean"),
            mean_within_cluster_var=("within_cluster_var", "mean"),
            mean_separation_score=("separation_score", "mean"),
        )
        .reset_index()
        .sort_values("feature_variance_cluster")
    )
    cluster_summary.to_csv(output_dir / "feature_variance_cluster_summary.csv", index=False)

    reduced = PCA(n_components=2, random_state=random_seed).fit_transform(scaled)
    work["feature_variance_pca_1"] = reduced[:, 0]
    work["feature_variance_pca_2"] = reduced[:, 1]
    work.to_csv(output_dir / "feature_variance_clusters_with_pca.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        work["feature_variance_pca_1"],
        work["feature_variance_pca_2"],
        c=work["feature_variance_cluster"],
        cmap="tab10",
        s=36,
        alpha=0.85,
    )
    ax.set_xlabel("Feature-Metric PCA 1")
    ax.set_ylabel("Feature-Metric PCA 2")
    ax.set_title("Feature Clusters from Between/Within Variance Metrics")
    fig.colorbar(scatter, ax=ax, label="feature_variance_cluster")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_variance_cluster_plot.png", dpi=150)
    plt.close(fig)

    if removed_separation_df is not None and not removed_separation_df.empty:
        removed_work = removed_separation_df.copy()
        for column in ("between_cluster_var", "within_cluster_var", "separation_score"):
            removed_work[f"log_{column}"] = np.log10(np.asarray(removed_work[column], dtype=np.float64) + 1e-6)
        removed_matrix = removed_work[
            [
                "log_between_cluster_var",
                "log_within_cluster_var",
                "log_separation_score",
            ]
        ].to_numpy(dtype=np.float32)
        removed_scaled = scaler.transform(removed_matrix)
        removed_reduced = PCA(n_components=2, random_state=random_seed).fit(scaled).transform(removed_scaled)
        removed_work["feature_variance_pca_1"] = removed_reduced[:, 0]
        removed_work["feature_variance_pca_2"] = removed_reduced[:, 1]
        removed_work.to_csv(output_dir / "removed_feature_variance_projection.csv", index=False)

        x_min = min(float(np.min(work["feature_variance_pca_1"])), float(np.min(removed_work["feature_variance_pca_1"])))
        x_max = max(float(np.max(work["feature_variance_pca_1"])), float(np.max(removed_work["feature_variance_pca_1"])))
        y_min = min(float(np.min(work["feature_variance_pca_2"])), float(np.min(removed_work["feature_variance_pca_2"])))
        y_max = max(float(np.max(work["feature_variance_pca_2"])), float(np.max(removed_work["feature_variance_pca_2"])))

        fig, ax = plt.subplots(figsize=(9, 7))
        ax.scatter(
            removed_work["feature_variance_pca_1"],
            removed_work["feature_variance_pca_2"],
            color="#9D9D9D",
            s=36,
            alpha=0.85,
        )
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("Feature-Metric PCA 1")
        ax.set_ylabel("Feature-Metric PCA 2")
        ax.set_title("Removed Features Projected into Retained Feature Variance PCA Space")
        fig.tight_layout()
        fig.savefig(output_dir / "feature_variance_removed_feature_projection.png", dpi=150)
        plt.close(fig)

    top_labeled = work.sort_values("separation_score", ascending=False).head(min(20, len(work)))
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        work["log_within_cluster_var"],
        work["log_between_cluster_var"],
        c=work["feature_variance_cluster"],
        cmap="tab10",
        s=36,
        alpha=0.8,
    )
    for row in top_labeled.itertuples(index=False):
        ax.annotate(
            row.feature,
            (row.log_within_cluster_var, row.log_between_cluster_var),
            fontsize=7,
            alpha=0.9,
        )
    ax.set_xlabel("log10(within_cluster_var)")
    ax.set_ylabel("log10(between_cluster_var)")
    ax.set_title("Feature Variance Landscape")
    fig.colorbar(scatter, ax=ax, label="feature_variance_cluster")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_variance_landscape.png", dpi=150)
    plt.close(fig)

    _save_hierarchical_outputs(
        output_dir=output_dir,
        work=work,
        scaled=scaled,
        n_clusters=n_clusters,
    )

    return output_dir
