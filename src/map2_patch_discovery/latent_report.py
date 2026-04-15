from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .features import extract_engineered_features, load_patch_npz
from .report_config import LatentReportConfig


def _fit_clusters(embedding: np.ndarray, method: str, n_clusters: int, random_seed: int) -> tuple[np.ndarray, np.ndarray]:
    if method == "gmm":
        model = GaussianMixture(n_components=n_clusters, random_state=random_seed)
        labels = model.fit_predict(embedding)
        centers = model.means_
        return labels.astype(int), centers
    model = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    labels = model.fit_predict(embedding)
    return labels.astype(int), model.cluster_centers_


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


def _save_representative_gallery(output_dir: Path, report_df: pd.DataFrame, representatives: dict[int, list[int]]) -> None:
    gallery_dir = output_dir / "representatives"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    for cluster_id, indices in representatives.items():
        fig, axes = plt.subplots(len(indices), 4, figsize=(12, max(3, len(indices) * 2.5)))
        if len(indices) == 1:
            axes = np.expand_dims(axes, axis=0)

        for row_idx, report_index in enumerate(indices):
            payload = load_patch_npz(report_df.iloc[report_index]["patch_path"])
            for col_idx, channel in enumerate(["MAP2", "FLAG", "HA", "SHANK2"]):
                ax = axes[row_idx, col_idx]
                key = f"channel_{channel}"
                if key in payload:
                    ax.imshow(np.max(payload[key], axis=0), cmap="gray")
                ax.set_title(channel)
                ax.axis("off")

        fig.suptitle(f"Cluster {cluster_id} representatives", fontsize=12)
        fig.tight_layout()
        fig.savefig(gallery_dir / f"cluster_{cluster_id:02d}.png", dpi=150)
        plt.close(fig)


def run_latent_report(config: LatentReportConfig) -> Path:
    output_dir = config.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(config.manifest_path)
    feature_df = extract_engineered_features(manifest=manifest, channels=config.features.channels)
    report_df = manifest.merge(feature_df, on="patch_id", how="inner")

    feature_columns = [column for column in feature_df.columns if column != "patch_id"]
    feature_matrix = report_df[feature_columns].to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(feature_matrix)

    n_components = min(config.dimensionality_reduction.n_pca_components, scaled.shape[1], max(2, scaled.shape[0] - 1))
    pca = PCA(n_components=n_components, random_state=config.clustering.random_seed)
    embedding = pca.fit_transform(scaled)

    labels, centers = _fit_clusters(
        embedding=embedding,
        method=config.clustering.method,
        n_clusters=config.clustering.n_clusters,
        random_seed=config.clustering.random_seed,
    )
    representatives = _select_representatives(
        embedding=embedding,
        labels=labels,
        centers=centers,
        k=config.reporting.representatives_per_cluster,
    )

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
    report_df.to_csv(output_dir / "patch_latent_report.csv", index=False)
    cluster_summary.to_csv(output_dir / "cluster_condition_patchgroup_counts.csv", index=False)
    numeric_means.to_csv(output_dir / "cluster_feature_means.csv", index=False)

    representatives_rows: list[dict[str, object]] = []
    for cluster_id, indices in representatives.items():
        for rank, report_index in enumerate(indices, start=1):
            row = report_df.iloc[report_index]
            representatives_rows.append(
                {
                    "cluster_id": cluster_id,
                    "rank": rank,
                    "patch_id": row["patch_id"],
                    "patch_path": row["patch_path"],
                    "condition": row["condition"],
                    "patch_group": row["patch_group"],
                }
            )
    pd.DataFrame.from_records(representatives_rows).to_csv(output_dir / "cluster_representatives.csv", index=False)

    _save_pca_plot(output_dir=output_dir, report_df=report_df)
    _save_representative_gallery(output_dir=output_dir, report_df=report_df, representatives=representatives)

    return output_dir
