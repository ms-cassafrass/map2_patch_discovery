from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FeatureReportConfig:
    channels: list[str]
    feature_variance_csv: Path | None = None
    feature_variance_cluster: int | None = None
    map2_feature_policy: str = "full"


@dataclass(frozen=True)
class ReductionConfig:
    n_pca_components: int


@dataclass(frozen=True)
class ClusteringConfig:
    method: str
    n_clusters: int
    random_seed: int


@dataclass(frozen=True)
class ReportingConfig:
    representatives_per_cluster: int


@dataclass(frozen=True)
class LatentReportConfig:
    manifest_path: Path
    output_dir: Path
    features: FeatureReportConfig
    dimensionality_reduction: ReductionConfig
    clustering: ClusteringConfig
    reporting: ReportingConfig


def _require_keys(data: dict[str, Any], keys: list[str], section: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise ValueError(f"Missing required keys in {section}: {missing}")


def load_latent_report_config(path: str | Path) -> LatentReportConfig:
    path = Path(path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError("Top-level latent report config must be a mapping")

    _require_keys(raw, ["manifest_path", "output_dir", "features", "dimensionality_reduction", "clustering", "reporting"], "root")
    _require_keys(raw["features"], ["channels"], "features")
    _require_keys(raw["dimensionality_reduction"], ["n_pca_components"], "dimensionality_reduction")
    _require_keys(raw["clustering"], ["method", "n_clusters", "random_seed"], "clustering")
    _require_keys(raw["reporting"], ["representatives_per_cluster"], "reporting")

    base_dir = path.parent
    manifest_path = Path(raw["manifest_path"])
    output_dir = Path(raw["output_dir"])
    if not manifest_path.is_absolute():
        manifest_path = (base_dir / manifest_path).resolve()
    if not output_dir.is_absolute():
        output_dir = (base_dir / output_dir).resolve()

    config = LatentReportConfig(
        manifest_path=manifest_path,
        output_dir=output_dir,
        features=FeatureReportConfig(
            channels=[str(v) for v in raw["features"]["channels"]],
            feature_variance_csv=(
                ((base_dir / Path(raw["features"]["feature_variance_csv"])).resolve()
                 if not Path(raw["features"]["feature_variance_csv"]).is_absolute()
                 else Path(raw["features"]["feature_variance_csv"]).resolve())
                if raw["features"].get("feature_variance_csv") is not None
                else None
            ),
            feature_variance_cluster=(
                int(raw["features"]["feature_variance_cluster"])
                if raw["features"].get("feature_variance_cluster") is not None
                else None
            ),
            map2_feature_policy=str(raw["features"].get("map2_feature_policy", "full")).lower(),
        ),
        dimensionality_reduction=ReductionConfig(n_pca_components=int(raw["dimensionality_reduction"]["n_pca_components"])),
        clustering=ClusteringConfig(
            method=str(raw["clustering"]["method"]).lower(),
            n_clusters=int(raw["clustering"]["n_clusters"]),
            random_seed=int(raw["clustering"]["random_seed"]),
        ),
        reporting=ReportingConfig(representatives_per_cluster=int(raw["reporting"]["representatives_per_cluster"])),
    )
    validate_latent_report_config(config)
    return config


def validate_latent_report_config(config: LatentReportConfig) -> None:
    if not config.manifest_path.exists():
        raise ValueError(f"Manifest file not found: {config.manifest_path}")
    if config.dimensionality_reduction.n_pca_components <= 0:
        raise ValueError("n_pca_components must be positive")
    if config.clustering.method not in {"gmm", "kmeans"}:
        raise ValueError("clustering.method must be one of: gmm, kmeans")
    if config.clustering.n_clusters <= 1:
        raise ValueError("n_clusters must be greater than 1")
    if config.reporting.representatives_per_cluster <= 0:
        raise ValueError("representatives_per_cluster must be positive")
    if (config.features.feature_variance_csv is None) != (config.features.feature_variance_cluster is None):
        raise ValueError(
            "features.feature_variance_csv and features.feature_variance_cluster must be provided together"
        )
    if config.features.feature_variance_csv is not None and not config.features.feature_variance_csv.exists():
        raise ValueError(f"Feature variance CSV not found: {config.features.feature_variance_csv}")
    if config.features.map2_feature_policy not in {"full", "prior_only"}:
        raise ValueError("features.map2_feature_policy must be one of: full, prior_only")
