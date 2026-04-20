from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Circle
from scipy.ndimage import binary_propagation
from scipy.spatial import cKDTree

try:
    from skimage.feature import blob_log
except ImportError:  # pragma: no cover
    blob_log = None

from .ome import load_binary_mask


DEFAULT_CHANNEL_ORDER = ("MAP2", "FLAG", "HA", "SHANK2")


@dataclass(frozen=True)
class OverlayResult:
    output_path: Path
    sample_id: str
    condition: str
    channel: str
    overlay_field: str
    patch_count: int


def _compute_patch_radius(sample_df: pd.DataFrame) -> float:
    patch_h = int(sample_df["patch_height_px"].iloc[0])
    patch_w = int(sample_df["patch_width_px"].iloc[0])
    return max(float(min(patch_h, patch_w)) / 4.0, 3.0)


def _default_output_path(
    report_path: Path,
    sample_id: str,
    channel_upper: str,
    overlay_field: str,
    suffix: str,
) -> Path:
    output_dir = report_path.parent / "pc_image_overlays"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{sample_id}_{channel_upper.lower()}_{overlay_field}_{suffix}.png"


def _load_sample_context(
    *,
    report_csv: str | Path,
    channel: str,
    condition: str,
    sample_id: str | None,
) -> tuple[Path, pd.DataFrame, str, np.ndarray, np.ndarray, str]:
    report_path = Path(report_csv).resolve()
    report_df = pd.read_csv(report_path)
    selected_sample = _choose_sample(report_df=report_df, condition=condition, sample_id=sample_id)
    sample_df = report_df[report_df["sample_id"].astype(str) == selected_sample].copy()
    if sample_df.empty:
        raise ValueError(f"No rows found for sample '{selected_sample}'.")

    source_image = Path(str(sample_df["source_image"].iloc[0])).resolve()
    mask_image = Path(str(sample_df["mask_image"].iloc[0])).resolve()
    channel_upper = channel.upper()
    channel_index = _resolve_channel_index(sample_df=sample_df, channel=channel_upper)

    volume = _load_channel_zyx(source_image, channel_index)
    max_projection = np.max(np.asarray(volume, dtype=np.float32), axis=0)
    mask = load_binary_mask(mask_image)
    if mask.shape != max_projection.shape:
        raise ValueError(
            f"Mask/image shape mismatch for sample '{selected_sample}': "
            f"mask={mask.shape}, projection={max_projection.shape}"
        )
    return report_path, sample_df, selected_sample, max_projection, mask, channel_upper


def _channel_order_from_schema(channel_schema: str | object) -> list[str]:
    schema = str(channel_schema or "").lower()
    ordered = [channel for channel in DEFAULT_CHANNEL_ORDER if channel.lower() in schema]
    return ordered or list(DEFAULT_CHANNEL_ORDER)


def _choose_sample(report_df: pd.DataFrame, condition: str, sample_id: str | None) -> str:
    work = report_df.copy()
    if condition:
        work = work[work["condition"].astype(str).str.lower() == condition.lower()]
    if work.empty:
        raise ValueError(f"No rows found for condition '{condition}'.")
    if sample_id is not None:
        sample_rows = work[work["sample_id"].astype(str) == str(sample_id)]
        if sample_rows.empty:
            raise ValueError(f"Sample '{sample_id}' not found within condition '{condition}'.")
        return str(sample_id)
    counts = work["sample_id"].astype(str).value_counts()
    return str(counts.index[0])


def _patch_score_overlay(sample_df: pd.DataFrame, image_shape: tuple[int, int], pc_column: str) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_shape
    sum_grid = np.zeros((height, width), dtype=np.float32)
    count_grid = np.zeros((height, width), dtype=np.float32)
    patch_h = int(sample_df["patch_height_px"].iloc[0])
    patch_w = int(sample_df["patch_width_px"].iloc[0])
    half_h = patch_h // 2
    half_w = patch_w // 2

    for row in sample_df.itertuples(index=False):
        score = float(getattr(row, pc_column))
        x = int(getattr(row, "x"))
        y = int(getattr(row, "y"))
        y0 = max(0, y - half_h)
        y1 = min(height, y + half_h)
        x0 = max(0, x - half_w)
        x1 = min(width, x + half_w)
        sum_grid[y0:y1, x0:x1] += score
        count_grid[y0:y1, x0:x1] += 1.0

    mean_grid = np.divide(sum_grid, count_grid, out=np.zeros_like(sum_grid), where=count_grid > 0)
    return mean_grid, count_grid > 0


def _cluster_component_overlay(
    sample_df: pd.DataFrame,
    image_shape: tuple[int, int],
    cluster_column: str,
    full_clusters: list[int],
    *,
    fill_uncovered: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = image_shape
    if not full_clusters:
        raise ValueError(f"No cluster values found in column '{cluster_column}'.")

    cluster_to_index = {cluster_id: idx for idx, cluster_id in enumerate(full_clusters)}
    votes = np.zeros((len(full_clusters), height, width), dtype=np.float32)

    patch_h = int(sample_df["patch_height_px"].iloc[0])
    patch_w = int(sample_df["patch_width_px"].iloc[0])
    sigma_y = max(float(patch_h) / 3.0, 1.0)
    sigma_x = max(float(patch_w) / 3.0, 1.0)
    radius_y = max(1, int(np.ceil(3.0 * sigma_y)))
    radius_x = max(1, int(np.ceil(3.0 * sigma_x)))

    for row in sample_df.itertuples(index=False):
        cluster_id = int(getattr(row, cluster_column))
        vote_index = cluster_to_index[cluster_id]
        x = int(getattr(row, "x"))
        y = int(getattr(row, "y"))
        y0 = max(0, y - radius_y)
        y1 = min(height, y + radius_y + 1)
        x0 = max(0, x - radius_x)
        x1 = min(width, x + radius_x + 1)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        kernel = np.exp(
            -0.5
            * (
                ((yy.astype(np.float32) - float(y)) / sigma_y) ** 2
                + ((xx.astype(np.float32) - float(x)) / sigma_x) ** 2
            )
        )
        votes[vote_index, y0:y1, x0:x1] += kernel.astype(np.float32)

    best_index = np.argmax(votes, axis=0)
    winning_votes = np.max(votes, axis=0)
    total_votes = np.sum(votes, axis=0)
    index_grid = best_index.astype(np.float32)

    uncovered = total_votes <= 0.0
    if fill_uncovered and np.any(uncovered):
        centers = sample_df[["y", "x"]].to_numpy(dtype=np.float32)
        cluster_indices = np.array(
            [cluster_to_index[int(value)] for value in sample_df[cluster_column].to_numpy()],
            dtype=np.int32,
        )
        tree = cKDTree(centers)
        uncovered_coords = np.column_stack(np.where(uncovered)).astype(np.float32)
        _, nearest = tree.query(uncovered_coords, k=1)
        index_grid[uncovered] = cluster_indices[nearest].astype(np.float32)

    confidence = np.divide(winning_votes, total_votes, out=np.zeros_like(winning_votes), where=total_votes > 0.0)
    covered_mask = total_votes > 0.0
    return index_grid, covered_mask, confidence


def _pc_norm(values: np.ndarray) -> Normalize:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return Normalize(vmin=0.0, vmax=1.0)
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if vmin < 0.0 < vmax:
        vmax_abs = max(abs(vmin), abs(vmax))
        return TwoSlopeNorm(vmin=-vmax_abs, vcenter=0.0, vmax=vmax_abs)
    if vmax <= vmin:
        return Normalize(vmin=vmin, vmax=vmin + 1.0)
    return Normalize(vmin=vmin, vmax=vmax)


def _cluster_cmap_norm_and_ticks(full_clusters: list[int]) -> tuple[ListedColormap, BoundaryNorm, np.ndarray]:
    colors = [plt.get_cmap("tab10")(int(cluster_id) % 10) for cluster_id in full_clusters]
    cmap = ListedColormap(colors, name="latent_patch_clusters")
    boundaries = np.arange(-0.5, len(full_clusters) + 0.5, 1.0)
    norm = BoundaryNorm(boundaries, cmap.N)
    ticks = np.arange(len(full_clusters))
    return cmap, norm, ticks


def _resolve_channel_index(sample_df: pd.DataFrame, channel: str) -> int:
    channel_order = _channel_order_from_schema(sample_df["channel_schema"].iloc[0])
    channel_upper = channel.upper()
    if channel_upper not in channel_order:
        raise ValueError(f"Channel '{channel}' not found in inferred channel order {channel_order}.")
    return channel_order.index(channel_upper)


def _load_channel_zyx(path: Path, channel_index: int) -> np.ndarray:
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = getattr(series, "axes", "") or ""
        data = np.asarray(series.asarray())

    index: list[object] = [0] * len(axes)
    for axis_pos, axis_name in enumerate(axes):
        if axis_name == "T":
            index[axis_pos] = 0
        elif axis_name == "C":
            index[axis_pos] = channel_index
        elif axis_name in {"Z", "Y", "X"}:
            index[axis_pos] = slice(None)
        else:
            index[axis_pos] = 0

    view = np.asarray(data[tuple(index)])
    view_axes = "".join(axis_name for axis_name in axes if axis_name in {"Z", "Y", "X"})
    if view_axes == "ZYX":
        return view
    if view_axes == "YX":
        return view[np.newaxis, :, :]
    raise ValueError(f"Unsupported OME axis order after slicing: {view_axes}")


def _draw_mask_outline(ax, mask: np.ndarray, *, color: str = "#2ca25f", linewidth: float = 1.0) -> None:
    ax.contour(
        np.asarray(mask, dtype=np.uint8),
        levels=[0.5],
        colors=[color],
        linewidths=linewidth,
    )


def _normalize_zero_one(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    amin = float(np.min(arr))
    amax = float(np.max(arr))
    if not np.isfinite(amin) or not np.isfinite(amax) or amax <= amin:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - amin) / (amax - amin)


def _detect_log_blobs_2d(
    image: np.ndarray,
    *,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold: float,
    overlap: float,
) -> np.ndarray:
    if blob_log is None:
        raise ValueError("scikit-image is required for segmented blob overlays.")
    norm = _normalize_zero_one(image)
    if not np.any(norm > 0):
        return np.zeros((0, 3), dtype=np.float32)
    blobs = blob_log(
        norm,
        min_sigma=float(min_sigma),
        max_sigma=float(max_sigma),
        num_sigma=int(num_sigma),
        threshold=float(threshold),
        overlap=float(overlap),
        exclude_border=False,
    )
    if len(blobs) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    blobs = np.asarray(blobs, dtype=np.float32)
    blobs[:, 2] *= np.sqrt(2.0)
    return blobs


def _compute_annulus_background_2d(
    img_roi: np.ndarray,
    center_local: tuple[float, float],
    radius: float,
    *,
    inner_mult: float = 2.5,
    outer_mult: float = 4.0,
) -> tuple[float, float]:
    yy, xx = np.ogrid[:img_roi.shape[0], :img_roi.shape[1]]
    y0, x0 = center_local
    dist_sq = (yy - y0) ** 2 + (xx - x0) ** 2
    r_inner = float(inner_mult) * float(radius)
    r_outer = float(outer_mult) * float(radius)
    annulus = (dist_sq >= r_inner**2) & (dist_sq <= r_outer**2)
    if np.any(annulus):
        bg_pixels = img_roi[annulus]
        return float(np.mean(bg_pixels)), float(np.std(bg_pixels))
    return float(np.mean(img_roi)), float(np.std(img_roi))


def _adaptive_thresholds(
    mu_signal: float,
    mu_bg: float,
    sigma_bg: float,
    *,
    base_k_high: float = 5.0,
    base_k_low: float = 3.0,
) -> tuple[float, float]:
    epsilon = 1e-6
    snr = (mu_signal - mu_bg) / (sigma_bg + epsilon)
    if snr >= 5.0:
        k_high, k_low = base_k_high, base_k_low
    elif snr >= 3.0:
        t = (snr - 3.0) / 2.0
        k_high = 3.5 + t * (base_k_high - 3.5)
        k_low = 2.0 + t * (base_k_low - 2.0)
    else:
        k_high, k_low = 3.5, 2.0
    return float(mu_bg + k_high * sigma_bg), float(mu_bg + k_low * sigma_bg)


def _segment_blob_hysteresis_2d(
    img_roi: np.ndarray,
    center_local: tuple[float, float],
    radius: float,
    mask_roi: np.ndarray | None,
    *,
    k_high: float = 5.0,
    k_low: float = 3.0,
    max_radius_mult: float = 5.0,
) -> np.ndarray:
    cy, cx = (int(round(center_local[0])), int(round(center_local[1])))
    if not (0 <= cy < img_roi.shape[0] and 0 <= cx < img_roi.shape[1]):
        return np.zeros_like(img_roi, dtype=bool)
    mu_signal = float(img_roi[cy, cx])
    mu_bg, sigma_bg = _compute_annulus_background_2d(img_roi, center_local, radius)
    t_high, t_low = _adaptive_thresholds(mu_signal, mu_bg, sigma_bg, base_k_high=k_high, base_k_low=k_low)
    candidate = img_roi >= t_low
    core = img_roi >= t_high
    if mask_roi is not None:
        valid = np.asarray(mask_roi, dtype=bool)
        candidate &= valid
        core &= valid
    if not np.any(core):
        core[cy, cx] = candidate[cy, cx]
    if not np.any(core):
        return np.zeros_like(img_roi, dtype=bool)
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    region = binary_propagation(core, mask=candidate, structure=struct)
    yy, xx = np.ogrid[:img_roi.shape[0], :img_roi.shape[1]]
    dist_sq = (yy - center_local[0]) ** 2 + (xx - center_local[1]) ** 2
    max_r = float(max_radius_mult) * float(radius)
    return np.asarray(region & (dist_sq <= max_r**2), dtype=bool)


def _segmented_blob_regions(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    min_sigma: float,
    max_sigma: float,
    num_sigma: int,
    threshold: float,
    overlap: float,
    roi_radius_mult: float = 4.0,
    k_high: float = 5.0,
    k_low: float = 3.0,
    max_radius_mult: float = 5.0,
) -> tuple[list[np.ndarray], np.ndarray]:
    blobs = _detect_log_blobs_2d(
        image,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
    )
    regions: list[np.ndarray] = []
    combined = np.zeros_like(image, dtype=bool)
    for y, x, radius in blobs:
        roi_r = max(1, int(np.ceil(float(roi_radius_mult) * float(radius))))
        y0 = max(0, int(np.floor(y)) - roi_r)
        y1 = min(image.shape[0], int(np.floor(y)) + roi_r + 1)
        x0 = max(0, int(np.floor(x)) - roi_r)
        x1 = min(image.shape[1], int(np.floor(x)) + roi_r + 1)
        img_roi = np.asarray(image[y0:y1, x0:x1], dtype=np.float32)
        mask_roi = np.asarray(mask[y0:y1, x0:x1], dtype=bool) if mask is not None else None
        region_roi = _segment_blob_hysteresis_2d(
            img_roi,
            center_local=(float(y - y0), float(x - x0)),
            radius=float(radius),
            mask_roi=mask_roi,
            k_high=k_high,
            k_low=k_low,
            max_radius_mult=max_radius_mult,
        )
        if not np.any(region_roi):
            continue
        region_global = np.zeros_like(image, dtype=bool)
        region_global[y0:y1, x0:x1] = region_roi
        regions.append(region_global)
        combined |= region_global
    return regions, combined


def _add_center_circles(
    ax,
    sample_df: pd.DataFrame,
    *,
    facecolors: list[tuple[float, float, float, float]],
    radius_px: float,
    alpha: float,
    edgecolor: str,
    linewidth: float,
) -> None:
    for row, facecolor in zip(sample_df.itertuples(index=False), facecolors):
        ax.add_patch(
            Circle(
                (float(getattr(row, "x")), float(getattr(row, "y"))),
                radius=radius_px,
                facecolor=facecolor,
                edgecolor=edgecolor,
                linewidth=linewidth,
                alpha=alpha,
            )
        )


def create_principal_component_overlay(
    *,
    report_csv: str | Path,
    channel: str,
    pc_column: str = "pca_1",
    condition: str = "positive",
    sample_id: str | None = None,
    output_path: str | Path | None = None,
) -> OverlayResult:
    report_path, sample_df, selected_sample, max_projection, mask, channel_upper = _load_sample_context(
        report_csv=report_csv,
        channel=channel,
        condition=condition,
        sample_id=sample_id,
    )
    report_df = pd.read_csv(report_path)
    if pc_column not in report_df.columns:
        raise ValueError(f"Principal component column '{pc_column}' not found in {report_path}.")

    pc_grid, covered_mask = _patch_score_overlay(
        sample_df=sample_df,
        image_shape=max_projection.shape,
        pc_column=pc_column,
    )
    norm = _pc_norm(pc_grid[covered_mask])
    masked_pc = np.ma.masked_where(~covered_mask, pc_grid)

    if output_path is None:
        output_path = _default_output_path(
            report_path=report_path,
            sample_id=selected_sample,
            channel_upper=channel_upper,
            overlay_field=pc_column,
            suffix="overlay",
        )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(max_projection, cmap="gray")
    axes[0].set_title(f"{channel_upper} Max Projection")

    axes[1].imshow(max_projection, cmap="gray")
    _draw_mask_outline(axes[1], mask)
    axes[1].set_title(f"{channel_upper} + MAP2 Mask")

    axes[2].imshow(max_projection, cmap="gray")
    _draw_mask_outline(axes[2], mask)
    pc_artist = axes[2].imshow(masked_pc, cmap="coolwarm", norm=norm, alpha=0.42)
    axes[2].set_title(f"{channel_upper} + MAP2 Mask + {pc_column.upper()}")

    fig.colorbar(pc_artist, ax=axes[2], fraction=0.046, pad=0.04, label=pc_column.upper())
    fig.suptitle(f"{selected_sample} ({condition})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return OverlayResult(
        output_path=output_path,
        sample_id=selected_sample,
        condition=str(sample_df["condition"].iloc[0]),
        channel=channel_upper,
        overlay_field=pc_column,
        patch_count=int(len(sample_df)),
    )


def create_cluster_component_overlay(
    *,
    report_csv: str | Path,
    channel: str,
    cluster_column: str = "cluster_id",
    condition: str = "positive",
    sample_id: str | None = None,
    output_path: str | Path | None = None,
) -> OverlayResult:
    report_path, sample_df, selected_sample, max_projection, mask, channel_upper = _load_sample_context(
        report_csv=report_csv,
        channel=channel,
        condition=condition,
        sample_id=sample_id,
    )
    report_df = pd.read_csv(report_path)
    if cluster_column not in report_df.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not found in {report_path}.")
    full_clusters = sorted({int(value) for value in report_df[cluster_column].dropna().astype(int).tolist()})

    cluster_grid, _, _ = _cluster_component_overlay(
        sample_df=sample_df,
        image_shape=max_projection.shape,
        cluster_column=cluster_column,
        full_clusters=full_clusters,
    )
    cmap, norm, ticks = _cluster_cmap_norm_and_ticks(full_clusters)

    if output_path is None:
        output_path = _default_output_path(
            report_path=report_path,
            sample_id=selected_sample,
            channel_upper=channel_upper,
            overlay_field=cluster_column,
            suffix="overlay",
        )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(max_projection, cmap="gray")
    axes[0].set_title(f"{channel_upper} Max Projection")

    axes[1].imshow(max_projection, cmap="gray")
    _draw_mask_outline(axes[1], mask)
    axes[1].set_title(f"{channel_upper} + MAP2 Mask")

    axes[2].imshow(max_projection, cmap="gray")
    _draw_mask_outline(axes[2], mask)
    cluster_artist = axes[2].imshow(cluster_grid, cmap=cmap, norm=norm, alpha=0.34)
    axes[2].set_title(f"{channel_upper} + MAP2 Mask + {cluster_column}")

    colorbar = fig.colorbar(cluster_artist, ax=axes[2], fraction=0.046, pad=0.04, ticks=ticks)
    colorbar.ax.set_yticklabels([str(cluster_id) for cluster_id in full_clusters])
    colorbar.set_label(cluster_column)
    fig.suptitle(f"{selected_sample} ({condition})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return OverlayResult(
        output_path=output_path,
        sample_id=selected_sample,
        condition=str(sample_df["condition"].iloc[0]),
        channel=channel_upper,
        overlay_field=cluster_column,
        patch_count=int(len(sample_df)),
    )


def create_cluster_patch_center_overlay(
    *,
    report_csv: str | Path,
    channel: str,
    cluster_column: str = "cluster_id",
    condition: str = "positive",
    sample_id: str | None = None,
    output_path: str | Path | None = None,
) -> OverlayResult:
    report_path, sample_df, selected_sample, max_projection, mask, channel_upper = _load_sample_context(
        report_csv=report_csv,
        channel=channel,
        condition=condition,
        sample_id=sample_id,
    )
    report_df = pd.read_csv(report_path)
    if cluster_column not in report_df.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not found in {report_path}.")

    full_clusters = sorted({int(value) for value in report_df[cluster_column].dropna().astype(int).tolist()})
    if not full_clusters:
        raise ValueError(f"No cluster values found in column '{cluster_column}'.")
    cluster_to_index = {cluster_id: idx for idx, cluster_id in enumerate(full_clusters)}
    cmap, norm, ticks = _cluster_cmap_norm_and_ticks(full_clusters)

    if output_path is None:
        output_path = _default_output_path(
            report_path=report_path,
            sample_id=selected_sample,
            channel_upper=channel_upper,
            overlay_field=cluster_column,
            suffix="patch_centers",
        )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(max_projection, cmap="gray")
    axes[0].set_title(f"{channel_upper} Max Projection")

    axes[1].imshow(max_projection, cmap="gray")
    _draw_mask_outline(axes[1], mask)
    axes[1].set_title(f"{channel_upper} + MAP2 Mask")

    axes[2].imshow(max_projection, cmap="gray")
    _draw_mask_outline(axes[2], mask)
    facecolors = [
        cmap(cluster_to_index[int(value)])
        for value in sample_df[cluster_column].to_numpy()
    ]
    _add_center_circles(
        axes[2],
        sample_df,
        facecolors=facecolors,
        radius_px=1.25,
        alpha=0.9,
        edgecolor="white",
        linewidth=0.35,
    )
    axes[2].set_title(f"{channel_upper} Patch Centers by {cluster_column}")

    scatter = axes[2].scatter([], [], c=[], cmap=cmap, norm=norm)
    colorbar = fig.colorbar(scatter, ax=axes[2], fraction=0.046, pad=0.04, ticks=ticks)
    colorbar.ax.set_yticklabels([str(cluster_id) for cluster_id in full_clusters])
    colorbar.set_label(cluster_column)
    fig.suptitle(f"{selected_sample} ({condition})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return OverlayResult(
        output_path=output_path,
        sample_id=selected_sample,
        condition=str(sample_df["condition"].iloc[0]),
        channel=channel_upper,
        overlay_field=cluster_column,
        patch_count=int(len(sample_df)),
    )


def create_cluster_puncta_candidate_overlay(
    *,
    report_csv: str | Path,
    channel: str,
    cluster_column: str = "cluster_id",
    candidate_column: str | None = None,
    candidate_quantile: float = 0.9,
    condition: str = "positive",
    sample_id: str | None = None,
    output_path: str | Path | None = None,
) -> OverlayResult:
    report_path, sample_df, selected_sample, max_projection, mask, channel_upper = _load_sample_context(
        report_csv=report_csv,
        channel=channel,
        condition=condition,
        sample_id=sample_id,
    )
    report_df = pd.read_csv(report_path)
    if cluster_column not in report_df.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not found in {report_path}.")

    candidate_metric = candidate_column or f"{channel_upper.lower()}_log_puncta_dominance"
    if candidate_metric not in sample_df.columns:
        raise ValueError(f"Candidate column '{candidate_metric}' not found in {report_path}.")
    if not 0.0 < candidate_quantile < 1.0:
        raise ValueError("candidate_quantile must be between 0 and 1.")

    scores = pd.to_numeric(sample_df[candidate_metric], errors="coerce")
    valid_scores = scores.dropna()
    if valid_scores.empty:
        raise ValueError(f"Candidate column '{candidate_metric}' has no finite values for sample '{selected_sample}'.")
    threshold = float(valid_scores.quantile(candidate_quantile))
    candidate_df = sample_df.loc[scores >= threshold].copy()
    if candidate_df.empty:
        raise ValueError(
            f"No candidate patches remain after applying {candidate_metric} >= quantile {candidate_quantile:.2f}."
        )

    full_clusters = sorted({int(value) for value in report_df[cluster_column].dropna().astype(int).tolist()})
    cluster_to_index = {cluster_id: idx for idx, cluster_id in enumerate(full_clusters)}
    cmap, norm, ticks = _cluster_cmap_norm_and_ticks(full_clusters)

    if output_path is None:
        output_path = _default_output_path(
            report_path=report_path,
            sample_id=selected_sample,
            channel_upper=channel_upper,
            overlay_field=cluster_column,
            suffix="puncta_candidates",
        )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(max_projection, cmap="gray")
    axes[0].set_title(f"{channel_upper} Max Projection")

    axes[1].imshow(max_projection, cmap="gray")
    _draw_mask_outline(axes[1], mask)
    _add_center_circles(
        axes[1],
        candidate_df,
        facecolors=["#f0e442"] * len(candidate_df),
        radius_px=1.5,
        alpha=0.95,
        edgecolor="#f0e442",
        linewidth=0.8,
    )
    axes[1].set_title(f"{channel_upper} Candidate Patches")

    axes[2].imshow(max_projection, cmap="gray")
    _draw_mask_outline(axes[2], mask)
    facecolors = [
        cmap(cluster_to_index[int(value)])
        for value in candidate_df[cluster_column].to_numpy()
    ]
    _add_center_circles(
        axes[2],
        candidate_df,
        facecolors=facecolors,
        radius_px=1.5,
        alpha=0.95,
        edgecolor="white",
        linewidth=0.4,
    )
    axes[2].set_title(
        f"{channel_upper} Candidate Clusters\n{candidate_metric} >= q{candidate_quantile:.2f} ({len(candidate_df)} patches)"
    )

    scatter = axes[2].scatter([], [], c=[], cmap=cmap, norm=norm)
    colorbar = fig.colorbar(scatter, ax=axes[2], fraction=0.046, pad=0.04, ticks=ticks)
    colorbar.ax.set_yticklabels([str(cluster_id) for cluster_id in full_clusters])
    colorbar.set_label(cluster_column)
    fig.suptitle(f"{selected_sample} ({condition})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return OverlayResult(
        output_path=output_path,
        sample_id=selected_sample,
        condition=str(sample_df["condition"].iloc[0]),
        channel=channel_upper,
        overlay_field=f"{cluster_column}:{candidate_metric}",
        patch_count=int(len(candidate_df)),
    )


def create_confidence_filtered_cluster_overlay(
    *,
    report_csv: str | Path,
    channel: str,
    cluster_column: str = "cluster_id",
    confidence_threshold: float = 0.55,
    condition: str = "positive",
    sample_id: str | None = None,
    output_path: str | Path | None = None,
) -> OverlayResult:
    report_path, sample_df, selected_sample, max_projection, mask, channel_upper = _load_sample_context(
        report_csv=report_csv,
        channel=channel,
        condition=condition,
        sample_id=sample_id,
    )
    report_df = pd.read_csv(report_path)
    if cluster_column not in report_df.columns:
        raise ValueError(f"Cluster column '{cluster_column}' not found in {report_path}.")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1.")

    full_clusters = sorted({int(value) for value in report_df[cluster_column].dropna().astype(int).tolist()})
    cluster_grid, covered_mask, confidence = _cluster_component_overlay(
        sample_df=sample_df,
        image_shape=max_projection.shape,
        cluster_column=cluster_column,
        full_clusters=full_clusters,
        fill_uncovered=False,
    )
    keep_mask = covered_mask & (confidence >= confidence_threshold)
    masked_clusters = np.ma.masked_where(~keep_mask, cluster_grid)
    cmap, norm, ticks = _cluster_cmap_norm_and_ticks(full_clusters)

    if output_path is None:
        output_path = _default_output_path(
            report_path=report_path,
            sample_id=selected_sample,
            channel_upper=channel_upper,
            overlay_field=cluster_column,
            suffix="confidence_filtered",
        )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(max_projection, cmap="gray")
    axes[0].set_title(f"{channel_upper} Max Projection")

    axes[1].imshow(max_projection, cmap="gray")
    _draw_mask_outline(axes[1], mask)
    confidence_artist = axes[1].imshow(np.ma.masked_where(~covered_mask, confidence), cmap="magma", vmin=0.0, vmax=1.0, alpha=0.46)
    axes[1].set_title(f"Vote Confidence ({cluster_column})")

    axes[2].imshow(max_projection, cmap="gray")
    _draw_mask_outline(axes[2], mask)
    cluster_artist = axes[2].imshow(masked_clusters, cmap=cmap, norm=norm, alpha=0.46)
    axes[2].set_title(f"{channel_upper} {cluster_column} | confidence >= {confidence_threshold:.2f}")

    fig.colorbar(confidence_artist, ax=axes[1], fraction=0.046, pad=0.04, label="vote confidence")
    colorbar = fig.colorbar(cluster_artist, ax=axes[2], fraction=0.046, pad=0.04, ticks=ticks)
    colorbar.ax.set_yticklabels([str(cluster_id) for cluster_id in full_clusters])
    colorbar.set_label(cluster_column)
    fig.suptitle(f"{selected_sample} ({condition})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return OverlayResult(
        output_path=output_path,
        sample_id=selected_sample,
        condition=str(sample_df["condition"].iloc[0]),
        channel=channel_upper,
        overlay_field=f"{cluster_column}:confidence",
        patch_count=int(len(sample_df)),
    )


def create_segmented_blob_overlay(
    *,
    report_csv: str | Path,
    channel: str,
    condition: str = "positive",
    sample_id: str | None = None,
    min_sigma: float = 0.75,
    max_sigma: float = 3.5,
    num_sigma: int = 6,
    blob_threshold: float = 0.03,
    overlap: float = 0.5,
    output_path: str | Path | None = None,
) -> OverlayResult:
    report_path, sample_df, selected_sample, max_projection, mask, channel_upper = _load_sample_context(
        report_csv=report_csv,
        channel=channel,
        condition=condition,
        sample_id=sample_id,
    )
    regions, combined = _segmented_blob_regions(
        max_projection,
        mask,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=blob_threshold,
        overlap=overlap,
    )
    if output_path is None:
        output_path = _default_output_path(
            report_path=report_path,
            sample_id=selected_sample,
            channel_upper=channel_upper,
            overlay_field="segmented_blobs",
            suffix="overlay",
        )
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    masked_regions = np.ma.masked_where(~combined, combined.astype(np.float32))
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(max_projection, cmap="gray")
    axes[0].set_title(f"{channel_upper} Max Projection")

    axes[1].imshow(max_projection, cmap="gray")
    _draw_mask_outline(axes[1], mask)
    axes[1].set_title(f"{channel_upper} + MAP2 Mask")

    axes[2].imshow(max_projection, cmap="gray")
    _draw_mask_outline(axes[2], mask)
    axes[2].imshow(masked_regions, cmap=ListedColormap(["#ff4f6f"]), alpha=0.24, vmin=0.0, vmax=1.0)
    if np.any(combined):
        axes[2].contour(
            combined.astype(np.uint8),
            levels=[0.5],
            colors=["#ff9fb0"],
            linewidths=0.9,
        )
    axes[2].set_title(f"{channel_upper} Segmented LoG Blobs")

    fig.suptitle(f"{selected_sample} ({condition})", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    return OverlayResult(
        output_path=output_path,
        sample_id=selected_sample,
        condition=str(sample_df["condition"].iloc[0]),
        channel=channel_upper,
        overlay_field="segmented_blobs",
        patch_count=int(len(regions)),
    )
