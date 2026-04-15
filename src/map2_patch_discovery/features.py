from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_patch_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def extract_engineered_features(manifest: pd.DataFrame, channels: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in manifest.itertuples(index=False):
        payload = load_patch_npz(row.patch_path)
        mask = payload["map2_mask"].astype(bool)
        flat_record: dict[str, object] = {"patch_id": row.patch_id}

        channel_means: dict[str, float] = {}
        channel_sums: dict[str, float] = {}
        for channel in channels:
            key = f"channel_{channel}"
            if key not in payload:
                raise ValueError(f"Patch file {row.patch_path} missing expected key {key}")
            crop = payload[key].astype(np.float32)
            max_proj = np.max(crop, axis=0)
            mean_proj = np.mean(crop, axis=0)
            inside_vals = mean_proj[mask] if np.any(mask) else mean_proj.ravel()
            outside_vals = mean_proj[~mask] if np.any(~mask) else mean_proj.ravel()

            channel_means[channel] = float(np.mean(mean_proj))
            channel_sums[channel] = float(np.sum(crop))

            flat_record[f"{channel.lower()}_proj_mean"] = float(np.mean(mean_proj))
            flat_record[f"{channel.lower()}_proj_max"] = float(np.max(max_proj))
            flat_record[f"{channel.lower()}_proj_std"] = float(np.std(mean_proj))
            flat_record[f"{channel.lower()}_inside_mean"] = float(np.mean(inside_vals))
            flat_record[f"{channel.lower()}_outside_mean"] = float(np.mean(outside_vals))
            flat_record[f"{channel.lower()}_inside_outside_ratio"] = _safe_ratio(
                float(np.mean(inside_vals)),
                float(np.mean(outside_vals)) if np.size(outside_vals) else 0.0,
            )
            z_profile = crop.mean(axis=(1, 2))
            flat_record[f"{channel.lower()}_z_peak"] = int(np.argmax(z_profile))
            flat_record[f"{channel.lower()}_z_std"] = float(np.std(z_profile))

        flat_record["map2_mask_fraction"] = float(np.mean(mask))
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

    return pd.DataFrame.from_records(records)
