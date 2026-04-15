from __future__ import annotations

import numpy as np


def summarize_channel_crop(crop: np.ndarray) -> dict[str, float | int]:
    arr = np.asarray(crop, dtype=np.float32)
    z_profile = arr.mean(axis=(1, 2)) if arr.ndim == 3 else np.array([float(arr.mean())], dtype=np.float32)
    peak_z = int(np.argmax(z_profile)) if z_profile.size > 0 else 0
    occupancy = int(np.count_nonzero(z_profile > 0))
    return {
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
        "sum": float(np.sum(arr)),
        "std": float(np.std(arr)),
        "z_peak_index": peak_z,
        "z_occupancy": occupancy,
    }
