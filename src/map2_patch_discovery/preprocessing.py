from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


VALID_SCALERS = {"standard", "robust", "minmax", "none"}


def scale_feature_matrix(feature_matrix: np.ndarray, scaler_name: str) -> np.ndarray:
    scaler_name = str(scaler_name).lower()
    matrix = np.asarray(feature_matrix, dtype=np.float64)
    if scaler_name == "standard":
        return StandardScaler().fit_transform(matrix)
    if scaler_name == "robust":
        return RobustScaler().fit_transform(matrix)
    if scaler_name == "minmax":
        return MinMaxScaler().fit_transform(matrix)
    if scaler_name == "none":
        return matrix
    raise ValueError(f"Unsupported scaler '{scaler_name}'. Valid options: {sorted(VALID_SCALERS)}")
