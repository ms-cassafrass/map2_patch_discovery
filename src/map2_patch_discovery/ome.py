from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

try:
    import zarr
except ImportError:  # pragma: no cover
    zarr = None


@dataclass
class OmeImage:
    path: Path
    data: Any
    axes: str
    shape: tuple[int, ...]
    channel_names: list[str]
    pixel_size_xy_um: float | None
    pixel_size_z_um: float | None
    is_lazy: bool

    def get_zyx(self, channel_index: int) -> np.ndarray:
        axes = self.axes
        arr = self.data
        index: list[Any] = [0] * len(axes)
        for axis_pos, axis_name in enumerate(axes):
            if axis_name == "T":
                index[axis_pos] = 0
            elif axis_name == "C":
                index[axis_pos] = channel_index
            elif axis_name in {"Z", "Y", "X"}:
                index[axis_pos] = slice(None)
            else:
                index[axis_pos] = 0
        view = np.asarray(arr[tuple(index)])
        view_axes = "".join(axis_name for axis_name in axes if axis_name in {"Z", "Y", "X"})
        if view_axes == "ZYX":
            return view
        if view_axes == "YX":
            return view[np.newaxis, :, :]
        raise ValueError(f"Unsupported OME axis order after slicing: {view_axes}")


def _read_channel_names(path: Path) -> list[str]:
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = getattr(series, "axes", "") or ""
        shape = tuple(int(v) for v in series.shape)
    if "C" not in axes:
        return ["channel_0"]
    n_channels = shape[axes.index("C")]
    return [f"channel_{idx}" for idx in range(n_channels)]


def _read_pixel_sizes(path: Path) -> tuple[float | None, float | None]:
    try:
        with tifffile.TiffFile(path) as tf:
            ome = tf.ome_metadata
        if not ome:
            return None, None
        import xml.etree.ElementTree as et

        root = et.fromstring(ome)
        pixels = root.find(".//{*}Pixels")
        if pixels is None:
            return None, None
        xy = pixels.attrib.get("PhysicalSizeX")
        z = pixels.attrib.get("PhysicalSizeZ")
        return (None if xy is None else float(xy), None if z is None else float(z))
    except Exception:
        return None, None


def open_ome_image(path: str | Path, channel_names: list[str] | None = None) -> OmeImage:
    path = Path(path).resolve()
    with tifffile.TiffFile(path) as tf:
        series = tf.series[0]
        axes = getattr(series, "axes", "") or ""
        shape = tuple(int(v) for v in series.shape)

    pixel_size_xy_um, pixel_size_z_um = _read_pixel_sizes(path)

    if zarr is not None:
        try:
            with tifffile.TiffFile(path) as tf:
                store = tf.series[0].aszarr()
                data = zarr.open(store, mode="r")
            is_lazy = True
        except (TypeError, AttributeError, ValueError):
            data = tifffile.imread(path)
            is_lazy = False
    else:
        data = tifffile.imread(path)
        is_lazy = False

    names = channel_names or _read_channel_names(path)
    return OmeImage(
        path=path,
        data=data,
        axes=axes,
        shape=shape,
        channel_names=names,
        pixel_size_xy_um=pixel_size_xy_um,
        pixel_size_z_um=pixel_size_z_um,
        is_lazy=is_lazy,
    )


def load_binary_mask(path: str | Path) -> np.ndarray:
    data = tifffile.imread(Path(path).resolve())
    data = np.asarray(data)
    while data.ndim > 2:
        data = np.max(data, axis=0)
    if data.ndim != 2:
        raise ValueError(f"Mask must resolve to 2D, got shape {data.shape}")
    return data > 0
