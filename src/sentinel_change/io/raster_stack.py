from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import rasterio
from rasterio.enums import ColorInterp

@dataclass(frozen=True)
class RasterBandMeta:
    crs_wkt: str
    transform: Tuple[float, float, float, float, float, float]
    width: int
    height: int
    dtype: str
    nodata: float | int | None


def _transform_to_tuple(transform) -> Tuple[float, float, float, float, float, float]:
    return (transform.a, transform.b, transform.c, transform.d, transform.e, transform.f)


def read_single_band(path: Path) -> tuple[np.ndarray, RasterBandMeta]:
    with rasterio.open(path) as src:
        array = src.read(1)
        meta = RasterBandMeta(
            crs_wkt=src.crs.to_wkt() if src.crs else "",
            transform=_transform_to_tuple(src.transform),
            width=src.width,
            height=src.height,
            dtype=src.dtypes[0],
            nodata=src.nodata,
        )
    return array, meta


def verify_consistency(metadata: Iterable[RasterBandMeta]) -> None:
    metas = list(metadata)
    if not metas:
        raise ValueError("No metadata provided for consistency check.")

    ref = metas[0]
    for idx, m in enumerate(metas[1:], start=2):
        if m.crs_wkt != ref.crs_wkt:
            raise ValueError(f"CRS mismatch at band index {idx}.")
        if m.transform != ref.transform:
            raise ValueError(f"Transform mismatch at band index {idx}.")
        if (m.width, m.height) != (ref.width, ref.height):
            raise ValueError(f"Dimension mismatch at band index {idx}.")


def stack_bands_by_name(input_dir: Path, band_names: Iterable[str]) -> tuple[np.ndarray, Dict[str, RasterBandMeta]]:
    arrays: list[np.ndarray] = []
    metas: Dict[str, RasterBandMeta] = {}

    for band_name in band_names:
        band_path = input_dir / f"{band_name}.tif"
        if not band_path.exists():
            raise FileNotFoundError(f"Missing expected band file: {band_path}")

        arr, meta = read_single_band(band_path)
        arrays.append(arr)
        metas[band_name] = meta

    verify_consistency(metas.values())

    stacked = np.stack(arrays, axis=0)
    return stacked, metas


def write_multiband_geotiff(output_path: Path, stacked: np.ndarray, meta: RasterBandMeta) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    profile = {
        "driver": "GTiff",
        "height": stacked.shape[1],
        "width": stacked.shape[2],
        "count": stacked.shape[0],
        "dtype": stacked.dtype,
        "crs": meta.crs_wkt or None,
        "transform": rasterio.Affine(*meta.transform),
        "compress": "deflate",
        "tiled": True,
    }

    if meta.nodata is not None:
        profile["nodata"] = meta.nodata

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(stacked)
        # if 3-band RGB, set color interpretation and band descriptions
        if stacked.shape[0] == 3:
            try:
                dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
                dst.set_band_description(1, "B04_Red")
                dst.set_band_description(2, "B03_Green")
                dst.set_band_description(3, "B02_Blue")
            except Exception:
                # some rasterio versions / drivers may not support setting these attributes
                pass


def _percentile_stretch_to_uint8(band: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    band_f = band.astype(np.float32)
    finite = np.isfinite(band_f)
    if not finite.any():
        return np.zeros_like(band_f, dtype=np.uint8)

    valid = band_f[finite]
    # Detect Sentinel-2 integer scale (0-10000) and rescale to 0-1
    approx_max = float(valid.max())
    if approx_max > 5000:
        band_f = band_f / 10000.0
        valid = band_f[finite]

    lo = float(np.percentile(valid, low_pct))
    hi = float(np.percentile(valid, high_pct))
    if hi <= lo:
        return np.zeros_like(band_f, dtype=np.uint8)

    stretched = (band_f - lo) / (hi - lo)
    stretched = np.clip(stretched, 0.0, 1.0)

    # Apply simple gamma correction for display
    gamma = 1.0 / 2.2
    stretched = np.power(stretched, gamma)

    return (stretched * 255.0).round().astype(np.uint8)


def write_rgb_preview_geotiff(output_path: Path, stacked_rgb: np.ndarray, meta: RasterBandMeta) -> None:
    if stacked_rgb.shape[0] != 3:
        raise ValueError("RGB preview expects exactly 3 bands in R,G,B order.")

    rgb_uint8 = np.stack(
        [_percentile_stretch_to_uint8(stacked_rgb[i]) for i in range(3)],
        axis=0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": rgb_uint8.shape[1],
        "width": rgb_uint8.shape[2],
        "count": 3,
        "dtype": "uint8",
        "crs": meta.crs_wkt or None,
        "transform": rasterio.Affine(*meta.transform),
        "compress": "deflate",
        "tiled": True,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(rgb_uint8)
        try:
            dst.colorinterp = (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            dst.set_band_description(1, "Red_stretched")
            dst.set_band_description(2, "Green_stretched")
            dst.set_band_description(3, "Blue_stretched")
        except Exception:
            pass
