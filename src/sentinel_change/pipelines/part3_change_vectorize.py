from __future__ import annotations

from pathlib import Path
import sys

import json
import math
import numpy as np
import rasterio
from rasterio.features import shapes, geometry_mask

# package import bootstrapping when run as script
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _approx_meters_per_degree(lat_deg: float) -> tuple[float, float]:
    """Approximate meters per degree at given latitude for lat and lon.

    Uses simple approximations; accurate enough for reporting approximate areas when CRS is geographic.
    """
    lat = math.radians(lat_deg)
    m_per_deg_lat = 111132.954 - 559.822 * math.cos(2 * lat) + 1.175 * math.cos(4 * lat)
    m_per_deg_lon = 111412.84 * math.cos(lat) - 93.5 * math.cos(3 * lat)
    return m_per_deg_lat, abs(m_per_deg_lon)


def run_part3_geojson(base_dir: Path, before_date: str = "2023-08-12", after_date: str = "2023-09-02") -> Path:
    processed = base_dir / "data" / "processed"
    binary = processed / "change_binary_sam.tif"
    if not binary.exists():
        raise FileNotFoundError("Binary change raster not found. Run Part 2 first.")

    with rasterio.open(binary) as src:
        arr = src.read(1)
        transform = src.transform
        crs = src.crs

    mask = arr == 255
    features = []

    # try fast path: label connected components once using scipy if available
    try:
        import scipy.ndimage as ndi

        labeled, ncomps = ndi.label(mask)
        if ncomps == 0:
            print("No change polygons found.")
            out_path = processed / "change_features.geojson"
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump({"type": "FeatureCollection", "features": []}, fh)
            return out_path

        # vectorize labeled array; each label becomes a polygon
        for geom, val in shapes(labeled, transform=transform):
            label_val = int(val)
            if label_val == 0:
                continue
            # compute pixel_count by counting label pixels equal to label_val
            # compute bounding box to limit area
            geom_bounds = geom["coordinates"][0]
            xs = [c[0] for c in geom_bounds]
            ys = [c[1] for c in geom_bounds]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            col_min, row_min = ~transform * (minx, maxy)
            col_max, row_max = ~transform * (maxx, miny)
            c0, r0 = int(max(0, math.floor(col_min))), int(max(0, math.floor(row_min)))
            c1, r1 = int(min(arr.shape[1], math.ceil(col_max))), int(min(arr.shape[0], math.ceil(row_max)))
            if r0 >= r1 or c0 >= c1:
                pixel_count = 0
            else:
                window = labeled[r0:r1, c0:c1]
                pixel_count = int(np.count_nonzero(window == label_val))

            # compute area in m2
            pixel_area_map_units = abs(transform.a * transform.e)
            if crs is not None and getattr(crs, "is_geographic", False):
                coords = geom.get("coordinates", [[]])[0]
                if not coords:
                    centroid_lat = 0.0
                else:
                    ys = [pt[1] for pt in coords]
                    centroid_lat = float(sum(ys) / len(ys))
                m_lat, m_lon = _approx_meters_per_degree(centroid_lat)
                meters2_per_deg2 = m_lat * m_lon
                area_m2 = pixel_count * pixel_area_map_units * meters2_per_deg2
            else:
                area_m2 = pixel_count * pixel_area_map_units

            feat = {
                "type": "Feature",
                "properties": {
                    "date_before": before_date,
                    "date_after": after_date,
                    "area_m2": float(area_m2),
                    "confidence": 1.0,
                    "pixel_count": int(pixel_count),
                },
                "geometry": geom,
            }
            features.append(feat)

    except Exception:
        # fall back to original per-shape bbox approach if scipy not available
        for geom, val in shapes(arr, mask=mask, transform=transform):
            if val != 255:
                continue
            # compute pixel count within this geometry by limiting to bbox window
            geom_bounds = geom["coordinates"][0]
            xs = [c[0] for c in geom_bounds]
            ys = [c[1] for c in geom_bounds]
            minx, maxx = min(xs), max(xs)
            miny, maxy = min(ys), max(ys)
            # convert bounds to row/col window
            col_min, row_min = ~transform * (minx, maxy)
            col_max, row_max = ~transform * (maxx, miny)
            c0, r0 = int(max(0, math.floor(col_min))), int(max(0, math.floor(row_min)))
            c1, r1 = int(min(arr.shape[1], math.ceil(col_max))), int(min(arr.shape[0], math.ceil(row_max)))
            if r0 >= r1 or c0 >= c1:
                pixel_count = 0
            else:
                window = mask[r0:r1, c0:c1]
                pixel_count = int(np.count_nonzero(window))

            # compute area in m2
            pixel_area_map_units = abs(transform.a * transform.e)
            if crs is not None and getattr(crs, "is_geographic", False):
                # geographic CRS: convert degrees^2 to m^2 approximately at polygon centroid
                coords = geom.get("coordinates", [[]])[0]
                if not coords:
                    centroid_lat = 0.0
                else:
                    ys = [pt[1] for pt in coords]
                    centroid_lat = float(sum(ys) / len(ys))
                m_lat, m_lon = _approx_meters_per_degree(centroid_lat)
                meters2_per_deg2 = m_lat * m_lon
                area_m2 = pixel_count * pixel_area_map_units * meters2_per_deg2
            else:
                # projected CRS: map units assumed meters
                area_m2 = pixel_count * pixel_area_map_units

            feat = {
                "type": "Feature",
                "properties": {
                    "date_before": before_date,
                    "date_after": after_date,
                    "area_m2": float(area_m2),
                    "confidence": 1.0,
                    "pixel_count": int(pixel_count),
                },
                "geometry": geom,
            }
            features.append(feat)

    fc = {"type": "FeatureCollection", "features": features}
    out_path = processed / "change_features.geojson"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(fc, fh)

    print("Wrote", out_path)
    return out_path


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    run_part3_geojson(project_root)
