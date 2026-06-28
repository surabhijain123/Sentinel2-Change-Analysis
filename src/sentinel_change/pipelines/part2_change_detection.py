from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import rasterio

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel_change.change_detection.sam import compute_sam_change


def _read_multiband(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        array = src.read()
        profile = src.profile.copy()
    return array, profile


def _write_single_band(path: Path, array: np.ndarray, profile: dict, dtype: str, nodata: int | float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(
        count=1,
        dtype=dtype,
        compress="deflate",
        tiled=True,
    )

    if nodata is not None:
        out_profile["nodata"] = nodata
    elif "nodata" in out_profile:
        out_profile.pop("nodata", None)

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array, 1)


def _write_rgba_overlay(path: Path, mask_uint8: np.ndarray, profile: dict) -> None:
    """Write an RGBA overlay where mask_uint8 is 0/255 for alpha."""
    path.parent.mkdir(parents=True, exist_ok=True)

    out_profile = profile.copy()
    out_profile.update(
        count=4,
        dtype="uint8",
        compress="deflate",
        tiled=True,
    )

    # build RGBA: red overlay where mask is 255
    h, w = mask_uint8.shape
    rgba = np.zeros((4, h, w), dtype=np.uint8)
    rgba[0] = mask_uint8  # R channel
    rgba[1] = 0
    rgba[2] = 0
    rgba[3] = mask_uint8  # Alpha

    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(rgba)


def run_part2_pipeline(base_dir: Path, threshold_method: str = "percentile") -> dict[str, Path]:
    processed_dir = base_dir / "data" / "processed"
    before_path = processed_dir / "sentinel2_20230812_stack.tif"
    after_path = processed_dir / "sentinel2_20230902_stack.tif"

    if not before_path.exists() or not after_path.exists():
        raise FileNotFoundError(
            "Stacked inputs not found. Run Part 1 pipeline first to create the date stack GeoTIFFs."
        )

    before, profile_before = _read_multiband(before_path)
    after, _ = _read_multiband(after_path)

    # compute per-pixel vector norms to identify valid overlap
    before_f = before.astype(np.float32)
    after_f = after.astype(np.float32)
    norm_before = np.linalg.norm(before_f, axis=0)
    norm_after = np.linalg.norm(after_f, axis=0)

    # define a small reflectance floor to treat padded/empty pixels as invalid
    reflectance_floor = 100
    valid_mask = (norm_before > reflectance_floor) & (norm_after > reflectance_floor)

    # mask-out invalid pixels by setting them to NaN so SAM will ignore them
    before_masked = before_f.copy()
    after_masked = after_f.copy()
    before_masked[:, ~valid_mask] = np.nan
    after_masked[:, ~valid_mask] = np.nan

    result = compute_sam_change(before_masked, after_masked, threshold_method=threshold_method)

    # apply the valid mask to outputs: ensure invalid pixels are marked as no-change
    angle_norm = result.angle_normalized
    # set masked pixels to 0 for visualization (they are not changed)
    angle_norm[~valid_mask] = 0.0
    binary_vis = (result.binary_map.astype(np.uint8) * 255).astype(np.uint8)

    try:
        from scipy import ndimage

        min_pixels = 25  # MMU: 25 pixels (~2500 m² at 10 m resolution)
        labeled, num_features = ndimage.label(binary_vis == 255)
        component_sizes = np.bincount(labeled.ravel())
        small_components = component_sizes < min_pixels
        small_components[0] = False  # don't remove background
        noise_mask = small_components[labeled]
        binary_vis[noise_mask] = 0
    except ImportError:
        print("Scipy not installed, skipping MMU filter.")
        pass  # scipy not installed, skip MMU filter

    binary_vis[~valid_mask] = 0

    # morphological closing to merge nearby detections and reduce salt-and-pepper
    try:
        from scipy import ndimage

        # binary image as boolean
        b = (binary_vis == 255)
        # closing: dilate then erode with a small structuring element
        structure = np.ones((3, 3), dtype=bool)
        b_closed = ndimage.binary_closing(b, structure=structure)

        # relabel and keep only top-N largest components
        labeled2, n2 = ndimage.label(b_closed)
        if n2 > 0:
            sizes = np.bincount(labeled2.ravel())
            sizes[0] = 0
            # indices of largest components (descending)
            top_n = 100
            largest_idx = np.argsort(sizes)[-top_n:][::-1]
            keep_mask = np.isin(labeled2, largest_idx)
            binary_vis = (keep_mask.astype(np.uint8) * 255).astype(np.uint8)
        else:
            binary_vis = np.zeros_like(binary_vis)
    except ImportError:
        # no scipy: attempt a crude bbox-based reduction of large regions (slow)
        print("Scipy not installed, skipping morphological closing/top-N reduction.")
        pass

    change_map_path = processed_dir / "change_map_sam.tif"
    change_binary_path = processed_dir / "change_binary_sam.tif"
    change_overlay_path = processed_dir / "change_overlay_sam.tif"

    angle_scaled_uint8 = (angle_norm * 255.0).round().astype(np.uint8)
    # do not set nodata to 0: keep full valid range and avoid viewers treating 0 as nodata
    _write_single_band(change_map_path, angle_scaled_uint8, profile_before, dtype="uint8", nodata=None)
    _write_single_band(change_binary_path, binary_vis, profile_before, dtype="uint8", nodata=None)
    _write_rgba_overlay(change_overlay_path, binary_vis, profile_before)

    return {
        "change_map": change_map_path,
        "change_binary": change_binary_path,
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    output_files = run_part2_pipeline(project_root, threshold_method="percentile")
    print("Part 2 complete. Created:")
    for k, v in output_files.items():
        print(f" - {k}: {v}")
