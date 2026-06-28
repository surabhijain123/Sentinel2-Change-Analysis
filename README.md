# Sentinel‑2 Change Detection

This repository contains a Sentinel‑2 change detection pipeline split into parts:

- Part 1: stack single-band TIFFs into 3‑band true‑color GeoTIFFs and write RGB previews
- Part 2: compute Spectral Angle Mapper (SAM) change, threshold, and postprocess
- Part 3: vectorize binary change raster to GeoJSON
- Part 4: produce a simple interactive HTML map with the RGB base and polygon overlays

This README explains how to set up the environment and run the pipeline on Windows PowerShell.

## Prerequisites

- Windows or any OS with a Python 3.10+ installation (this project used Python 3.14 in development).
- GDAL/rasterio binary dependencies. On Windows it's easiest to use the prebuilt wheels or a conda environment.

## Quick setup (PowerShell)

Open PowerShell in the project root (where `src/` is located). If you cloned the repository, create a local virtual environment for your machine; example commands:

```powershell
# create a venv (if you don't already have one)
python -m venv .venv

# activate the venv (PowerShell)
.\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt
```

If activation prints a `Exit-CondaEnvironment` warning but the prompt shows `(.venv)`, you can continue; otherwise call the venv python explicitly as explained below.

## Install notes (GDAL/rasterio)

On Windows the simplest approach is to install rasterio from wheels that match your Python version. If you hit issues installing `rasterio`, consider creating a conda environment and installing `gdal`/`rasterio` via conda-forge.

## Run the pipeline

Recommended: run the pipeline package as a module using the venv python so package imports work correctly.

PowerShell (recommended):
```powershell
# run all parts (discovery runner will execute part*.py in order)
& ".\.venv\Scripts\python.exe" -m src.sentinel_change.pipelines
```

Alternative: run a specific part script directly (useful for debugging):
```powershell
# Part 1 (prepare stacks)
& ".\.venv\Scripts\python.exe" "src\sentinel_change\pipelines\part1_prepare_data.py"

# Part 2 (change detection)
& ".\.venv\Scripts\python.exe" "src\sentinel_change\pipelines\part2_change_detection.py"

# Part 3 (vectorize)
& ".\.venv\Scripts\python.exe" "src\sentinel_change\pipelines\part3_change_vectorize.py"

# Part 4 (visualize)
& ".\.venv\Scripts\python.exe" "src\sentinel_change\pipelines\part4_visualize.py"
```

## Adjusting parameters

If you want to tune detection sensitivity, edit `src/sentinel_change/pipelines/part2_change_detection.py`:
- `reflectance_floor` — excludes low-reflectance (padded/empty) pixels from consideration
- `min_pixels` — MMU (minimum mapping unit) removing small components
- `top_n` — number of largest components to keep after closing

After editing, rerun Part 2 (or run the full pipeline with `-m`).

## Outputs

Produced files are written to `data/processed/`, for example:
- `*_stack.tif` — stacked GeoTIFFs for each date
- `*_rgb_preview.tif` / `.png` — RGB previews
- `change_map_sam.tif` — scaled SAM angle map
- `change_binary_sam.tif` — binary change mask (0/255)
- `change_overlay_sam.tif` — RGBA overlay
- `change_features.geojson` — vectorized polygons
- `part4_map.html` — interactive HTML map

## License and notes

This code is intended for educational/assessment use. Modify parameters to match your study area's scale and sensor characteristics.
