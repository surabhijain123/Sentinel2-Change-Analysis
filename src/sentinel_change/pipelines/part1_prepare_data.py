from __future__ import annotations

from pathlib import Path
import sys
from typing import Iterable

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sentinel_change.io.raster_stack import (
    stack_bands_by_name,
    write_multiband_geotiff,
    write_rgb_preview_geotiff,
)

REQUIRED_BANDS = ("B04", "B03", "B02")


def build_stack_for_date(date_dir: Path, output_path: Path, bands: Iterable[str] = REQUIRED_BANDS) -> Path:
    stacked, metas = stack_bands_by_name(date_dir, bands)
    write_multiband_geotiff(output_path, stacked, metas[next(iter(bands))])
    # write a visualization-friendly RGB preview (R=band1, G=band2, B=band3)
    preview_path = output_path.with_name(output_path.stem + "_rgb_preview.tif")
    write_rgb_preview_geotiff(preview_path, stacked, metas[next(iter(bands))])
    return output_path


def run_part1_pipeline(base_dir: Path) -> list[Path]:
    data_dir = base_dir / "data"
    processed_dir = data_dir / "processed"

    date_dirs = [
        data_dir / "sentinel2_20230812",
        data_dir / "sentinel2_20230902",
    ]

    outputs: list[Path] = []
    for d in date_dirs:
        if not d.exists():
            raise FileNotFoundError(f"Missing input date folder: {d}")

        output_name = f"{d.name}_stack.tif"
        out_path = processed_dir / output_name
        outputs.append(build_stack_for_date(d, out_path))

    return outputs


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[3]
    result_files = run_part1_pipeline(project_root)
    print("Part 1 complete. Created:")
    for file in result_files:
        print(f" - {file}")
