from pathlib import Path

def run_all_parts(base_dir):
	"""Run parts 1→4 using the existing pipeline scripts.

	Parameters
	- base_dir: path-like pointing at project root (contains `data/`).
	"""
	base = Path(base_dir)

	# lazy imports so importing this module doesn't trigger heavy deps
	from .part1_prepare_data import run_part1_pipeline
	from .part2_change_detection import run_part2_pipeline
	from .part3_change_vectorize import run_part3_geojson
	from .part4_visualize import run_part4_visualize 

	run_part1_pipeline(base)
	run_part2_pipeline(base)
	run_part3_geojson(base)
	run_part4_visualize()

if __name__ == '__main__':
    # Run using current working directory as project root
    run_all_parts(Path.cwd())