from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SamResult:
    angle_map: np.ndarray
    angle_normalized: np.ndarray
    threshold: float
    binary_map: np.ndarray


def _safe_normalize_01(values: np.ndarray) -> np.ndarray:
    v_min = np.nanmin(values)
    v_max = np.nanmax(values)
    if not np.isfinite(v_min) or not np.isfinite(v_max) or v_max <= v_min:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - v_min) / (v_max - v_min)).astype(np.float32)


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return 0.0

    v_min = float(finite_values.min())
    v_max = float(finite_values.max())
    if v_max <= v_min:
        return v_min

    hist, bin_edges = np.histogram(finite_values, bins=bins, range=(v_min, v_max))
    hist = hist.astype(np.float64)

    prob = hist / hist.sum()
    omega = np.cumsum(prob)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    mu = np.cumsum(prob * bin_centers)
    mu_t = mu[-1]

    denominator = omega * (1.0 - omega)
    denominator[denominator == 0] = np.nan
    sigma_b2 = ((mu_t * omega - mu) ** 2) / denominator

    idx = int(np.nanargmax(sigma_b2)) if np.any(np.isfinite(sigma_b2)) else bins // 2
    return float(bin_centers[idx])


def compute_sam_change(
    image_before: np.ndarray,
    image_after: np.ndarray,
    threshold_method: str = "otsu",
    percentile: float = 99.0,
) -> SamResult:
    if image_before.shape != image_after.shape:
        raise ValueError("Input images must have the same shape.")
    if image_before.ndim != 3:
        raise ValueError("Input images must have shape (bands, height, width).")

    before = image_before.astype(np.float32)
    after = image_after.astype(np.float32)

    dot = np.sum(before * after, axis=0)
    norm_before = np.linalg.norm(before, axis=0)
    norm_after = np.linalg.norm(after, axis=0)

    eps = 1e-8
    cosine = dot / np.maximum(norm_before * norm_after, eps)
    cosine = np.clip(cosine, -1.0, 1.0)

    angle = np.arccos(cosine).astype(np.float32)
    angle_normalized = _safe_normalize_01(angle)

    method = threshold_method.lower().strip()
    if method == "otsu":
        threshold = otsu_threshold(angle_normalized)
    elif method == "percentile":
        threshold = float(np.percentile(angle_normalized[np.isfinite(angle_normalized)], percentile))
    else:
        raise ValueError("threshold_method must be one of: 'otsu', 'percentile'.")

    binary_map = (angle_normalized >= threshold).astype(np.uint8)

    return SamResult(
        angle_map=angle,
        angle_normalized=angle_normalized,
        threshold=threshold,
        binary_map=binary_map,
    )
