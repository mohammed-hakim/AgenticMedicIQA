"""Classical medical image quality indicators.

These indicators are deliberately simple, explainable, and uncalibrated. They
are computed on the display-rendered image and reported as auxiliary evidence
alongside tool scores. They are not fused into the 1-5 quality score and must
not be presented as clinical measurements.

The module uses only NumPy, OpenCV, and scikit-image, all of which are already
project dependencies.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union

import numpy as np
from PIL import Image

from src.agentic.medical_io import get_display_image

AUXILIARY_NOTE = (
    "Classical indicators are computed on the display-rendered image. They are "
    "uncalibrated, scale-dependent proxies for acquisition quality, not clinical "
    "measurements."
)


def _round(value: float, digits: int = 6) -> Optional[float]:
    if value is None:
        return None
    if not np.isfinite(value):
        return None
    return round(float(value), digits)


def _to_gray(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def _noise_sigma(gray: np.ndarray) -> float:
    try:
        from skimage.restoration import estimate_sigma

        value = float(estimate_sigma(gray, channel_axis=None))
        if np.isfinite(value):
            return value
    except Exception:
        pass

    # Robust fallback based on the median absolute deviation of a Laplacian.
    import cv2

    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    median = float(np.median(lap))
    mad = float(np.median(np.abs(lap - median))) + 1e-8
    return 1.4826 * mad


def _tile_illumination_cv(gray: np.ndarray, tiles: int = 4) -> float:
    height, width = gray.shape
    means: List[float] = []
    for row in range(tiles):
        for col in range(tiles):
            y0 = int(row * height / tiles)
            y1 = int((row + 1) * height / tiles)
            x0 = int(col * width / tiles)
            x1 = int((col + 1) * width / tiles)
            tile = gray[y0:y1, x0:x1]
            if tile.size:
                means.append(float(tile.mean()))
    if not means:
        return 0.0
    mean = float(np.mean(means))
    if mean <= 1e-6:
        return 0.0
    return float(np.std(means) / mean)


def _edge_density(gray: np.ndarray) -> float:
    import cv2

    gray_u8 = (gray * 255.0).clip(0, 255).astype(np.uint8)
    median = float(np.median(gray_u8))
    lower = int(max(0, 0.66 * median))
    upper = int(min(255, 1.33 * median))
    if lower == upper:
        lower, upper = 50, 150
    edges = cv2.Canny(gray_u8, lower, upper)
    return float(np.count_nonzero(edges)) / float(edges.size)


def _build_flags(metrics: Dict[str, float]) -> List[str]:
    flags: List[str] = []

    if metrics["focus_relative"] < 0.035:
        flags.append("low_local_sharpness")
    if metrics["noise_sigma"] > 0.065:
        flags.append("high_noise_estimate")
    if metrics["contrast_rms"] < 0.10 or metrics["contrast_michelson"] < 0.22:
        flags.append("low_contrast")
    if metrics["exposure_clipping_low"] > 0.05 or metrics["exposure_clipping_high"] > 0.05:
        flags.append("exposure_clipping")
    if metrics["illumination_cv"] > 0.30:
        flags.append("uneven_illumination")
    if metrics["entropy_bits"] < 3.5:
        flags.append("low_information_content")
    if metrics["edge_density"] < 0.003:
        flags.append("sparse_edges")

    return flags


def compute_medical_qc(
    image: Union[str, Image.Image],
    max_size: int = 1024,
) -> Dict[str, Any]:
    """Compute classical QC indicators for a medical image.

    Args:
        image: Path to a supported image (DICOM/NIfTI/raster) or a PIL image.
        max_size: Longest edge used for metric computation. Small values keep
            runtime predictable; metrics are therefore relative.

    Returns:
        A JSON-serializable dictionary with indicator values, heuristic flags,
        and a capability note. On failure, ``available`` is False and an
        ``error`` string is returned instead of raising.
    """
    try:
        import cv2
        from skimage.measure import shannon_entropy

        if isinstance(image, Image.Image):
            pil_image = image
        else:
            pil_image = get_display_image(str(image), max_size=max_size)

        gray = _to_gray(pil_image)
        if gray.size == 0:
            raise ValueError("Empty image")

        laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        focus_laplacian = float(laplacian.var())

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        focus_tenengrad = float(np.mean(gx * gx + gy * gy))

        gray_std = float(gray.std())
        focus_relative = focus_laplacian / (gray_std * gray_std + 1e-8)

        noise_sigma = _noise_sigma(gray)
        contrast_rms = gray_std

        low_p, high_p = np.percentile(gray, [1.0, 99.0])
        contrast_michelson = float((high_p - low_p) / (high_p + low_p + 1e-8))

        gray_u8 = (gray * 255.0).clip(0, 255)
        exposure_clipping_low = float(np.mean(gray_u8 <= 2.0))
        exposure_clipping_high = float(np.mean(gray_u8 >= 253.0))

        illumination_cv = _tile_illumination_cv(gray)
        entropy_bits = float(shannon_entropy((gray_u8).astype(np.uint8)))
        edge_density = _edge_density(gray)

        metrics = {
            "focus_laplacian_variance": _round(focus_laplacian),
            "focus_tenengrad": _round(focus_tenengrad),
            "focus_relative": _round(focus_relative),
            "noise_sigma": _round(noise_sigma),
            "contrast_rms": _round(contrast_rms),
            "contrast_michelson": _round(contrast_michelson),
            "exposure_clipping_low": _round(exposure_clipping_low),
            "exposure_clipping_high": _round(exposure_clipping_high),
            "illumination_cv": _round(illumination_cv),
            "entropy_bits": _round(entropy_bits),
            "edge_density": _round(edge_density),
        }

        return {
            "available": True,
            "kind": "classical_medical_qc",
            "metrics": metrics,
            "heuristic_flags": _build_flags(metrics),
            "note": AUXILIARY_NOTE,
        }

    except Exception as exc:
        return {
            "available": False,
            "kind": "classical_medical_qc",
            "error": f"Classical QC indicators unavailable: {exc}",
            "note": AUXILIARY_NOTE,
        }


def format_medical_qc_for_prompt(qc: Optional[Dict[str, Any]]) -> str:
    """Format QC indicators for inclusion in the Summarizer prompt."""
    if not qc:
        return "No classical QC indicators available."

    if not qc.get("available", False):
        return qc.get("error", "Classical QC indicators unavailable.")

    payload = {
        "metrics": qc.get("metrics", {}),
        "heuristic_flags": qc.get("heuristic_flags", []),
        "note": qc.get("note", AUXILIARY_NOTE),
    }
    return json.dumps(payload, indent=2)
