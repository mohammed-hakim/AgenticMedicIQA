"""Tests for classical medical QC indicators."""

import json

import numpy as np
import pytest
from PIL import Image, ImageFilter

from src.agentic.medical_metrics import (
    AUXILIARY_NOTE,
    compute_medical_qc,
    format_medical_qc_for_prompt,
)


@pytest.fixture
def checkerboard():
    size = 256
    block = 16
    array = np.zeros((size, size), dtype=np.uint8)
    for row in range(0, size, block):
        for col in range(0, size, block):
            if ((row // block) + (col // block)) % 2 == 0:
                array[row:row + block, col:col + block] = 255
    return Image.fromarray(array, mode="L").convert("RGB")


def test_metrics_are_available_and_json_serializable(checkerboard):
    result = compute_medical_qc(checkerboard)
    assert result["available"] is True
    assert result["kind"] == "classical_medical_qc"
    assert result["note"] == AUXILIARY_NOTE
    assert isinstance(result["heuristic_flags"], list)

    metrics = result["metrics"]
    expected = {
        "focus_laplacian_variance",
        "focus_tenengrad",
        "focus_relative",
        "noise_sigma",
        "contrast_rms",
        "contrast_michelson",
        "exposure_clipping_low",
        "exposure_clipping_high",
        "illumination_cv",
        "entropy_bits",
        "edge_density",
    }
    assert expected <= set(metrics)
    json.dumps(result)


def test_blur_reduces_focus_indicator(checkerboard):
    blurred = checkerboard.filter(ImageFilter.GaussianBlur(radius=4))
    sharp_qc = compute_medical_qc(checkerboard)
    blurred_qc = compute_medical_qc(blurred)

    assert (
        sharp_qc["metrics"]["focus_laplacian_variance"]
        > blurred_qc["metrics"]["focus_laplacian_variance"]
    )
    assert (
        sharp_qc["metrics"]["focus_tenengrad"]
        > blurred_qc["metrics"]["focus_tenengrad"]
    )


def test_exposure_clipping_flag():
    array = np.zeros((64, 64, 3), dtype=np.uint8)
    array[:32, :, :] = 255
    image = Image.fromarray(array, mode="RGB")

    qc = compute_medical_qc(image)
    assert qc["metrics"]["exposure_clipping_low"] > 0.4
    assert qc["metrics"]["exposure_clipping_high"] > 0.4
    assert "exposure_clipping" in qc["heuristic_flags"]


def test_low_contrast_flag():
    array = np.full((64, 64, 3), 128, dtype=np.uint8)
    image = Image.fromarray(array, mode="RGB")

    qc = compute_medical_qc(image)
    assert "low_contrast" in qc["heuristic_flags"]
    assert "low_information_content" in qc["heuristic_flags"]


def test_format_for_prompt_handles_missing_and_unavailable():
    assert "No classical QC indicators" in format_medical_qc_for_prompt(None)
    unavailable = {"available": False, "error": "boom"}
    assert "boom" in format_medical_qc_for_prompt(unavailable)


def test_format_for_prompt_is_json_backed(checkerboard):
    qc = compute_medical_qc(checkerboard)
    text = format_medical_qc_for_prompt(qc)
    payload = json.loads(text)
    assert "metrics" in payload
    assert "heuristic_flags" in payload
    assert "uncalibrated" in payload["note"].lower()


def test_path_input_uses_display_rendering(tmp_path):
    path = tmp_path / "image.png"
    checker = np.zeros((64, 64), dtype=np.uint8)
    checker[::2, ::2] = 255
    Image.fromarray(checker, mode="L").save(path)

    result = compute_medical_qc(str(path))
    assert result["available"] is True
    assert result["metrics"]["edge_density"] > 0.0


def test_metric_failure_returns_available_false():
    result = compute_medical_qc("not-a-real-image.png")
    assert result["available"] is False
    assert "error" in result
