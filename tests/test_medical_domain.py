"""Tests for medical modality profiles and prompt guidance."""

from src.agentic.medical_domain import (
    ARTIFACT_TO_METRIC_CATEGORY,
    CANONICAL_DISTORTION_TYPES,
    GENERIC_PROFILE_KEY,
    MODALITY_PROFILES,
    QUALITY_RUBRIC,
    VALID_DISTORTION_TYPES,
    analysis_guidance,
    detection_guidance,
    planner_guidance,
    resolve_profile,
    summarizer_guidance,
    tool_selection_guidance,
)


EXPECTED_PROFILES = {
    "radiography",
    "ct",
    "mri",
    "ultrasound",
    "mammography",
    "fundus_oct",
    "dermatology",
    "histopathology",
    "generic_medical",
}


def test_all_expected_profiles_exist():
    assert set(MODALITY_PROFILES) == EXPECTED_PROFILES
    assert GENERIC_PROFILE_KEY in MODALITY_PROFILES


def test_resolve_known_aliases():
    assert resolve_profile("chest x-ray")["key"] == "radiography"
    assert resolve_profile("CT scan")["key"] == "ct"
    assert resolve_profile("magnetic resonance imaging")["key"] == "mri"
    assert resolve_profile("echocardiography")["key"] == "ultrasound"
    assert resolve_profile("mammogram")["key"] == "mammography"
    assert resolve_profile("fundus photography")["key"] == "fundus_oct"
    assert resolve_profile("dermoscopy")["key"] == "dermatology"
    assert resolve_profile("whole slide imaging")["key"] == "histopathology"


def test_resolve_unknown_falls_back_to_generic():
    assert resolve_profile(None)["key"] == GENERIC_PROFILE_KEY
    assert resolve_profile("")["key"] == GENERIC_PROFILE_KEY
    assert resolve_profile("not a real modality")["key"] == GENERIC_PROFILE_KEY


def test_valid_distortion_types_include_canonical_and_medical_artifacts():
    for distortion in CANONICAL_DISTORTION_TYPES:
        assert distortion in VALID_DISTORTION_TYPES
    for artifact in [
        "Motion artifact",
        "Beam hardening",
        "Speckle",
        "Bias field",
        "Stain variation",
        "Tissue folds",
    ]:
        assert artifact in VALID_DISTORTION_TYPES


def test_artifact_mapping_targets_canonical_categories():
    for artifact, category in ARTIFACT_TO_METRIC_CATEGORY.items():
        assert artifact
        assert category in CANONICAL_DISTORTION_TYPES


def test_rubric_covers_five_levels():
    assert set(QUALITY_RUBRIC) == {1, 2, 3, 4, 5}
    assert "Diagnostic quality" in QUALITY_RUBRIC[5]
    assert "Non-diagnostic" in QUALITY_RUBRIC[1]


def test_guidance_mentions_modality_specific_artifacts():
    ct = resolve_profile("ct")
    assert "Beam hardening" in detection_guidance(ct)
    assert "Preferred metrics" in tool_selection_guidance(ct)
    assert "Diagnostic quality rubric" in analysis_guidance(ct)

    histology = resolve_profile("histopathology")
    assert "Stain variation" in detection_guidance(histology)
    assert "stain" in summarizer_guidance(histology).lower()

    planner = planner_guidance()
    assert "generic_medical" in planner["modality_catalog"]
    assert "motion" in planner["artifact_mapping"].lower()
    assert "spatial distortions" in planner["artifact_mapping"].lower()
    assert "diagnos" in planner["medical_safety"].lower()
