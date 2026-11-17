"""Medical domain profiles for Medical AgenticIQA.

This module is the single source of truth for modality-specific guidance used by
the Planner, Executor, and Summarizer prompts. It intentionally contains data
and small helpers only, so adding a new modality does not require changing the
graph or the API contracts.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

DOMAIN_NAME = "Medical AgenticIQA"

GENERIC_PROFILE_KEY = "generic_medical"

DISCLAIMER = (
    "This system assesses image acquisition quality and diagnostic usability only. "
    "It does not provide diagnoses, clinical findings, or treatment advice, and it is "
    "not a medical device. Tool scores are uncalibrated proxies, not clinical measurements."
)

MEDICAL_SAFETY_RULES = (
    "Assess image quality, acquisition artifacts, and diagnostic usability only. "
    "Do not provide diagnoses, clinical interpretations, or treatment recommendations. "
    "If the image is non-diagnostic, state that repeat acquisition or review by a "
    "qualified clinician is recommended. Treat all tool scores as relative, "
    "uncalibrated proxies rather than clinical measurements."
)

CANONICAL_DISTORTION_TYPES: List[str] = [
    "Blurs",
    "Color distortions",
    "Compression",
    "Noise",
    "Brightness change",
    "Sharpness",
    "Contrast",
    "Spatial distortions",
]

QUALITY_RUBRIC: Dict[int, str] = {
    5: "Diagnostic quality: no artifact that limits interpretation.",
    4: "Good: minor artifact unlikely to affect interpretation.",
    3: "Fair: moderate artifact may reduce confidence in some regions.",
    2: "Poor: significant artifact substantially limits interpretation.",
    1: "Non-diagnostic: severe artifact; repeat acquisition or expert review recommended.",
}

# Maps modality-specific artifact names to the broader categories that the
# bundled IQA tools can score. The Executor may report either the precise
# artifact name or the canonical category.
ARTIFACT_TO_METRIC_CATEGORY: Dict[str, str] = {
    "Motion artifact": "Blurs",
    "Out-of-focus": "Sharpness",
    "Blurs": "Blurs",
    "Noise": "Noise",
    "Speckle": "Noise",
    "Metal artifact": "Spatial distortions",
    "Ring artifact": "Spatial distortions",
    "Aliasing": "Spatial distortions",
    "Ghosting": "Spatial distortions",
    "Truncation": "Spatial distortions",
    "Reverberation": "Spatial distortions",
    "Beam hardening": "Contrast",
    "Shadowing": "Contrast",
    "Contrast": "Contrast",
    "Bias field": "Brightness change",
    "Non-uniform illumination": "Brightness change",
    "Exposure clipping": "Brightness change",
    "Brightness change": "Brightness change",
    "Low resolution": "Sharpness",
    "Sharpness": "Sharpness",
    "Compression": "Compression",
    "Stain variation": "Color distortions",
    "Color cast": "Color distortions",
    "Color distortions": "Color distortions",
    "Positioning": "Spatial distortions",
    "Cropping": "Spatial distortions",
    "Tissue folds": "Spatial distortions",
    "Bubbles": "Spatial distortions",
    "Hair artifact": "Spatial distortions",
}

MODALITY_PROFILES: Dict[str, Dict[str, Any]] = {
    "radiography": {
        "display_name": "Radiography (X-ray, DXA, fluoroscopy)",
        "aliases": [
            "x ray",
            "xray",
            "radiograph",
            "radiography",
            "plain film",
            "chest x ray",
            "chest radiograph",
            "dxa",
            "fluoroscopy",
        ],
        "artifacts": [
            "Motion artifact",
            "Blurs",
            "Noise",
            "Contrast",
            "Brightness change",
            "Exposure clipping",
            "Spatial distortions",
            "Metal artifact",
            "Positioning",
            "Cropping",
        ],
        "criteria": [
            "No motion or double contours",
            "Adequate exposure without clipping",
            "Acceptable contrast and noise for the anatomy",
            "Correct positioning and collimation",
            "No metal or grid artifact obscuring anatomy",
        ],
        "preferred_tools": [
            "MUSIQ",
            "MANIQA",
            "BRISQUE",
            "NIQE",
            "CLIPIQA",
            "TOPIQ_FR",
            "SSIM",
            "GMSD",
        ],
    },
    "ct": {
        "display_name": "Computed tomography (CT)",
        "aliases": [
            "ct",
            "computed tomography",
            "cat scan",
            "ct scan",
        ],
        "artifacts": [
            "Noise",
            "Motion artifact",
            "Metal artifact",
            "Ring artifact",
            "Beam hardening",
            "Truncation",
            "Contrast",
            "Spatial distortions",
            "Low resolution",
        ],
        "criteria": [
            "Noise is acceptable for the acquisition protocol",
            "No severe motion artifact",
            "No metal, ring, or beam-hardening artifact obscuring anatomy",
            "No truncation of the reconstructed field of view",
            "Adequate contrast and windowing",
        ],
        "preferred_tools": [
            "MUSIQ",
            "BRISQUE",
            "NIQE",
            "MANIQA",
            "CLIPIQA",
            "TOPIQ_FR",
            "SSIM",
            "GMSD",
        ],
    },
    "mri": {
        "display_name": "Magnetic resonance imaging (MRI)",
        "aliases": [
            "mri",
            "mr",
            "magnetic resonance",
            "magnetic resonance imaging",
        ],
        "artifacts": [
            "Motion artifact",
            "Ghosting",
            "Aliasing",
            "Noise",
            "Bias field",
            "Spatial distortions",
            "Contrast",
            "Cropping",
        ],
        "criteria": [
            "No motion or ghosting",
            "No aliasing or wrap artifact",
            "Adequate signal-to-noise ratio",
            "Bias field does not obscure anatomy",
            "Full anatomical coverage without cropping",
        ],
        "preferred_tools": [
            "MUSIQ",
            "NIQE",
            "BRISQUE",
            "CLIPIQA",
            "TOPIQ_FR",
            "SSIM",
            "LPIPS",
            "DISTS",
        ],
    },
    "ultrasound": {
        "display_name": "Ultrasound",
        "aliases": [
            "ultrasound",
            "us",
            "sonography",
            "echo",
            "echocardiography",
        ],
        "artifacts": [
            "Speckle",
            "Shadowing",
            "Reverberation",
            "Noise",
            "Blurs",
            "Contrast",
            "Brightness change",
            "Cropping",
        ],
        "criteria": [
            "Adequate gain and depth settings",
            "No obscuring acoustic shadowing or reverberation",
            "Acceptable speckle and noise for the exam",
            "Complete field of view without cropping",
        ],
        "preferred_tools": [
            "MUSIQ",
            "CLIPIQA",
            "BRISQUE",
            "NIQE",
            "TOPIQ_FR",
            "SSIM",
            "LPIPS",
        ],
    },
    "mammography": {
        "display_name": "Mammography / breast imaging",
        "aliases": [
            "mammography",
            "mammogram",
            "breast imaging",
            "dbt",
            "tomosynthesis",
        ],
        "artifacts": [
            "Motion artifact",
            "Positioning",
            "Cropping",
            "Contrast",
            "Noise",
            "Exposure clipping",
            "Metal artifact",
            "Blurs",
        ],
        "criteria": [
            "Correct positioning with adequate compression",
            "No motion artifact",
            "Skin line included without cropping",
            "Adequate contrast and noise",
        ],
        "preferred_tools": [
            "MUSIQ",
            "MANIQA",
            "BRISQUE",
            "NIQE",
            "CLIPIQA",
            "TOPIQ_FR",
            "SSIM",
            "GMSD",
        ],
    },
    "fundus_oct": {
        "display_name": "Fundus photography / OCT",
        "aliases": [
            "fundus",
            "fundus photography",
            "oct",
            "optical coherence tomography",
            "retinal imaging",
            "retina",
        ],
        "artifacts": [
            "Blurs",
            "Non-uniform illumination",
            "Color cast",
            "Noise",
            "Shadowing",
            "Motion artifact",
            "Cropping",
            "Exposure clipping",
        ],
        "criteria": [
            "Focused image",
            "Even illumination without obscuring reflections",
            "No shadowing that hides the region of interest",
            "Adequate field of view",
            "Consistent color rendering",
        ],
        "preferred_tools": [
            "CLIPIQA",
            "MUSIQ",
            "NIQE",
            "MANIQA",
            "TOPIQ_FR",
            "LPIPS",
            "SSIM",
        ],
    },
    "dermatology": {
        "display_name": "Dermatology / dermoscopy",
        "aliases": [
            "dermatology",
            "skin",
            "dermoscopy",
            "dermatoscopic",
            "wound photography",
        ],
        "artifacts": [
            "Blurs",
            "Color cast",
            "Non-uniform illumination",
            "Exposure clipping",
            "Noise",
            "Hair artifact",
            "Cropping",
            "Compression",
        ],
        "criteria": [
            "Sharp focus on the skin surface",
            "Consistent lighting and color balance",
            "No hair or glare obscuring the lesion",
            "Lesion fully within the frame",
        ],
        "preferred_tools": [
            "MUSIQ",
            "CLIPIQA",
            "MANIQA",
            "BRISQUE",
            "TOPIQ_FR",
            "LPIPS",
            "SSIM",
        ],
    },
    "histopathology": {
        "display_name": "Histopathology / microscopy",
        "aliases": [
            "histopathology",
            "histology",
            "pathology",
            "wsi",
            "whole slide imaging",
            "microscopy",
            "h and e",
        ],
        "artifacts": [
            "Blurs",
            "Stain variation",
            "Color cast",
            "Tissue folds",
            "Bubbles",
            "Noise",
            "Compression",
            "Cropping",
            "Low resolution",
        ],
        "criteria": [
            "Sharp focus at the intended magnification",
            "Consistent stain and color balance",
            "No tissue folds or bubbles obscuring structures",
            "Adequate resolution for the structure of interest",
            "Tissue not cut off by cropping",
        ],
        "preferred_tools": [
            "BRISQUE",
            "NIQE",
            "CLIPIQA",
            "MUSIQ",
            "TOPIQ_FR",
            "SSIM",
            "LPIPS",
        ],
    },
    "generic_medical": {
        "display_name": "Generic medical image",
        "aliases": [
            "generic",
            "unknown",
            "medical",
            "other",
        ],
        "artifacts": [
            "Blurs",
            "Noise",
            "Contrast",
            "Brightness change",
            "Spatial distortions",
            "Compression",
            "Sharpness",
            "Color distortions",
            "Motion artifact",
            "Cropping",
        ],
        "criteria": [
            "Image is sharp enough for the intended anatomy",
            "Adequate contrast and exposure",
            "No severe noise or compression artifact",
            "No obvious acquisition artifact obscuring the region of interest",
            "Complete field of view",
        ],
        "preferred_tools": [
            "MUSIQ",
            "BRISQUE",
            "NIQE",
            "CLIPIQA",
            "MANIQA",
            "TOPIQ_FR",
            "SSIM",
            "GMSD",
        ],
    },
}

ALL_MEDICAL_ARTIFACTS: List[str] = sorted(
    {
        artifact
        for profile in MODALITY_PROFILES.values()
        for artifact in profile["artifacts"]
    }
)

VALID_DISTORTION_TYPES = set(CANONICAL_DISTORTION_TYPES) | set(ALL_MEDICAL_ARTIFACTS)


def _normalize_label(value: str) -> str:
    """Normalize a modality label for matching."""
    return re.sub(r"[^a-z0-9]+", " ", value.strip().lower()).strip()


def profile_catalog_text() -> str:
    """Return a compact catalog of available modality profiles."""
    lines = []
    for key, profile in MODALITY_PROFILES.items():
        lines.append(f"- {key}: {profile['display_name']}")
    return "\n".join(lines)


def resolve_profile(modality: Optional[str]) -> Dict[str, Any]:
    """Resolve a free-text modality label to a profile with a 'key' field.

    Unknown or ambiguous labels fall back to the generic medical profile.
    """
    if not modality:
        profile = dict(MODALITY_PROFILES[GENERIC_PROFILE_KEY])
        profile["key"] = GENERIC_PROFILE_KEY
        return profile

    value = _normalize_label(modality)
    value_tokens = set(value.split())

    for key, profile in MODALITY_PROFILES.items():
        candidates = [key, profile["display_name"], *profile.get("aliases", [])]
        for candidate in candidates:
            normalized = _normalize_label(candidate)
            if value == normalized:
                resolved = dict(profile)
                resolved["key"] = key
                return resolved
            candidate_tokens = set(normalized.split())
            if candidate_tokens and candidate_tokens.issubset(value_tokens):
                resolved = dict(profile)
                resolved["key"] = key
                return resolved

    profile = dict(MODALITY_PROFILES[GENERIC_PROFILE_KEY])
    profile["key"] = GENERIC_PROFILE_KEY
    return profile


def _artifact_mapping_lines(profile: Dict[str, Any]) -> str:
    lines = []
    for artifact in profile["artifacts"]:
        category = ARTIFACT_TO_METRIC_CATEGORY.get(artifact)
        if category:
            lines.append(f"- {artifact} -> {category}")
    return "\n".join(lines)


def planner_guidance() -> Dict[str, str]:
    """Return prompt snippets for the Planner."""
    return {
        "modality_catalog": profile_catalog_text(),
        "artifact_mapping": (
            "Map descriptive terms to these standard distortion categories:\n"
            "- blur, out of focus, motion -> Blurs\n"
            "- noise, grain, speckle -> Noise\n"
            "- metal, ring, streak, aliasing, ghosting, artifact -> Spatial distortions\n"
            "- dark, bright, exposure, clipping, window -> Brightness change\n"
            "- contrast, washed out, beam hardening -> Contrast\n"
            "- compressed, JPEG, blocky -> Compression\n"
            "- pixelated, low resolution, soft -> Sharpness\n"
            "- color, stain, tint -> Color distortions"
        ),
        "medical_safety": MEDICAL_SAFETY_RULES,
    }


def detection_guidance(profile: Dict[str, Any]) -> str:
    """Return prompt guidance for distortion detection."""
    artifacts = ", ".join(profile["artifacts"])
    categories = ", ".join(CANONICAL_DISTORTION_TYPES)
    return (
        f"Modality profile: {profile['display_name']}\n"
        f"Modality-specific artifacts to consider: {artifacts}\n"
        f"Canonical distortion categories: {categories}\n"
        "Use the precise artifact name when it is visible; otherwise use the most "
        "appropriate canonical category. Do not infer artifacts that are not present."
    )


def analysis_guidance(profile: Dict[str, Any]) -> str:
    """Return prompt guidance for distortion severity analysis."""
    criteria = "\n".join(f"- {item}" for item in profile["criteria"])
    rubric = "\n".join(f"- {score}: {text}" for score, text in QUALITY_RUBRIC.items())
    return (
        f"Modality profile: {profile['display_name']}\n"
        f"Modality-specific quality criteria:\n{criteria}\n"
        f"Diagnostic quality rubric:\n{rubric}"
    )


def tool_selection_guidance(profile: Dict[str, Any]) -> str:
    """Return prompt guidance for tool selection."""
    preferred = ", ".join(profile["preferred_tools"])
    mapping = _artifact_mapping_lines(profile)
    return (
        f"Modality profile: {profile['display_name']}\n"
        f"Preferred metrics for this modality: {preferred}\n"
        f"Artifact-to-metric hints:\n{mapping}\n"
        "The bundled metrics were developed or calibrated on natural images. "
        "Use them as relative acquisition-quality proxies, not as clinical measurements."
    )


def summarizer_guidance(profile: Dict[str, Any]) -> str:
    """Return prompt guidance for the Summarizer."""
    criteria = "; ".join(profile["criteria"])
    rubric = "; ".join(f"{score} = {text}" for score, text in QUALITY_RUBRIC.items())
    return (
        f"Modality profile: {profile['display_name']}. "
        f"Quality criteria: {criteria}. "
        f"Diagnostic quality rubric: {rubric}. "
        "Report the modality, anatomical region, artifact severity, and diagnostic "
        "usability. Do not provide a diagnosis or treatment recommendation."
    )
