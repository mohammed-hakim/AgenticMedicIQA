"""Tests for medical prompt wiring and state fields."""

import json

from src.agentic.medical_domain import resolve_profile
from src.agentic.nodes.executor import (
    DISTORTION_ANALYSIS_PROMPT_TEMPLATE,
    DISTORTION_DETECTION_PROMPT_TEMPLATE,
    TOOL_SELECTION_PROMPT_TEMPLATE,
    distortion_analysis_subtask,
    distortion_detection_subtask,
)
from src.agentic.nodes.planner import construct_planner_prompt
from src.agentic.nodes.summarizer import (
    MEDICAL_UNIFIED_PROMPT_TEMPLATE,
    PAPER_UNIFIED_PROMPT_TEMPLATE,
    format_medical_context,
)
from src.agentic.state import (
    DistortionAnalysis,
    ExecutorOutput,
    PlanControlFlags,
    PlannerOutput,
)


def _plan(modality="radiography"):
    return PlannerOutput(
        task_type="IQA",
        reference_type="No-Reference",
        required_object_names=["chest"],
        required_distortions={"chest": ["Motion artifact"]},
        required_tools=None,
        distortion_source="explicit",
        modality=modality,
        plan=PlanControlFlags(
            distortion_detection=False,
            distortion_analysis=True,
            tool_selection=False,
            tool_execute=False,
        ),
    )


class TestPlannerPrompt:
    def test_prompt_contains_profile_catalog_and_safety(self):
        prompt = construct_planner_prompt(
            "Is this chest x-ray adequate?",
            has_reference=False,
            image_metadata={
                "source_format": "dicom",
                "modality_code": "DX",
                "body_part": "CHEST",
                "modality_hint": "radiography",
            },
        )
        assert "Available modality profiles" in prompt
        assert "generic_medical" in prompt
        assert "DICOM modality code: DX" in prompt
        assert "suggests profile: radiography" in prompt
        assert "Do not provide diagnoses" in prompt

    def test_prompt_keeps_reference_note(self):
        prompt = construct_planner_prompt("Compare", has_reference=True)
        assert "reference image is also provided" in prompt


class TestMedicalState:
    def test_modality_is_optional_and_serialized(self):
        plan = _plan("ct")
        assert plan.modality == "ct"
        assert '"modality":"ct"' in plan.model_dump_json().replace(" ", "")

        default_plan = _plan()
        default_plan.modality = None
        assert default_plan.modality is None

    def test_medical_distortion_type_is_accepted(self):
        analysis = DistortionAnalysis(
            type="Beam hardening",
            severity="moderate",
            explanation="Streak artifact between dense structures.",
        )
        assert analysis.type == "Beam hardening"

    def test_executor_output_accepts_medical_qc(self):
        output = ExecutorOutput(
            medical_qc={
                "available": True,
                "kind": "classical_medical_qc",
                "metrics": {"focus_relative": 0.04},
                "heuristic_flags": [],
            }
        )
        assert output.medical_qc["available"] is True


class TestMedicalSummarizerPrompt:
    def test_template_placeholders_and_format(self):
        for placeholder in [
            "{query}",
            "{distortion_analysis}",
            "{tool_scores}",
            "{reference_type}",
            "{prior_answer}",
            "{medical_context}",
            "<image>",
        ]:
            assert placeholder in MEDICAL_UNIFIED_PROMPT_TEMPLATE

        prompt = MEDICAL_UNIFIED_PROMPT_TEMPLATE.format(
            query="Is this image diagnostic?",
            distortion_analysis="{}",
            tool_scores="{}",
            reference_type="No-Reference",
            prior_answer="This is the first iteration",
            medical_context="Modality profile: radiography",
        )
        assert "medical image quality summarizer" in prompt
        assert "Modality profile: radiography" in prompt
        assert "<image>" in prompt

    def test_paper_template_is_unchanged_for_compatibility(self):
        assert "{medical_context}" not in PAPER_UNIFIED_PROMPT_TEMPLATE
        assert "{query}" in PAPER_UNIFIED_PROMPT_TEMPLATE

    def test_format_medical_context_includes_profile_metadata_and_qc(self):
        plan = _plan("histopathology")
        executor_output = ExecutorOutput(
            medical_qc={
                "available": True,
                "kind": "classical_medical_qc",
                "metrics": {"focus_relative": 0.012},
                "heuristic_flags": ["low_local_sharpness"],
                "source": {
                    "source_format": "dicom",
                    "modality_code": "SM",
                    "rows": 512,
                    "columns": 512,
                    "frames": 1,
                    "body_part": "TISSUE",
                },
                "note": "uncalibrated",
            }
        )

        text = format_medical_context(plan, executor_output)
        assert "Histopathology" in text
        assert "DICOM modality code: SM" in text
        assert "matrix: 512x512" in text
        assert "low_local_sharpness" in text
        assert "Do not provide a diagnosis" in text
        assert "PatientName" not in text


class TestExecutorPromptWiring:
    def test_detection_prompt_uses_modality_artifacts(self, monkeypatch):
        captured = {}

        class FakeVLM:
            def generate(self, prompt, images, **kwargs):
                captured["prompt"] = prompt
                return json.dumps({"distortion_set": {"Global": ["Noise"]}})

        result = distortion_detection_subtask(
            "Check quality",
            [],
            FakeVLM(),
            profile=resolve_profile("ct"),
        )
        assert result == {"Global": ["Noise"]}
        assert "Beam hardening" in captured["prompt"]
        assert "Canonical distortion categories" in captured["prompt"]
        assert "{medical_guidance}" not in captured["prompt"]

    def test_analysis_prompt_uses_modality_rubric(self):
        captured = {}

        class FakeVLM:
            def generate(self, prompt, images, **kwargs):
                captured["prompt"] = prompt
                return json.dumps(
                    {
                        "distortion_analysis": {
                            "Global": [
                                {
                                    "type": "Noise",
                                    "severity": "slight",
                                    "explanation": "Mild noise.",
                                }
                            ]
                        }
                    }
                )

        result = distortion_analysis_subtask(
            "Check quality",
            [],
            {"Global": ["Noise"]},
            FakeVLM(),
            profile=resolve_profile("mri"),
        )
        assert result["Global"][0].type == "Noise"
        assert "Bias field" in captured["prompt"]
        assert "Diagnostic quality rubric" in captured["prompt"]

    def test_prompts_have_no_unfilled_medical_placeholder(self):
        for template in [
            DISTORTION_DETECTION_PROMPT_TEMPLATE,
            DISTORTION_ANALYSIS_PROMPT_TEMPLATE,
            TOOL_SELECTION_PROMPT_TEMPLATE,
        ]:
            assert "{medical_guidance}" in template
