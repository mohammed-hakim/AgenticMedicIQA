"""Tests for deterministic scoring-query detection."""

from src.agentic.nodes.summarizer import (
    QueryType,
    detect_query_type_rule_based,
    detect_query_type_with_vlm,
    is_explicit_scoring_query,
)


class RecordingVLM:
    def __init__(self, response: str = '{"type": "EXPLANATION", "reasoning": "test"}'):
        self.response = response
        self.calls = []

    def generate(self, prompt, images=None, temperature=0.0, max_tokens=100):
        self.calls.append(prompt)
        return self.response


def test_explicit_numeric_score_request_skips_vlm_classification():
    query = (
        "Assign a diagnostic quality score from 1 to 5 for this retinal fundus "
        "photograph, where 1 is unusable and 5 is diagnostic quality. "
        "Return only the numerical score."
    )
    client = RecordingVLM('{"type": "EXPLANATION", "reasoning": "wrong"}')

    query_type = detect_query_type_with_vlm(query, client)

    assert query_type == QueryType.SCORING
    assert client.calls == []


def test_assign_score_phrase_is_detected_without_scale():
    assert is_explicit_scoring_query("Assign a quality score to this image.")
    assert is_explicit_scoring_query("Give this radiograph a rating.")
    assert is_explicit_scoring_query("Report the numerical score.")


def test_explanation_about_scores_is_not_treated_as_scoring():
    assert not is_explicit_scoring_query("Explain why the tool score is low.")
    assert not is_explicit_scoring_query("Which artifacts are visible in this image?")


def test_vlm_classification_is_used_for_ambiguous_explanation_query():
    client = RecordingVLM('{"type": "EXPLANATION", "reasoning": "descriptive"}')

    query_type = detect_query_type_with_vlm(
        "Describe the artifacts affecting this image.", client
    )

    assert query_type == QueryType.EXPLANATION
    assert len(client.calls) == 1


def test_rule_based_default_benchmark_query_is_scoring():
    query = (
        "Assign a diagnostic quality score from 1 to 5 for this retinal fundus "
        "photograph, where 1 is unusable and 5 is diagnostic quality."
    )
    assert detect_query_type_rule_based(query) == QueryType.SCORING
