"""Tests for the medical IQA evaluation metrics and score extraction."""

import pytest

from scripts.eval_medical_iqa import (
    compute_metrics,
    deduplicate_results,
    evaluate,
    extract_quality_score,
    extract_tool_mean,
    mos_to_label,
    quadratic_weighted_kappa,
)


def _prediction(sample_id: str, score: float, label: str, **extra):
    record = {
        "sample_id": sample_id,
        "summarizer_result": {
            "quality_score": score,
            "final_answer": str(score),
            "quality_reasoning": "test",
        },
        "quality_label": label,
    }
    record.update(extra)
    return record


def test_extract_quality_score_prefers_summarizer_quality_score():
    record = {
        "summarizer_result": {"quality_score": 3.75, "final_answer": "4.5"},
        "quality_score": 2.0,
    }
    score, source = extract_quality_score(record)
    assert score == pytest.approx(3.75)
    assert source == "summarizer.quality_score"


def test_extract_quality_score_parses_numeric_final_answer():
    record = {"summarizer_result": {"final_answer": "4.2/5"}}
    score, source = extract_quality_score(record)
    assert score == pytest.approx(4.2)
    assert source == "summarizer.final_answer"


def test_extract_quality_score_uses_tool_mean_fallback():
    record = {
        "executor_evidence": {
            "quality_scores": {
                "Global": {
                    "Blurs": ["MUSIQ", 3.0],
                    "Noise": ["BRISQUE", 2.0],
                }
            }
        }
    }
    score, source = extract_quality_score(record)
    assert score == pytest.approx(2.5)
    assert source == "tool_mean"


def test_extract_quality_score_rejects_out_of_range_values():
    record = {"summarizer_result": {"quality_score": 9.0, "final_answer": "C"}}
    score, source = extract_quality_score(record)
    assert score is None
    assert source == "missing"


def test_extract_tool_mean_handles_dicts_and_scalars():
    record = {
        "executor_evidence": {
            "quality_scores": {
                "Global": {"A": {"tool": "MUSIQ", "score": 4.0}, "B": 2.0},
                "Region": {"C": ["TOPIQ", 3.0]},
            }
        }
    }
    assert extract_tool_mean(record) == pytest.approx(3.0)


def test_mos_to_label_boundaries():
    assert mos_to_label(5.0) == "Good"
    assert mos_to_label(4.0) == "Good"
    assert mos_to_label(3.9) == "Bad"
    assert mos_to_label(2.0) == "Bad"
    assert mos_to_label(1.9) == "Outlier"
    assert mos_to_label(1.0) == "Outlier"


def test_quadratic_weighted_kappa_perfect_and_reversed():
    labels = ["Good"] * 3 + ["Bad"] * 3 + ["Outlier"] * 3
    assert quadratic_weighted_kappa(labels, labels) == pytest.approx(1.0)
    reversed_labels = list(reversed(labels))
    assert quadratic_weighted_kappa(labels, reversed_labels) == pytest.approx(-1.0)


def test_compute_metrics_confusion_matrix_and_errors():
    predictions = [5.0, 3.0, 1.0]
    truth = [5.0, 3.0, 1.0]
    true_labels = ["Good", "Bad", "Outlier"]
    predicted_labels = ["Good", "Bad", "Outlier"]

    metrics = compute_metrics(predictions, truth, true_labels, predicted_labels)

    assert metrics["n_samples"] == 3
    assert metrics["srcc"] == pytest.approx(1.0)
    assert metrics["plcc"] == pytest.approx(1.0)
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["ordinal_accuracy"] == pytest.approx(1.0)
    assert metrics["quadratic_weighted_kappa"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"]["Good"]["Good"] == 1
    assert metrics["confusion_matrix"]["Bad"]["Outlier"] == 0


def test_evaluate_joins_by_sample_id_and_bootstraps():
    predictions = [
        _prediction("a", 5.0, "Good"),
        _prediction("b", 3.0, "Bad"),
        _prediction("c", 1.0, "Outlier"),
        _prediction("d", 4.0, "Good"),
    ]
    ground_truth = [
        {"sample_id": "a", "mos": 5.0, "quality_label": "Good"},
        {"sample_id": "b", "mos": 3.0, "quality_label": "Bad"},
        {"sample_id": "c", "mos": 1.0, "quality_label": "Outlier"},
        {"sample_id": "d", "mos": 5.0, "quality_label": "Good"},
    ]

    report = evaluate(
        predictions,
        ground_truth,
        n_bootstrap=25,
        confidence=0.9,
        seed=7,
    )

    assert report["n_matched"] == 4
    assert report["score_source_counts"] == {"summarizer.quality_score": 4}
    assert report["metrics"]["srcc"] == pytest.approx(0.9486832980505138)
    assert report["metrics"]["mae"] == pytest.approx(0.25)
    assert report["bootstrap"]["n_bootstrap"] == 25
    assert report["bootstrap"]["confidence"] == pytest.approx(0.9)
    assert len(report["samples"]) == 4


def test_deduplicate_results_prefers_score_bearing_success_over_error():
    records = [
        {"sample_id": "a", "metadata": {"final_status": "error"}},
        _prediction("a", 4.0, "Good"),
        _prediction("b", 2.0, "Bad"),
        {"sample_id": "b", "metadata": {"final_status": "error"}},
    ]

    deduplicated, duplicates = deduplicate_results(records)

    assert duplicates == 2
    assert len(deduplicated) == 2
    scores = {record["sample_id"]: record["summarizer_result"]["quality_score"] for record in deduplicated}
    assert scores == {"a": 4.0, "b": 2.0}


def test_evaluate_retries_after_error_record_are_counted_once():
    predictions = [
        {"sample_id": "a", "metadata": {"final_status": "error"}},
        _prediction("a", 5.0, "Good"),
        _prediction("b", 3.0, "Bad"),
    ]
    ground_truth = [
        {"sample_id": "a", "mos": 5.0, "quality_label": "Good"},
        {"sample_id": "b", "mos": 3.0, "quality_label": "Bad"},
    ]

    report = evaluate(predictions, ground_truth, n_bootstrap=0)

    assert report["n_prediction_records_raw"] == 3
    assert report["n_duplicate_prediction_records"] == 1
    assert report["n_prediction_records"] == 2
    assert report["n_matched"] == 2


def test_evaluate_can_exclude_a_label_for_sensitivity_analysis():
    predictions = [_prediction("a", 5.0, "Good"), _prediction("b", 1.0, "Outlier")]
    ground_truth = [
        {"sample_id": "a", "mos": 5.0, "quality_label": "Good"},
        {"sample_id": "b", "mos": 1.0, "quality_label": "Outlier"},
    ]

    report = evaluate(predictions, ground_truth, n_bootstrap=0, exclude_labels=["Outlier"])

    assert report["n_matched"] == 1
    assert report["n_excluded_by_label"] == 1
    assert report["excluded_labels"] == ["Outlier"]
    assert report["metrics"]["n_samples"] == 1


def test_evaluate_without_ground_truth_file_reads_labels_from_predictions():
    predictions = [
        _prediction("a", 5.0, "Good"),
        _prediction("b", 3.0, "Bad"),
    ]
    report = evaluate(predictions, None, n_bootstrap=0)
    assert report["n_matched"] == 2
    assert report["metrics"]["srcc"] == pytest.approx(1.0)
