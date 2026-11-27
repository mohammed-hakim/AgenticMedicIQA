"""Tests for pipeline result serialization and benchmark field preservation."""

import json

from run_pipeline import (
    copy_sample_fields,
    get_processed_sample_ids,
    has_quality_score,
    is_successful_result,
    serialize_state_for_output,
)


def _sample():
    return {
        "sample_id": "drimdb_good_001",
        "query": "Assign a score from 1 to 5.",
        "image_path": "data/processed/drimdb/images/drimdb_good_001.jpg",
        "dataset": "drimdb",
        "split": "pilot",
        "mos": 5.0,
        "quality_label": "Good",
        "metadata": {
            "source_dataset": "DRIMDB",
            "source_file": "good_001.jpg",
        },
    }


def test_copy_sample_fields_preserves_benchmark_metadata():
    output = {}
    copy_sample_fields(_sample(), output)

    assert output["dataset"] == "drimdb"
    assert output["split"] == "pilot"
    assert output["mos"] == 5.0
    assert output["quality_label"] == "Good"
    assert output["sample_metadata"]["source_dataset"] == "DRIMDB"


def test_serialize_state_preserves_labels_and_mirrors_scores():
    state = {
        "query": "Assign a score from 1 to 5.",
        "image_path": "data/processed/drimdb/images/drimdb_good_001.jpg",
        "summarizer_result": {
            "final_answer": 4.25,
            "quality_score": 4.25,
            "quality_reasoning": "Good diagnostic usability.",
            "need_replan": False,
        },
    }

    output = serialize_state_for_output(state, _sample(), execution_time=1.25)

    assert output["mos"] == 5.0
    assert output["quality_label"] == "Good"
    assert output["dataset"] == "drimdb"
    assert output["sample_metadata"]["source_file"] == "good_001.jpg"
    assert output["quality_score"] == 4.25
    assert output["final_answer"] == 4.25
    assert output["summarizer_result"]["quality_score"] == 4.25
    assert output["metadata"]["execution_time_ms"] == 1250.0
    assert output["metadata"]["final_status"] == "success"


def test_resume_skips_successes_but_retries_errors(tmp_path):
    output_path = tmp_path / "results.jsonl"
    records = [
        {
            "sample_id": "good_001",
            "summarizer_result": {"quality_score": 4.0},
            "metadata": {"final_status": "success"},
        },
        {
            "sample_id": "bad_001",
            "metadata": {"final_status": "error", "error_details": "API failure"},
        },
        {
            "sample_id": "bad_002",
            "quality_score": 2.5,
            "metadata": {"final_status": "success"},
        },
    ]
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    processed = get_processed_sample_ids(output_path)

    assert processed == {"good_001", "bad_002"}
    assert is_successful_result(records[0]) is True
    assert is_successful_result(records[1]) is False


def test_resume_can_retry_successes_without_a_quality_score(tmp_path):
    output_path = tmp_path / "results.jsonl"
    records = [
        {
            "sample_id": "scored",
            "summarizer_result": {"quality_score": 4.0, "final_answer": 4.0},
            "metadata": {"final_status": "success"},
        },
        {
            "sample_id": "text_only",
            "summarizer_result": {"quality_score": None, "final_answer": "Looks acceptable."},
            "metadata": {"final_status": "success"},
        },
    ]
    with open(output_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    default_processed = get_processed_sample_ids(output_path)
    scored_processed = get_processed_sample_ids(output_path, require_quality_score=True)

    assert default_processed == {"scored", "text_only"}
    assert scored_processed == {"scored"}
    assert has_quality_score(records[0]) is True
    assert has_quality_score(records[1]) is False
