"""Tests for pipeline result serialization and benchmark field preservation."""

from run_pipeline import copy_sample_fields, serialize_state_for_output


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
