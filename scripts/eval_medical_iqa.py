#!/usr/bin/env python3
"""Evaluate medical IQA predictions against quality labels or MOS values.

The evaluator joins prediction records and ground truth by ``sample_id`` and
reports rank correlation, linear correlation, error, ordinal accuracy, quadratic
weighted kappa (QWK), and a confusion matrix. Bootstrap confidence intervals are
computed by resampling samples with replacement.

DRIMDB has three ordered quality classes rather than a continuous MOS. The
preparation script maps them to an ordinal pseudo-MOS:

    Good    -> 5.0
    Bad     -> 3.0
    Outlier -> 1.0

Correlation is meaningful for this ordering. Absolute error and accuracy depend
on the arbitrary spacing of the pseudo-MOS and should be reported as secondary
descriptors, not as clinical validity.

Usage:
    python scripts/eval_medical_iqa.py \
        --input results/drimdb_30.jsonl \
        --ground-truth data/processed/drimdb/ground_truth.jsonl \
        --output results/drimdb_30_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

# DRIMDB ordinal mapping. Kept here as a fallback for manifests that provide a
# label but no explicit MOS field.
LABEL_TO_MOS: Dict[str, float] = {
    "Good": 5.0,
    "Bad": 3.0,
    "Outlier": 1.0,
}
MOS_TO_LABEL: Dict[float, str] = {value: key for key, value in LABEL_TO_MOS.items()}
ORDINAL_LABELS = ["Outlier", "Bad", "Good"]

CAVEATS = [
    "DRIMDB labels are categorical; pseudo-MOS values are ordinal, not clinical measurements.",
    "Correlation is the primary pilot metric. MAE, RMSE, and accuracy depend on the chosen label spacing.",
    "Bootstrap intervals describe sampling variability for this small subset only.",
    "This is a benchmark pilot, not clinical validation.",
]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records, skipping blank or invalid lines."""
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping invalid JSON on line {line_number}: {exc}")
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _finite_float(value: Any) -> Optional[float]:
    """Return a finite float for numeric values, otherwise None."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def _parse_numeric_text(value: Any) -> Optional[float]:
    """Parse a numeric score from text such as ``4``, ``4.2/5`` or ``4 out of 5``."""
    if not isinstance(value, str):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", "."))
    if not match:
        return None
    number = float(match.group(0))
    return number if math.isfinite(number) else None


def _in_scale(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value if 1.0 <= value <= 5.0 else None


def extract_tool_mean(result: Dict[str, Any]) -> Optional[float]:
    """Extract the mean of normalized tool scores from executor evidence."""
    evidence = result.get("executor_evidence")
    if not isinstance(evidence, dict):
        return None

    scores: List[float] = []
    quality_scores = evidence.get("quality_scores") or {}
    if isinstance(quality_scores, dict):
        for object_scores in quality_scores.values():
            if not isinstance(object_scores, dict):
                continue
            for tool_score in object_scores.values():
                if isinstance(tool_score, (list, tuple)) and tool_score:
                    value = _finite_float(tool_score[-1])
                elif isinstance(tool_score, dict):
                    value = _finite_float(
                        tool_score.get("score", tool_score.get("normalized_score"))
                    )
                else:
                    value = _finite_float(tool_score)
                if value is not None and 1.0 <= value <= 5.0:
                    scores.append(value)

    if not scores:
        return None
    return float(np.mean(scores))


def extract_quality_score(
    result: Dict[str, Any],
    score_source: str = "auto",
) -> Tuple[Optional[float], str]:
    """Extract a predicted 1-5 quality score and report where it came from.

    ``auto`` prefers the fused ``summarizer_result.quality_score``, then a numeric
    ``final_answer``, then top-level score fields, then the mean tool score. The
    fallback order keeps the evaluator usable with older result files.
    """
    summarizer_result = result.get("summarizer_result")
    if not isinstance(summarizer_result, dict):
        summarizer_result = {}

    candidates: List[Tuple[str, Optional[float]]] = []
    if score_source in ("auto", "quality_score"):
        candidates.append(
            ("summarizer.quality_score", _finite_float(summarizer_result.get("quality_score")))
        )
    if score_source in ("auto", "final_answer"):
        final_answer = summarizer_result.get("final_answer")
        candidates.append(
            (
                "summarizer.final_answer",
                _finite_float(final_answer) if not isinstance(final_answer, str) else _parse_numeric_text(final_answer),
            )
        )
    if score_source == "auto":
        candidates.extend(
            [
                ("quality_score", _finite_float(result.get("quality_score"))),
                ("final_score", _finite_float(result.get("final_score"))),
                ("fused_score", _finite_float(result.get("fused_score"))),
            ]
        )
    if score_source in ("auto", "tool_mean"):
        candidates.append(("tool_mean", extract_tool_mean(result)))

    for source, value in candidates:
        value = _in_scale(value)
        if value is not None:
            return value, source

    return None, "missing"


def extract_ground_truth(record: Dict[str, Any], mos_field: str = "mos") -> Optional[Dict[str, Any]]:
    """Extract MOS and/or label from a ground-truth or prediction record."""
    metadata = record.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    mos = _finite_float(record.get(mos_field))
    if mos is None:
        mos = _finite_float(record.get("mos"))
    if mos is None:
        mos = _finite_float(metadata.get(mos_field))

    label = record.get("quality_label") or record.get("label")
    if not isinstance(label, str):
        label = None
    if label is None and mos is not None:
        label = MOS_TO_LABEL.get(round(mos, 3))

    if mos is None and label in LABEL_TO_MOS:
        mos = LABEL_TO_MOS[label]

    if mos is None and label is None:
        return None
    return {"mos": mos, "quality_label": label}


def mos_to_label(mos: float) -> str:
    """Map a continuous 1-5 score to the nearest DRIMDB ordinal class."""
    if mos >= 4.0:
        return "Good"
    if mos >= 2.0:
        return "Bad"
    return "Outlier"


def _result_rank(record: Dict[str, Any], score_source: str) -> int:
    """Rank duplicate records, preferring records with a usable prediction."""
    metadata = record.get("metadata")
    if isinstance(metadata, dict) and metadata.get("final_status") == "error":
        return 0
    score, _ = extract_quality_score(record, score_source=score_source)
    if score is not None:
        return 3
    if isinstance(record.get("summarizer_result"), dict):
        return 2
    return 1


def deduplicate_results(
    records: Sequence[Dict[str, Any]],
    score_source: str = "auto",
) -> Tuple[List[Dict[str, Any]], int]:
    """Keep the best record per sample_id.

    Resume runs append new results after old error records. This keeps the
    highest-ranked record (successful and score-bearing over an error) and
    prefers the later record when ranks tie.
    """
    best: Dict[str, Tuple[int, int, Dict[str, Any]]] = {}
    order: List[str] = []
    duplicates = 0

    for index, record in enumerate(records):
        sample_id = record.get("sample_id")
        if not sample_id:
            continue
        sample_id = str(sample_id)
        candidate = (_result_rank(record, score_source), index, record)
        if sample_id not in best:
            order.append(sample_id)
            best[sample_id] = candidate
        else:
            duplicates += 1
            if candidate[0] >= best[sample_id][0]:
                best[sample_id] = candidate

    return [best[sample_id][2] for sample_id in order], duplicates


def quadratic_weighted_kappa(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    labels: Sequence[str] = ORDINAL_LABELS,
) -> Optional[float]:
    """Compute quadratic weighted kappa without a scikit-learn dependency."""
    if len(true_labels) != len(predicted_labels) or not true_labels:
        return None

    index = {label: position for position, label in enumerate(labels)}
    try:
        true_indices = [index[label] for label in true_labels]
        predicted_indices = [index[label] for label in predicted_labels]
    except KeyError:
        return None

    size = len(labels)
    observed = np.zeros((size, size), dtype=float)
    for true_index, predicted_index in zip(true_indices, predicted_indices):
        observed[true_index, predicted_index] += 1.0

    true_hist = observed.sum(axis=1)
    predicted_hist = observed.sum(axis=0)
    total = observed.sum()
    if total == 0:
        return None
    expected = np.outer(true_hist, predicted_hist) / total

    weights = np.zeros((size, size), dtype=float)
    for row in range(size):
        for column in range(size):
            weights[row, column] = ((row - column) ** 2) / ((size - 1) ** 2)

    denominator = float(np.sum(weights * expected))
    if denominator == 0.0:
        return None
    return float(1.0 - np.sum(weights * observed) / denominator)


def confusion_matrix(
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    labels: Sequence[str] = ORDINAL_LABELS,
) -> Dict[str, Dict[str, int]]:
    """Return a nested confusion matrix keyed by label names."""
    matrix = {true_label: {predicted_label: 0 for predicted_label in labels} for true_label in labels}
    for true_label, predicted_label in zip(true_labels, predicted_labels):
        if true_label in matrix and predicted_label in matrix[true_label]:
            matrix[true_label][predicted_label] += 1
    return matrix


def _safe_correlation(function, predictions: np.ndarray, truth: np.ndarray) -> Optional[float]:
    if len(predictions) < 2 or len(np.unique(predictions)) < 2 or len(np.unique(truth)) < 2:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        value = function(predictions, truth)[0]
    return None if value is None or not math.isfinite(float(value)) else float(value)


def compute_metrics(
    predictions: Sequence[float],
    ground_truth: Sequence[float],
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
) -> Dict[str, Any]:
    """Compute point-estimate metrics for matched predictions and labels."""
    pred = np.asarray(predictions, dtype=float)
    truth = np.asarray(ground_truth, dtype=float)
    if pred.size == 0:
        return {"n_samples": 0}

    errors = pred - truth
    kappa = quadratic_weighted_kappa(true_labels, predicted_labels)
    accuracy = float(np.mean(np.asarray(true_labels) == np.asarray(predicted_labels)))

    return {
        "n_samples": int(pred.size),
        "srcc": _safe_correlation(stats.spearmanr, pred, truth),
        "plcc": _safe_correlation(stats.pearsonr, pred, truth),
        "kendall_tau": _safe_correlation(stats.kendalltau, pred, truth),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
        "ordinal_accuracy": accuracy,
        "quadratic_weighted_kappa": kappa,
        "prediction_mean": float(np.mean(pred)),
        "prediction_std": float(np.std(pred)),
        "prediction_min": float(np.min(pred)),
        "prediction_max": float(np.max(pred)),
        "confusion_matrix": confusion_matrix(true_labels, predicted_labels),
    }


def bootstrap_metrics(
    predictions: Sequence[float],
    ground_truth: Sequence[float],
    true_labels: Sequence[str],
    predicted_labels: Sequence[str],
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> Dict[str, Any]:
    """Bootstrap percentile intervals for the main metrics."""
    if n_bootstrap <= 0 or len(predictions) < 2:
        return {}

    pred = np.asarray(predictions, dtype=float)
    truth = np.asarray(ground_truth, dtype=float)
    true_labels_array = np.asarray(true_labels)
    predicted_labels_array = np.asarray(predicted_labels)
    rng = np.random.default_rng(seed)

    metric_values: Dict[str, List[float]] = {
        "srcc": [],
        "plcc": [],
        "kendall_tau": [],
        "mae": [],
        "rmse": [],
        "ordinal_accuracy": [],
        "quadratic_weighted_kappa": [],
    }
    skipped = {name: 0 for name in metric_values}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, len(pred), size=len(pred))
        sample_metrics = compute_metrics(
            pred[indices],
            truth[indices],
            true_labels_array[indices].tolist(),
            predicted_labels_array[indices].tolist(),
        )
        for name in metric_values:
            value = sample_metrics.get(name)
            if value is None:
                skipped[name] += 1
            else:
                metric_values[name].append(float(value))

    alpha = (1.0 - confidence) / 2.0
    intervals: Dict[str, Any] = {}
    for name, values in metric_values.items():
        intervals[name] = {
            "low": float(np.quantile(values, alpha)) if values else None,
            "high": float(np.quantile(values, 1.0 - alpha)) if values else None,
            "valid_replicates": len(values),
            "skipped_replicates": skipped[name],
        }
    return {
        "intervals": intervals,
        "n_bootstrap": n_bootstrap,
        "confidence": confidence,
        "seed": seed,
    }


def evaluate(
    prediction_records: Sequence[Dict[str, Any]],
    ground_truth_records: Optional[Sequence[Dict[str, Any]]],
    score_source: str = "auto",
    mos_field: str = "mos",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
    exclude_labels: Sequence[str] = (),
) -> Dict[str, Any]:
    """Join predictions with ground truth and compute all pilot metrics.

    ``exclude_labels`` supports sensitivity analyses, for example dropping the
    DRIMDB Outlier class when its non-retinal content makes ordinal placement
    debatable.
    """
    excluded_labels = {label.strip() for label in exclude_labels if label.strip()}
    raw_prediction_count = len(prediction_records)
    prediction_records, n_duplicate_predictions = deduplicate_results(
        prediction_records, score_source=score_source
    )

    truth_by_id: Dict[str, Dict[str, Any]] = {}
    if ground_truth_records:
        for record in ground_truth_records:
            sample_id = record.get("sample_id")
            truth = extract_ground_truth(record, mos_field=mos_field)
            if sample_id and truth is not None:
                truth_by_id[str(sample_id)] = truth

    matched: List[Dict[str, Any]] = []
    source_counts: Dict[str, int] = {}
    missing_predictions: List[str] = []
    unmatched_ground_truth: List[str] = []
    n_excluded = 0

    for record in prediction_records:
        sample_id = record.get("sample_id")
        if not sample_id:
            continue
        sample_id = str(sample_id)

        prediction, source = extract_quality_score(record, score_source=score_source)
        source_counts[source] = source_counts.get(source, 0) + 1
        if prediction is None:
            missing_predictions.append(sample_id)
            continue

        if truth_by_id:
            truth = truth_by_id.get(sample_id)
            if truth is None:
                continue
        else:
            truth = extract_ground_truth(record, mos_field=mos_field)
            if truth is None:
                continue

        ground_truth_mos = float(truth["mos"])
        ground_truth_label = truth.get("quality_label") or mos_to_label(ground_truth_mos)
        if ground_truth_label in excluded_labels:
            n_excluded += 1
            continue
        matched.append(
            {
                "sample_id": sample_id,
                "prediction": prediction,
                "ground_truth_mos": ground_truth_mos,
                "predicted_label": mos_to_label(prediction),
                "ground_truth_label": ground_truth_label,
                "score_source": source,
            }
        )

    if truth_by_id:
        matched_ids = {entry["sample_id"] for entry in matched}
        unmatched_ground_truth = sorted(set(truth_by_id) - matched_ids)

    predictions = [entry["prediction"] for entry in matched]
    ground_truth = [entry["ground_truth_mos"] for entry in matched]
    predicted_labels = [entry["predicted_label"] for entry in matched]
    true_labels = [entry["ground_truth_label"] for entry in matched]

    metrics = compute_metrics(predictions, ground_truth, true_labels, predicted_labels)
    intervals = bootstrap_metrics(
        predictions,
        ground_truth,
        true_labels,
        predicted_labels,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
    )

    per_class: Dict[str, Dict[str, Any]] = {}
    for label in ORDINAL_LABELS:
        class_predictions = [
            entry["prediction"] for entry in matched if entry["ground_truth_label"] == label
        ]
        class_correct = sum(
            1
            for entry in matched
            if entry["ground_truth_label"] == label and entry["predicted_label"] == label
        )
        per_class[label] = {
            "n": len(class_predictions),
            "prediction_mean": float(np.mean(class_predictions)) if class_predictions else None,
            "recall": (class_correct / len(class_predictions)) if class_predictions else None,
        }

    return {
        "n_prediction_records": len(prediction_records),
        "n_prediction_records_raw": raw_prediction_count,
        "n_duplicate_prediction_records": n_duplicate_predictions,
        "n_matched": len(matched),
        "n_ground_truth": len(truth_by_id) if truth_by_id else len(matched),
        "excluded_labels": sorted(excluded_labels),
        "n_excluded_by_label": n_excluded,
        "score_source_requested": score_source,
        "score_source_counts": source_counts,
        "missing_prediction_ids": missing_predictions,
        "unmatched_ground_truth_ids": unmatched_ground_truth,
        "metrics": metrics,
        "bootstrap": intervals,
        "per_class": per_class,
        "label_mapping": LABEL_TO_MOS,
        "samples": matched,
        "caveats": CAVEATS,
    }


def _format_interval(bootstrap: Dict[str, Any], metric: str) -> str:
    point = bootstrap.get("point", {}).get(metric)
    interval = bootstrap.get("intervals", {}).get(metric, {})
    if point is None:
        return f"{metric.upper():<24} n/a"
    if not interval:
        return f"{metric.upper():<24} {point:.4f}"
    low = interval.get("low")
    high = interval.get("high")
    if low is None or high is None:
        return f"{metric.upper():<24} {point:.4f}"
    return f"{metric.upper():<24} {point:.4f}  [{low:.4f}, {high:.4f}]"


def print_summary(report: Dict[str, Any]) -> None:
    """Print a compact human-readable summary."""
    metrics = report.get("metrics", {})
    intervals = report.get("bootstrap", {}).get("intervals", {})

    print("=" * 72)
    print("Medical IQA Evaluation")
    print("=" * 72)
    print(f"Matched samples: {report.get('n_matched', 0)}")
    if report.get("excluded_labels"):
        print(
            f"Excluded labels: {report['excluded_labels']} "
            f"({report.get('n_excluded_by_label', 0)} samples)"
        )
    print(f"Score sources: {report.get('score_source_counts', {})}")

    point_metrics = {name: metrics.get(name) for name in (
        "srcc",
        "plcc",
        "kendall_tau",
        "mae",
        "rmse",
        "ordinal_accuracy",
        "quadratic_weighted_kappa",
    )}
    bootstrap_view = {"point": point_metrics, "intervals": intervals}

    print("\nMetric                      Point      Bootstrap interval")
    for metric in point_metrics:
        print("  " + _format_interval(bootstrap_view, metric))

    print("\nConfusion matrix (rows = ground truth, columns = predicted)")
    labels = ORDINAL_LABELS
    print("  " + " " * 10 + "".join(f"{label:>9}" for label in labels))
    matrix = metrics.get("confusion_matrix", {})
    for true_label in labels:
        row = matrix.get(true_label, {})
        print("  " + f"{true_label:<10}" + "".join(f"{row.get(pred, 0):>9}" for pred in labels))

    print("\nPer-class recall:")
    for label, values in report.get("per_class", {}).items():
        recall = values.get("recall")
        recall_text = f"{recall:.3f}" if recall is not None else "n/a"
        print(f"  {label:<10} n={values.get('n', 0):<3} recall={recall_text}")

    print("\nCaveats:")
    for caveat in report.get("caveats", []):
        print(f"  - {caveat}")
    print("=" * 72)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Medical AgenticIQA predictions against quality labels/MOS"
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Prediction JSONL")
    parser.add_argument(
        "--ground-truth",
        "-g",
        type=Path,
        help="Ground-truth JSONL. If omitted, MOS/labels are read from the input records.",
    )
    parser.add_argument("--output", "-o", type=Path, help="Write the full report as JSON")
    parser.add_argument(
        "--score-source",
        choices=["auto", "quality_score", "final_answer", "tool_mean"],
        default="auto",
        help="Prediction field to evaluate (default: auto)",
    )
    parser.add_argument("--mos-field", default="mos", help="Ground-truth MOS field (default: mos)")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Bootstrap resamples; 0 disables intervals (default: 1000)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Bootstrap confidence level (default: 0.95)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed (default: 42)")
    parser.add_argument(
        "--exclude-labels",
        default="",
        help=(
            "Comma-separated ground-truth labels to exclude for a sensitivity "
            "analysis, e.g. Outlier"
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file not found: {args.input}")
        return 1
    if args.ground_truth and not args.ground_truth.exists():
        print(f"Error: ground-truth file not found: {args.ground_truth}")
        return 1

    predictions = load_jsonl(args.input)
    ground_truth = load_jsonl(args.ground_truth) if args.ground_truth else None
    exclude_labels = [label.strip() for label in args.exclude_labels.split(",") if label.strip()]

    report = evaluate(
        predictions,
        ground_truth,
        score_source=args.score_source,
        mos_field=args.mos_field,
        n_bootstrap=args.n_bootstrap,
        confidence=args.confidence,
        seed=args.seed,
        exclude_labels=exclude_labels,
    )

    if report["n_matched"] == 0:
        print("Error: no matched samples with both a prediction and ground truth.")
        print(f"Prediction sources: {report.get('score_source_counts', {})}")
        return 1

    print_summary(report)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"\nFull report written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
