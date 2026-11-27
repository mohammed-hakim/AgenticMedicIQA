#!/usr/bin/env bash
# Run the DRIMDB 30-image pilot on a Linux host, including Kaggle notebooks.
#
# Prerequisites:
#   - Repository dependencies installed (pip install -r requirements.txt)
#   - OPENAI_API_KEY set in the environment (or another configured backend)
#   - Network access for the DRIMDB download and model weights
#
# Usage:
#   bash scripts/run_drimdb_pilot.sh
#
# Optional environment variables:
#   AGENTIC_BACKEND          VLM backend (default: openai.gpt-4o-mini)
#   DRIMDB_PER_CLASS         Images per class (default: 10)
#   DRIMDB_SEED              Sampling seed (default: 42)
#   AGENTIC_EXCLUDED_TOOLS   Tools to hide (default: QAlign)
#   DRIMDB_OUTPUT            Prediction JSONL (default: results/drimdb_30.jsonl)
#   DRIMDB_DRY_RUN           Set to 1 to prepare data and stop before inference

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export AGENTIC_ROOT="$REPO_ROOT"
export PYTHONUNBUFFERED=1

BACKEND="${AGENTIC_BACKEND:-openai.gpt-4o-mini}"
PER_CLASS="${DRIMDB_PER_CLASS:-10}"
SEED="${DRIMDB_SEED:-42}"
OUTPUT="${DRIMDB_OUTPUT:-results/drimdb_30.jsonl}"
export AGENTIC_EXCLUDED_TOOLS="${AGENTIC_EXCLUDED_TOOLS:-QAlign}"

DATA_PARENT="data/raw/drimdb"
DATASET_URL="https://www.kaggle.com/api/v1/datasets/download/subhajournal/drimdb-diabetic-retinopathy-images-database"

echo "==> Repository: $REPO_ROOT"
echo "==> Backend: $BACKEND"
echo "==> Excluded tools: $AGENTIC_EXCLUDED_TOOLS"

if [ ! -d "$DATA_PARENT/DRIMDB" ] && [ ! -d "$DATA_PARENT/Good" ]; then
  echo "==> Downloading DRIMDB"
  mkdir -p "$DATA_PARENT"
  if [ ! -f "$DATA_PARENT/drimdb.zip" ]; then
    if ! curl -L --fail --retry 3 -o "$DATA_PARENT/drimdb.zip" "$DATASET_URL"; then
      rm -f "$DATA_PARENT/drimdb.zip"
      if command -v kaggle >/dev/null 2>&1; then
        echo "==> curl failed; trying the Kaggle CLI"
        kaggle datasets download \
          -d subhajournal/drimdb-diabetic-retinopathy-images-database \
          -p "$DATA_PARENT" --unzip
      else
        echo "Error: DRIMDB download failed and the kaggle CLI is unavailable." >&2
        exit 1
      fi
    fi
  fi
  if [ -f "$DATA_PARENT/drimdb.zip" ]; then
    unzip -q -o "$DATA_PARENT/drimdb.zip" -d "$DATA_PARENT"
  fi
else
  echo "==> Using existing DRIMDB data under $DATA_PARENT"
fi

echo "==> Preparing stratified pilot subset"
python scripts/data_prep/prepare_drimdb.py \
  --data-dir "$DATA_PARENT" \
  --per-class "$PER_CLASS" \
  --seed "$SEED"

if [ "${DRIMDB_DRY_RUN:-0}" = "1" ]; then
  echo "==> DRIMDB_DRY_RUN=1; stopping before inference"
  exit 0
fi

echo "==> Running pipeline -> $OUTPUT"
python run_pipeline.py \
  --input data/processed/drimdb/manifest.jsonl \
  --output "$OUTPUT" \
  --resume \
  --max-replan 1 \
  --backend-override "planner.backend=$BACKEND" \
  --backend-override "executor.backend=$BACKEND" \
  --backend-override "summarizer.backend=$BACKEND"

echo "==> Evaluating all three classes"
python scripts/eval_medical_iqa.py \
  --input "$OUTPUT" \
  --ground-truth data/processed/drimdb/ground_truth.jsonl \
  --output results/drimdb_30_eval.json

echo "==> Evaluating Good-vs-Bad sensitivity analysis"
python scripts/eval_medical_iqa.py \
  --input "$OUTPUT" \
  --ground-truth data/processed/drimdb/ground_truth.jsonl \
  --exclude-labels Outlier \
  --output results/drimdb_30_eval_good_bad.json

echo "==> Done"
echo "    predictions: $OUTPUT"
echo "    report:      results/drimdb_30_eval.json"
echo "    sensitivity: results/drimdb_30_eval_good_bad.json"
