# Medical AgenticIQA

**Medical AgenticIQA** is a research prototype for medical image acquisition-quality
assessment. It extends the AgenticIQA planner/executor/summarizer architecture with
modality-aware quality criteria, file-level DICOM and NIfTI ingestion, classical QC
indicators, and privacy safeguards.

> **Not a medical device. No clinical validation. No diagnosis.**
> This project assesses image quality and diagnostic usability only. It does not
> provide clinical findings, diagnoses, or treatment advice. Tool scores and
> classical indicators are uncalibrated proxies, not clinical measurements. The
> project includes safeguards but does **not** claim HIPAA/GDPR compliance.

This repository is an unofficial reproduction and medical adaptation inspired by the
AgenticIQA paper. It preserves the original Planner → Executor → Summarizer workflow
and API contracts where possible.

---

## Contents

- [What it does](#what-it-does)
- [Supported modalities](#supported-modalities)
- [Supported inputs](#supported-inputs)
- [Quick start](#quick-start)
- [Running the pipeline](#running-the-pipeline)
- [REST API](#rest-api)
- [How assessment works](#how-assessment-works)
- [Medical QC indicators](#medical-qc-indicators)
- [Scoring and metric caveats](#scoring-and-metric-caveats)
- [Privacy and safety](#privacy-and-safety)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Limitations and non-goals](#limitations-and-non-goals)
- [Attribution](#attribution)

---

## What it does

For each input image, the pipeline:

1. **Planner** infers the task, regions of interest, requested distortions, and an
   optional medical modality profile.
2. **Executor** detects and analyzes acquisition artifacts using modality-aware
   prompts, selects existing IQA tools, executes them, and computes auxiliary
   classical QC indicators.
3. **Summarizer** combines artifact evidence, tool scores, modality criteria, and QC
   indicators into a quality judgment and a 1–5 diagnostic-usability score when a
   numeric score is requested.

Key properties:

- **Modality profiles + generic fallback** — eight medical groups plus `generic_medical`.
- **DICOM and NIfTI file support** — single files are rendered to cached PNGs so the
  VLM and IQA tools can consume them.
- **No new learned weights** — the bundled IQA models are used as relative proxies.
- **Auxiliary QC** — classical indicators support the explanation but are not fused
  into the 1–5 score.
- **Existing behavior preserved** — raster images, planner flags, and the API
  request/response shapes still work.

---

## Supported modalities

The planner may return any profile key below. Unknown or ambiguous modalities fall
back to `generic_medical` instead of failing.

| Profile key | Coverage | Example artifacts |
|---|---|---|
| `radiography` | X-ray, DXA, fluoroscopy | motion, exposure clipping, positioning, metal, cropping |
| `ct` | Computed tomography | noise, motion, metal, ring, beam hardening, truncation |
| `mri` | Magnetic resonance imaging | motion, ghosting, aliasing/wrap, bias field, cropping |
| `ultrasound` | Ultrasound / echocardiography | speckle, shadowing, reverberation, gain/depth, cropping |
| `mammography` | Mammography / breast imaging | positioning, motion, skin-line cropping, contrast, exposure |
| `fundus_oct` | Fundus photography / OCT | focus, uneven illumination, color cast, shadowing, cropping |
| `dermatology` | Dermatology / dermoscopy | focus, color cast, glare, hair artifact, cropping |
| `histopathology` | Histopathology / microscopy | focus, stain variation, tissue folds, bubbles, resolution |
| `generic_medical` | Fallback for unknown modalities | blur, noise, contrast, exposure, compression, spatial artifacts |

Each profile defines artifact vocabulary, quality criteria, tool-selection hints, and
a shared diagnostic-usability rubric. Profiles live in
`src/agentic/medical_domain.py` and are data-only, so adding a modality does not
change the graph.

---

## Supported inputs

| Format | Extensions | Handling |
|---|---|---|
| Raster | `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tif`, `.tiff` | Read directly with Pillow |
| DICOM | `.dcm`, `.dicom`, `.ima` | Decode pixel data, apply rescale/window, render cached PNG |
| NIfTI | `.nii`, `.nii.gz` | Select a representative slice, normalize intensities, render cached PNG |

### DICOM details

- Applies `RescaleSlope` / `RescaleIntercept`.
- Uses `WindowCenter` / `WindowWidth` when present; otherwise applies robust 1st–99th
  percentile normalization.
- Handles `MONOCHROME1` inversion, multi-frame selection (middle frame), and common
  color spaces.
- Extracts only non-identifying metadata such as modality code, body part, matrix
  size, frame count, slice thickness, and display window.
- Compressed transfer syntaxes may require optional `pylibjpeg` plugins.

### NIfTI details

- A 4D volume uses the middle volume; a 3D volume uses the middle slice along the
  third axis.
- Intensities are normalized with robust percentiles.
- Only shape, voxel size, and data type are reported as metadata.
- Orientation and slice selection are heuristics, not a substitute for a viewing
  protocol.

### Not supported

- Live PACS, DICOMweb, C-FIND/C-GET/C-STORE, and full-study/series traversal.
- Automatic selection of the most diagnostic slice or series.
- Burned-in PHI detection or OCR redaction.
- Medical-device regulatory workflows.

---

## Quick start

### 1. Create an environment

```bash
conda create -n medical-agentic-iqa python=3.10 -y
conda activate medical-agentic-iqa
```

Alternatively, recreate the pinned environment directly:

```bash
conda env create -f environment.yml
conda activate medical-agentic-iqa
```

Install PyTorch for your platform, then the project dependencies:

```bash
# Example CUDA 12.1 build; adjust to your driver/CUDA version.
pip install torch==2.3.0 torchvision==0.18.0 \
  --extra-index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

Install IQA-PyTorch for the bundled metrics:

```bash
git clone https://github.com/chaofengc/IQA-PyTorch.git
cd IQA-PyTorch
pip install -e .
cd ..
```

Medical ingestion requires `pydicom` and `nibabel`, which are included in
`requirements.txt`.

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```bash
AGENTIC_ROOT=/absolute/path/to/medical-agentic-iqa
AGENTIC_DATA_ROOT=${AGENTIC_ROOT}/data
AGENTIC_TOOL_HOME=${AGENTIC_ROOT}/iqa_tools
AGENTIC_LOG_ROOT=${AGENTIC_ROOT}/logs
AGENTIC_RENDER_CACHE=${AGENTIC_DATA_ROOT}/processed/render_cache

OPENAI_API_KEY=sk-...
# or ANTHROPIC_API_KEY / GOOGLE_API_KEY
```

For local OpenAI-compatible inference:

```bash
OPENAI_BASE_URL=http://localhost:8000/v1
```

### 3. Verify the environment

```bash
python scripts/check_env.py
python scripts/verify_iqa_tools.py --type NR --skip-slow
```

The first tool run may download model weights.

---

## Running the pipeline

### Batch CLI

Create a JSONL file. A medical example set is available at
`examples/medical_samples.jsonl`:

```jsonl
{"sample_id": "chest_xray_001", "query": "Rate the diagnostic quality of this chest radiograph and describe any acquisition artifacts.", "image_path": "data/raw/chest_xray.dcm"}
{"sample_id": "mri_brain_001", "query": "Assess the diagnostic usability of this brain MRI slice and note any ghosting or aliasing.", "image_path": "data/raw/brain_mri.nii.gz"}
{"sample_id": "ultrasound_001", "query": "Does this ultrasound image have adequate gain and contrast for interpretation?", "image_path": "data/raw/ultrasound.jpg"}
```

Run:

```bash
python run_pipeline.py \
  --input examples/medical_samples.jsonl \
  --output results/medical_output.jsonl
```

Common options:

| Option | Description |
|---|---|
| `--input, -i` | Input JSONL path (required) |
| `--output, -o` | Output JSONL path (required) |
| `--resume` | Skip samples already present in the output file |
| `--max-samples, -n` | Limit the number of samples |
| `--max-replan` | Maximum replanning iterations (default: 2) |
| `--dry-run` | Validate configuration and show the execution plan |
| `--verbose, -v` | Enable detailed logs |

Result records include `sample_id`, the original query and paths, the plan,
executor evidence (including `medical_qc`), the summarizer result, timing
metadata, and top-level `quality_score` / `final_answer` mirrors. Benchmark label
fields (`mos`, `quality_label`, `dataset`, `split`) and manifest provenance
(`sample_metadata`) are copied from the input record when present, so evaluation
can join by `sample_id` without reopening the manifest.

### Python

```python
from src.agentic.graph import run_pipeline

state = run_pipeline(
    query="Is this chest radiograph adequate for interpretation?",
    image_path="data/raw/chest_xray.dcm",
    max_replan_iterations=2,
)

result = state["summarizer_result"]
print(result.final_answer)
print(result.quality_reasoning)
print(state["executor_evidence"].medical_qc["heuristic_flags"])
```

---

## REST API

Start the service:

```bash
python scripts/run_api.py
# or
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

- Swagger UI: `http://localhost:8000/docs`
- Health check: `GET /health`
- Assessment: `POST /assess`

### Request

```bash
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Rate the diagnostic quality of this chest radiograph.",
    "image_path": "/data/raw/chest_xray.dcm",
    "max_replan_iterations": 2
  }'
```

Full-reference request:

```bash
curl -X POST http://localhost:8000/assess \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare this image with the reference and rate its quality.",
    "image_path": "/data/raw/degraded.png",
    "reference_path": "/data/raw/reference.png"
  }'
```

### Response

```json
{
  "final_answer": 3.8,
  "quality_score": 3.8,
  "quality_reasoning": "The radiograph is adequately exposed with no severe motion artifact. Mild noise is present. Tool scores and classical indicators are treated as auxiliary evidence.",
  "need_replan": false,
  "execution_metadata": {
    "iteration_count": 0,
    "replan_history": [],
    "execution_time_seconds": 8.4
  }
}
```

The public API shape is unchanged. The optional `modality` field appears inside the
serialized plan in pipeline outputs when the planner provides it.

---

## How assessment works

```text
Planner
  ├─ infer task type, regions, requested distortions
  ├─ infer modality profile (or fall back to generic_medical)
  └─ decide which Executor subtasks to run

Executor
  ├─ detect artifacts with modality-aware prompts
  ├─ analyze severity and diagnostic impact
  ├─ select existing IQA tools (FR when a reference exists, otherwise NR)
  ├─ execute tools and normalize scores to [1, 5]
  └─ compute auxiliary classical QC indicators

Summarizer
  ├─ combine artifact evidence, tool scores, and modality rubric
  ├─ report 1–5 diagnostic usability for scoring queries
  ├─ explain findings without providing a diagnosis
  └─ optionally request replanning when evidence is missing
```

### Diagnostic-usability rubric

| Score | Meaning |
|---|---|
| 5 | Diagnostic quality: no artifact that limits interpretation |
| 4 | Good: minor artifact unlikely to affect interpretation |
| 3 | Fair: moderate artifact may reduce confidence in some regions |
| 2 | Poor: significant artifact substantially limits interpretation |
| 1 | Non-diagnostic: severe artifact; repeat acquisition or expert review recommended |

The rubric describes **image usability**, not a patient's condition.

---

## Medical QC indicators

`src/agentic/medical_metrics.py` computes lightweight indicators from the
display-rendered image:

- Focus: Laplacian variance, Tenengrad, relative focus.
- Noise: robust sigma estimate.
- Contrast: RMS and percentile-based Michelson contrast.
- Exposure: low/high clipping fractions.
- Illumination uniformity: coefficient of variation across tiles.
- Information content: Shannon entropy.
- Edge density: Canny edge fraction.
- Heuristic flags such as `low_local_sharpness`, `high_noise_estimate`,
  `low_contrast`, `exposure_clipping`, `uneven_illumination`, and `sparse_edges`.

These values are returned in `ExecutorOutput.medical_qc` and passed to the
summarizer as **auxiliary evidence**. They are not fused into the final score, and
they are not calibrated per modality. A flag is a prompt for review, not a finding.

---

## Scoring and metric caveats

- The bundled metrics (MUSIQ, MANIQA, BRISQUE, NIQE, CLIPIQA, TOPIQ, SSIM, LPIPS,
  etc.) were developed or calibrated on natural images.
- The five-parameter logistic normalization coefficients in
  `iqa_tools/metadata/tools.json` were fitted on KADID-10k natural images. They are
  retained for reproducibility and are **not medical calibration**.
- Medical textures, backgrounds, and display windowing can produce misleading
  metric behavior. Scores are useful for relative comparison and explanation, not
  for clinical thresholds.
- No medical IQA weights were trained or fine-tuned for this project.
- Clinical validation requires a labeled clinical dataset, a defined protocol, and
  statistical analysis. This repository provides evaluation plumbing, not validation.

---

## Privacy and safety

Implemented safeguards:

- DICOM ingestion reads only an allow-list of non-identifying technical attributes.
- Full DICOM headers are never logged or returned.
- File paths are redacted in agent and API log messages. Batch result files and the
  optional execution log still record input paths, so protect those artifacts.
- Rendered PNGs are written to a configurable local cache.
- The README documents local OpenAI-compatible endpoints for on-host inference.
- Prompts explicitly prohibit diagnoses and treatment advice.

Important limits:

- **Burned-in PHI remains in pixels.** Header stripping does not remove text or
  overlays embedded in the image.
- **External VLM providers receive image data.** That is a data transfer. Use a
  local endpoint if the data cannot leave your environment.
- **Path names can contain PHI.** Redaction covers this project's logs, not every
  downstream system.
- **Access control, audit logging, retention, encryption, BAAs, and legal review
  are organizational responsibilities.** This project does not certify HIPAA, GDPR,
  or any other regulatory regime.

Do not use this project as the sole quality gate for clinical care.

---

## Configuration

- `configs/model_backends.yaml` — VLM backends, temperatures, optional base URLs.
- `configs/pipeline.yaml` — replanning, executor subtasks, logging, evaluation, and
  a `medical` section. The renderer reads `AGENTIC_RENDER_CACHE` first and falls
  back to `medical.render_cache_dir`. The remaining medical entries record fixed
  safety invariants rather than optional toggles:

```yaml
medical:
  enabled: true
  render_cache_dir: ${AGENTIC_RENDER_CACHE:-data/processed/render_cache}
  classical_qc: true
  fuse_classical_qc_into_score: false
  redact_paths_in_logs: true
  strip_dicom_identifiers: true
```

Runtime backend override:

```bash
python run_pipeline.py \
  --input examples/medical_samples.jsonl \
  --output results/output.jsonl \
  --backend-override planner.backend=openai.gpt-4o-mini
```

To hide unavailable or heavy tools from the Executor's selection prompt, set a
comma-separated exclusion list:

```bash
export AGENTIC_EXCLUDED_TOOLS=QAlign,SomeOtherTool
```

Excluded tools remain in the registry file but are not offered to the VLM, and
they cannot be selected for execution.

---

## Evaluation

The existing evaluation scripts can be used with user-provided ground truth:

```bash
# SRCC / PLCC against reference scores (reads summarizer_result.quality_score first)
python scripts/eval_correlation.py \
  --input results/medical_output.jsonl \
  --ground-truth data/processed/my_benchmark/ground_truth.jsonl

# Multiple-choice accuracy, if your manifest has MCQ labels
python scripts/eval_mcq_accuracy.py --input results/medical_output.jsonl
```

For a full medical-IQA pilot report (SRCC, PLCC, Kendall tau, MAE, RMSE, ordinal
accuracy, quadratic weighted kappa, confusion matrix, per-class recall, and
bootstrap confidence intervals):

```bash
python scripts/eval_medical_iqa.py \
  --input results/drimdb_30.jsonl \
  --ground-truth data/processed/drimdb/ground_truth.jsonl \
  --output results/drimdb_30_eval.json
```

The evaluator joins predictions and labels by `sample_id`. Prediction scores come
from `summarizer_result.quality_score` in `auto` mode, with numeric
`final_answer`, top-level score fields, and the mean normalized tool score as
fallbacks. `--score-source` selects one source explicitly for ablations.

### DRIMDB pilot

[DRIMDB](https://www.researchgate.net/publication/282641760_DRIMDB_Diabetic_Retinopathy_Images_Database_Database_for_Quality_Testing_of_Retinal_Images)
is a public retinal fundus image quality benchmark with 216 images labeled
**Good (125)**, **Bad (69)**, and **Outlier (22)**. Because it has categorical
labels rather than a continuous MOS, the preparation script maps the ordered
classes to a pseudo-MOS:

| DRIMDB label | Pseudo-MOS |
|---|---|
| Good | 5.0 |
| Bad | 3.0 |
| Outlier | 1.0 |

This mapping supports rank correlation only. It is not a clinical score, and the
spacing is arbitrary. MAE, RMSE, and accuracy are secondary descriptors. The
Outlier class includes non-retinal images, so treating it as the lowest quality
level is a protocol choice; the evaluator supports `--exclude-labels Outlier` for
a Good-vs-Bad sensitivity analysis.

The default pilot uses 10 images per class (30 total) with a fixed seed:

```bash
# One-time data download (Kaggle API credentials required)
curl -L -o drimdb.zip \
  https://www.kaggle.com/api/v1/datasets/download/subhajournal/drimdb-diabetic-retinopathy-images-database
unzip -q drimdb.zip -d data/raw/drimdb

# Build a deterministic stratified subset, labels, and flat image directory
python scripts/data_prep/prepare_drimdb.py \
  --data-dir data/raw/drimdb/DRIMDB \
  --per-class 10 \
  --seed 42

# Run the pipeline with hosted VLM inference; QAlign needs a 7B model and is
# excluded from tool selection for CPU-only runs.
export AGENTIC_EXCLUDED_TOOLS=QAlign
python run_pipeline.py \
  --input data/processed/drimdb/manifest.jsonl \
  --output results/drimdb_30.jsonl \
  --max-replan 1 \
  --backend-override planner.backend=openai.gpt-4o-mini \
  --backend-override executor.backend=openai.gpt-4o-mini \
  --backend-override summarizer.backend=openai.gpt-4o-mini

# Evaluate rank correlation and ordinal agreement
python scripts/eval_medical_iqa.py \
  --input results/drimdb_30.jsonl \
  --ground-truth data/processed/drimdb/ground_truth.jsonl \
  --output results/drimdb_30_eval.json

# Optional sensitivity analysis: drop Outlier images, which include non-retinal
# content and make the ordinal placement of that class debatable.
python scripts/eval_medical_iqa.py \
  --input results/drimdb_30.jsonl \
  --ground-truth data/processed/drimdb/ground_truth.jsonl \
  --exclude-labels Outlier \
  --output results/drimdb_30_eval_good_bad.json
```

On Windows PowerShell, set `$env:AGENTIC_EXCLUDED_TOOLS = "QAlign"` instead of
using `export`.

Notes:

- Use only the 216 top-level originals. Some redistributions include a nested
  `iwebalbumfiles` directory with thumbnails and album copies; the preparation
  script ignores it.
- `AGENTIC_EXCLUDED_TOOLS` is a comma-separated list of tool names hidden from
  the Executor's tool-selection prompt.
- Thirty images is a pilot, not a benchmark result. Confidence intervals will be
  wide, and the subset is not a substitute for the full 216-image evaluation.
- Report only the data and protocol actually evaluated. Do not describe these
  results as clinical validation.

---

## Testing

```bash
# Windows PowerShell
$env:AGENTIC_ROOT=(Get-Location).Path
python -m pytest tests -q
```

Medical-specific tests cover modality profile resolution, medical I/O helpers, and
classical QC indicators. DICOM/NIfTI round-trip tests skip automatically when
`pydicom` or `nibabel` is not installed, so install both to exercise the full
ingestion path.

---

## Project structure

```text
medical-agentic-iqa/
├── .env.example
├── configs/
│   ├── api.yaml
│   ├── model_backends.yaml
│   └── pipeline.yaml
├── data/
│   └── processed/render_cache/     # cached PNGs rendered from DICOM/NIfTI
├── examples/
│   ├── api_client_example.py
│   └── medical_samples.jsonl
├── iqa_tools/
│   └── metadata/tools.json
├── scripts/
│   ├── check_env.py
│   ├── data_prep/prepare_drimdb.py  # DRIMDB pilot subset + labels
│   ├── eval_correlation.py
│   ├── eval_medical_iqa.py          # SRCC/MAE/QWK report with bootstrap CIs
│   ├── run_api.py
│   └── verify_iqa_tools.py
├── src/
│   ├── agentic/
│   │   ├── graph.py
│   │   ├── medical_domain.py       # modality profiles and rubrics
│   │   ├── medical_io.py           # DICOM/NIfTI rendering and metadata
│   │   ├── medical_metrics.py      # classical QC indicators
│   │   ├── privacy.py              # log/path redaction helpers
│   │   ├── state.py
│   │   ├── tool_registry.py
│   │   ├── vlm_client.py
│   │   └── nodes/
│   │       ├── planner.py
│   │       ├── executor.py
│   │       └── summarizer.py
│   └── api/
├── tests/
├── run_pipeline.py
└── requirements.txt
```

---

## Troubleshooting

### DICOM cannot be decoded

- Install `pydicom` and ensure the file contains pixel data.
- Compressed transfer syntaxes may need `pylibjpeg`, `pylibjpeg-openjpeg`,
  `pylibjpeg-libjpeg`, or `pylibjpeg-rle`.
- Multi-frame files use a representative middle frame, not a full study.

### NIfTI looks unexpected

- Confirm the volume orientation and slice axis.
- The pipeline uses the middle volume for 4D data and the middle slice for 3D data.
- Intensity normalization is percentile-based and may differ from a clinical viewer.

### Tool weights fail to download

- Check network access and the IQA-PyTorch installation.
- Retry with `--skip-slow`.
- Place weights in the expected IQA-PyTorch cache if your environment is offline.

### GPU out of memory

- Use a hosted VLM backend instead of a local model.
- Set `resources.gpu.enable: false` in `configs/pipeline.yaml` for CPU execution.
- Run fewer or smaller tools.

### API keys are missing

- Confirm `.env` contains the key for the configured backend.
- Confirm the backend name starts with `openai.`, `anthropic.`, or `google.`.

### Scores look wrong on medical images

- Expect this. The metrics are natural-image proxies.
- Treat scores as relative evidence, not calibrated clinical measurements.
- Recalibrate on modality-specific labeled data before any quantitative use.

---

## Limitations and non-goals

Out of scope:

- Live PACS or DICOMweb integration.
- Full study/series traversal and automated series selection.
- Medical-specific model training, fine-tuning, or new learned quality weights.
- Clinical validation or regulatory clearance.
- Certified PHI redaction, HIPAA/GDPR compliance, or medical-device status.

In scope for future work:

- DICOM series folders with representative-slice selection.
- NIfTI multi-slice sampling and orientation controls.
- Local-only endpoint enforcement.
- Modality-specific evaluation manifests and calibration reports.

---

## Attribution

This project is an unofficial reproduction and medical adaptation inspired by the
AgenticIQA paper and the IQA-PyTorch ecosystem. Refer to the original paper and
upstream repositories for their citations and licenses. The bundled tool
coefficients and model weights retain their original terms.

The DRIMDB pilot uses the public Diabetic Retinopathy Image Database described by
Şevik et al., *Journal of Biomedical Optics* (2014). Check the original dataset
terms before redistribution.
