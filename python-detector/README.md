# VeriTrue Python Detector (Standalone)

This folder contains a modular, production-oriented **AI-generated image detection pipeline**.
It outputs an **AI-likelihood score (0–100)** plus an explanation of which signals contributed.

## What this is (and isn’t)
- This is an **ensemble forensic pipeline** with pluggable evaluators (metadata, signal/noise, model hooks, originality, optional C2PA).
- It **does not magically guarantee accuracy** without trained model weights and representative evaluation data.
- Confidence is **probabilistic**, not absolute.

## Requirements
- Python 3.10+
- Optional (recommended): ExifTool installed and on PATH.
  - Windows: download ExifTool, add `exiftool.exe` to PATH.

## Install
```bash
cd python-detector
python -m venv .venv
.\.venv\Scripts\pip install -U pip
.\.venv\Scripts\pip install -e .
```

Optional extras:
```bash
.\.venv\Scripts\pip install -e ".[api,torch]"
```

## Run (CLI)
```bash
veritrue-detect path\to\image.jpg --json
```

## Evaluate (metrics)
Create a CSV with columns: `path,label` where label is `0` for real and `1` for AI.
```bash
python tools\evaluate.py --csv dataset.csv
```

## Plugging in models
Model-based detectors are implemented as **wrappers** that load user-provided weights.
See `veritrue_detector/models/torch_image_classifier.py`.

