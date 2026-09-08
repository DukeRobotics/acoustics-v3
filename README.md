# Hydrophone Proximity Analysis

Python tools for recording four-channel hydrophone data with Saleae Logic 2 and classifying whether an acoustic source is nearby.

The current pipeline is Logic 2 only. It supports live voting, offline analysis of saved captures, and training a nearby/far Random Forest model.

## Pipeline

```text
Logic 2 capture
    -> Logic 2 capture directory
    -> HydrophoneArray
    -> TOA envelope analysis
    -> nearby model inference
    -> result dictionaries or analysis CSV
```

Hydrophones are processed independently. The controller currently combines results with simple rules:

- Every selected hydrophone must pass TOA validation.
- Any selected hydrophone classified as nearby makes the recording nearby.
- The default selection is H0 only.

## Requirements

- Python 3.10 or newer
- Saleae Logic 2 and the Saleae automation package
- A Logic 2 device, or the Logic 2 simulation device when using mock mode

Install Python dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

## Recording Data

Edit the configuration at the bottom of [scripts/recorder.py](scripts/recorder.py), then run:

```powershell
python scripts/recorder.py
```

The recorder creates a timestamped test directory containing one Logic 2 capture directory per epoch:

```text
data/
  7.9.2026/
    H0_20ft_32kHz_2026-07-09--13-39-18/
      H0_20ft_32kHz_epoch_0/
        ... Logic 2 channel files ...
      H0_20ft_32kHz_epoch_1/
        ...
```

`recorder.py` captures binary Logic 2 data for channels 0 through 3. Set `is_mock=True` in its configuration when a simulated device is required.

## Live Detection

[scripts/controller.py](scripts/controller.py) captures continuously, analyzes completed captures in worker processes, and stops after enough matching votes are collected.

Run it from the repository root:

```powershell
python scripts/controller.py
```

Important configuration values are near the top of the file:

- `SELECTED`: hydrophones to analyze, for example `[True, False, False, False]` for H0 only.
- `ANALYZERS`: the configured TOA and nearby analyzers.
- `SAMPLING_FREQ`: capture sample rate.
- `CAPTURE_TIME`: duration of each capture.
- `MAX_CONCURRENT_ANALYSIS_THREADS`: maximum number of simultaneous sample analyses.
- `USE_MOCK_DEVICE`: use the Logic 2 simulation device instead of hardware.

The active nearby model is loaded from:

```text
scripts/artifacts/proximity_classifier_10ft_threshold_2026-04-12--23-04-00.pkl
```

The Logic 2 adapter currently contains a Linux AppImage path in [scripts/logic/logic2.py](scripts/logic/logic2.py). Update that path for a different deployment environment before using real hardware there.

## Offline Analysis

Use [scripts/parser.py](scripts/parser.py) to analyze previously recorded test directories and write a CSV report:

1. Edit `paths_to_analyze` at the bottom of the file.
2. Run:

```powershell
python scripts/parser.py
```

Reports are written to `analysis/analysis_<timestamp>.csv`.

Each row represents one epoch and includes:

- Capture path and folder-derived ground truth metadata.
- Overall validity and nearby result.
- Per-hydrophone TOA and validation reason.
- Nearby prediction and confidence.
- The three model features.

## Nearby Model

The active model uses a 10-foot boundary:

```text
nearby: <= 10 ft
far:    > 10 ft
```

The model uses three H0 features:

1. `RAW_spectral_flatness`
2. `FILTERED_spectral_centroid_hz`
3. `RAW_rise_time_ms`

Train a new model from an analysis CSV with:

```powershell
python scripts/general_utilities/train_nearby_model.py
```

The training script reads the configured CSV, keeps rows where `ALL_VALID` is true, performs five-fold stratified cross-validation, trains a Random Forest, and saves the serialized model package to `scripts/artifacts/`.

Check predictions against folder-distance ground truth with:

```powershell
python scripts/general_utilities/check_accuracy.py
```

Update the CSV path at the bottom of that script before running it.

## Signal Processing

### TOA envelope analysis

`TOAEnvelopeAnalyzer` applies a Butterworth bandpass filter, computes a Hilbert envelope, and selects the first threshold crossing as the time of arrival. It falls back to the largest envelope peak when no threshold crossing is found.

### Garbage validation

The current `GarbageDetector` is used by the TOA analyzer. A selected hydrophone is invalid when:

- Its signal peak is below the configured raw-signal threshold.
- Its TOA is within the front recording margin.
- Its TOA is within the end recording margin.

The default controller thresholds are a raw-signal threshold of `0.5` and timing margins of `0.1` seconds.

## Project Layout

```text
scripts/
  controller.py                 Live capture, analysis, and voting
  recorder.py                   Batch Logic 2 acquisition
  parser.py                     Offline capture analysis and CSV output
  analyzers/                    TOA and nearby analysis
  hydrophones/                  Logic 2 loading and hydrophone data containers
  logic/logic2.py               Saleae Logic 2 interface
  general_utilities/            Model training and accuracy checks
  artifacts/                    Serialized nearby models
data/                            Recorded Logic 2 captures
analysis/                        Generated analysis CSV files
```

## Current Scope

- Logic 2 only.
- Four-channel capture support.
- Per-hydrophone analysis with simple result aggregation.
- Default runtime analysis uses H0 only.
- No cross-hydrophone timing, localization, triangulation, or array-level signal comparison yet.
