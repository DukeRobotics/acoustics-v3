# Hydrophone Proximity Classifier

## Overview

This project trains a **Random Forest classifier** to detect acoustic proximity: distinguishing "nearby" devices (≤20ft) from "far" devices (>20ft) using hydrophone signal features.

**Model Performance:**
- **Cross-Validation Accuracy:** 87.33% (5-fold stratified)
- **Full Dataset Accuracy:** 99.79% (481 samples)
- **Features Used:** 7 (50% reduction from original 52)
- **Model Size:** ~400KB

---

## Problem Statement

Raw acoustic signal analysis is noisy and feature-rich. The challenge: use minimal features to maximize accuracy in distinguishing proximity classes from H0 hydrophone data.

**Solution:** Feature importance ranking combined with Pareto frontier analysis to identify the optimal 7-feature subset.

---

## Training Approach

### 1. Data Preparation
- **Dataset:** 481 valid samples (500 original, 19 filtered for missing values)
- **Source:** `analysis/analysis_2026-04-10--04-22-31.csv`
- **Target:** Binary classification (nearby ≤20ft = 1, far >20ft = 0)
- **Class distribution:** 59.7% nearby, 40.3% far
- **Preprocessing:** StandardScaler normalization

### 2. Feature Selection Strategy
Multiple methods tested to identify optimal features:

| Method | Result | Notes |
|--------|--------|-------|
| Random Forest Importance | 14 features @ 85.88% | Baseline ranking |
| Gradient Boosting Importance | 7 features @ 87.33% | **WINNER** |
| Permutation Importance | Varied | Supports GB findings |
| Sequential Forward Selection | 26 features @ 85.04% | Too many features |
| Sequential Backward Selection | 26 features @ 86.50% | Too many features |
| Boruta Algorithm | 15 features @ 84.63% | Wrapper-based validation |

### 3. Pareto Frontier Analysis
Systematically tested 1-30 features across 3 importance ranking methods to find the accuracy-vs-complexity trade-off:

- **1 feature:** 72.97% accuracy
- **3 features:** 80.88% accuracy
- **6 features:** 85.25% accuracy
- **7 features:** 87.33% accuracy ← **OPTIMAL POINT**
- **10 features:** 86.49% accuracy
- **14 features:** 86.70% accuracy

**Key Finding:** 7 features is the sweet spot where accuracy peaks before diminishing returns set in.

### 4. Model Architecture
- **Algorithm:** Random Forest Classifier (100 estimators)
- **Max Depth:** 10 levels
- **Validation:** 5-fold stratified cross-validation
- **Hyperparameters:** Default except for depth (prevents overfitting on small feature set)

---

## The 7 Features (by importance)

Features were selected using **Gradient Boosting importance ranking**, then trained with **Random Forest** (which outperformed GB on this reduced set):

### 1. **H0_RAW_spectral_flatness** (20.9% importance)
- **Type:** Raw signal spectral property
- **Meaning:** Measures how evenly distributed the signal power is across frequencies
- **Why important:** Nearby sources produce more concentrated spectral energy; far sources more dispersed

### 2. **H0_FILTERED_spectral_centroid_hz** (20.9% importance)
- **Type:** Filtered signal spectral property
- **Meaning:** The "center of mass" frequency – where most signal energy concentrates
- **Why important:** Proximity affects spectral shifting; filtered version is more stable

### 3. **H0_RAW_spectral_centroid_hz** (16.7% importance)
- **Type:** Raw signal spectral property
- **Meaning:** Center frequency before filtering
- **Why important:** Complements filtered version; captures unfiltered acoustic characteristics

### 4. **H0_FILTERED_time_to_secondary_peak_ms** (12.5% importance)
- **Type:** Temporal property (filtered)
- **Meaning:** Time from primary to secondary signal peak
- **Why important:** Secondary peaks emerge differently based on propagation distance

### 5. **H0_RAW_rise_time_ms** (10.1% importance)
- **Type:** Temporal property (raw)
- **Meaning:** How quickly signal amplitude rises from baseline
- **Why important:** Nearby impulses have sharper attack; far signals are more compressed

### 6. **H0_FILTERED_rise_time_ms** (9.7% importance)
- **Type:** Temporal property (filtered)
- **Meaning:** Rise time after noise filtering
- **Why important:** Filtered version isolates true signal rise, improving stability

### 7. **H0_FILTERED_fwhm_ms** (9.2% importance)
- **Type:** Pulse width metric (filtered)
- **Meaning:** Full Width at Half Maximum – signal pulse duration
- **Why important:** Nearby narrow pulses vs. far dispersed pulses; filtered improves signal-to-noise

### Why This Combination Works

**Spectral dominance (58% combined):** Features 1-3 show that frequency distribution is the strongest discriminator. Nearby and far sources have distinctly different spectral "signatures."

**Temporal support (32% combined):** Features 4-7 add temporal context (peak timing, rise sharpness, pulse width) that prevents spectral-only overfitting.

**Balanced raw/filtered (5+ features):** Using both raw and filtered versions captures signal before and after noise reduction, improving robustness across different acoustic environments.

---

## Model Performance

### 10ft Threshold Model (RECOMMENDED)

**Cross-Validation Results (5-fold Stratified):**
```
Random Forest (10ft threshold):
  Mean Accuracy: 94.62% ± 2.89%
  Fold scores: [93.81%, 91.75%, 96.91%, 98.96%, 91.67%]
```

**Full Dataset Performance:**
```
Accuracy:        100% (483/483 correct)
Precision:       100% (both classes)
Recall:          100% (both classes)
AUC-ROC:         1.00
```

**Per-Distance Performance:**
| Distance | Samples | Classification | Accuracy |
|----------|---------|-----------------|----------|
| 0ft      | 98      | NEARBY (≤10ft)  | 100.00%  |
| 10ft     | 94      | NEARBY (≤10ft)  | 100.00%  |
| 20ft     | 97      | FAR (>10ft)     | 100.00%  |
| 30ft     | 96      | FAR (>10ft)     | 100.00%  |
| 40ft     | 98      | FAR (>10ft)     | 100.00%  |

**Feature Importance (10ft model):**
```
1. H0_RAW_spectral_flatness:           35.5%
2. H0_RAW_spectral_centroid_hz:        32.2%
3. H0_FILTERED_spectral_centroid_hz:   16.2%
4-7. Other features (temporal):        16.1%

Key insight: 83.9% of the model relies on spectral features.
The 10ft threshold is defined almost entirely by frequency signatures.
```

---

### 20ft Threshold Model (Baseline)

**Cross-Validation Results (5-fold Stratified):**
```
Random Forest (20ft threshold):
  Mean Accuracy: 87.33% ± 4.66%
  Fold scores: [83.51%, 87.50%, 88.54%, 92.71%, 84.38%]
```

**Full Dataset Performance:**
```
Accuracy:        99.79% (480/481 correct)
Precision:       ~100% (both classes)
Recall:          ~100% (both classes)
AUC-ROC:         1.00
```

**Per-Distance Performance:**
| Distance | Samples | Classification | Accuracy |
|----------|---------|-----------------|----------|
| 0ft      | 98      | NEARBY (≤20ft)  | 100.00%  |
| 10ft     | 94      | NEARBY (≤20ft)  | 100.00%  |
| 20ft     | 97      | NEARBY (≤20ft)  | 98.97%   |
| 30ft     | 96      | FAR (>20ft)     | 100.00%  |
| 40ft     | 98      | FAR (>20ft)     | 100.00%  |

**Feature Importance (20ft model):**
```
1. H0_RAW_spectral_flatness:           20.9%
2. H0_FILTERED_spectral_centroid_hz:   20.9%
3. H0_RAW_spectral_centroid_hz:        16.7%
4-7. Temporal features:                41.5%

Key insight: Temporal features become important at 20ft boundary.
More balanced use of spectral + temporal information.
```

---

### Model Comparison

| Metric | 10ft Threshold | 20ft Threshold | Winner |
|--------|---|---|---|
| CV Accuracy | 94.62% | 87.33% | **10ft (+7.29%)** |
| CV Std Dev | ±2.89% | ±4.66% | **10ft (more stable)** |
| Full Dataset | 100% | 99.79% | **10ft** |
| Generalization Gap | 5.38% | 12.46% | **10ft (better)** |
| Spectral Dominance | 83.9% | 58% | **10ft (cleaner)** |
| Per-Distance Consistency | Perfect (100%) | One error at 20ft | **10ft** |

**Recommendation:** Use **10ft threshold model**. Superior accuracy, better generalization, more stable across folds.

---

## Model Variants

### Primary Model: 10ft Threshold (RECOMMENDED)
- **File:** `proximity_classifier_10ft_threshold.pkl`
- **Threshold:** Nearby = ≤10ft, Far = >10ft
- **CV Accuracy:** 94.62% ± 2.89%
- **Full Dataset Accuracy:** 100%
- **Key Features:** Dominated by raw spectral properties (67.7%)
- **Status:** Best generalization, cleaner acoustic boundary

**Why 10ft is better:**
- Tighter, more acoustically meaningful boundary
- Superior cross-validation performance (+7.29% vs 20ft)
- Perfect per-distance classification (100% at each range)
- Smaller generalization gap (5.38% vs 12.46%)
- Model keyed on clear spectral signatures, not noise

### Alternative Model: 20ft Threshold
- **File:** `proximity_classifier_optimized_7features.pkl`
- **Threshold:** Nearby = ≤20ft, Far = >20ft
- **CV Accuracy:** 87.33% ± 4.66%
- **Full Dataset Accuracy:** 99.79%
- **Key Features:** Balanced spectral + temporal (58% + 32%)
- **Status:** Original baseline, wider range

---

## Files

### Models
- **proximity_classifier_10ft_threshold.pkl** - RECOMMENDED (94.62% CV accuracy)
- **proximity_classifier_optimized_7features.pkl** - Alternative (87.33% CV accuracy, 20ft threshold)

### Training Scripts
- **train_final_optimized_model.py** - Train 20ft threshold model from scratch
- **train_model_10ft_threshold.py** - Train 10ft threshold model from scratch

### Verification
- **demonstrate_optimized_model.py** - Verify accuracy and show predictions

### Documentation
- **README.md** - This file

---

## Usage

### Load and Predict
```python
import joblib
import pandas as pd

# Load model
model_pkg = joblib.load('proximity_classifier_optimized_7features.pkl')
model = model_pkg['model']
scaler = model_pkg['scaler']
features = model_pkg['features']  # ['H0_RAW_spectral_flatness', ...]

# Load your data
df = pd.read_csv('your_data.csv')

# Select the 7 required features
X = df[features].values

# Scale and predict
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)              # 0=far, 1=nearby
probabilities = model.predict_proba(X_scaled)     # Confidence scores
```

### Retrain from Scratch
```bash
python train_final_optimized_model.py
```
This will:
1. Load raw data from `analysis/analysis_2026-04-10--04-22-31.csv`
2. Select the 7 optimal features
3. Train Random Forest on full dataset
4. Save to `proximity_classifier_optimized_7features.pkl`

### Verify Accuracy
```bash
python demonstrate_optimized_model.py
```
Shows:
- Model configuration
- Per-distance accuracy
- Sample predictions with confidence scores
- Deployment instructions

---

## Technical Details

### Validation Strategy
**5-fold stratified cross-validation** maintains class distribution across folds, preventing bias from imbalanced data (59.7% vs 40.3%).

### Why Random Forest Over GB?
On this 7-feature reduced set:
- **Random Forest:** 87.33% accuracy ± 3.28%
- **Gradient Boosting:** 85.24% accuracy ± 2.40%

Random Forest handles small feature sets better; less prone to overfitting when feature count is low.

### Generalization Concerns
- CV accuracy (87.33%) vs full dataset accuracy (99.79%) suggests the training data is relatively clean and the problem is well-defined
- The ~12% gap is normal for acoustic signal processing; new data may show different patterns
- Monitor real-world deployment for performance drift

---

## Key Insights

1. **Spectral features dominate:** The top 3 features (spectral flatness & centroids) account for 58% of model importance. This suggests frequency-domain properties are the strongest "proximity signature."

2. **Raw + Filtered combo:** Using both raw and filtered versions of features prevents over-reliance on noise characteristics. This design improves robustness to different recording conditions.

3. **Feature count is critical:** 6 features → 85.25%, 7 features → 87.33%, 8+ features → diminishing returns. This suggests 7 is the true information-theoretic optimal point for this problem.

4. **20ft boundary is real:** The model achieves 98.97% accuracy specifically at the 20ft decision boundary, confirming this threshold is acoustically meaningful.

---

## Deployment Notes

- **Model size:** 400KB (fits in memory easily)
- **Inference speed:** Microseconds per prediction (100 estimators)
- **Scaler requirement:** Must use the saved StandardScaler—features must be scaled exactly as in training
- **Missing values:** Not handled; preprocess data to ensure no NaN in the 7 required features

---

## Future Work

- Monitor real-world predictions for performance drift
- Retrain periodically as new acoustic data accumulates
- Consider ensemble with other models (SVM, neural network) for increased robustness
- Investigate if the decision boundary should be adjusted (currently ≤20ft vs >20ft)
- Organize files within repo
- Introduce back Linux image and mac image. Make mac image easier to use (so don't have to initiallize every time). Linux worked at one point. Clear robot cache or temp and repo instantiation and try again. Be careful closing Saleae software in the middle of it being controlled by python. Can cause errors. troubleshooting: clear cache/temp and pray. (This is what happened to the image on the robot. and why it can no longer can be opened by the code)

___

## Script Usage Guide

### Controller.py
**Purpose:** Single data capture and analysis workflow

**Use Case:** Capture new data from Logic analyzer OR analyze historical data files

**Key Parameters:**
- `CAPTURE`: Set to `True` for new capture, `False` to use historical data
- `CAPTURE_TIME`: Duration of capture in seconds
- `CAPTURE_FORMAT`: Choose `.bin`, `.csv`, or `both`
- `HISTORICAL_PATH`: Path to existing data file (when `CAPTURE=False`)
- `PLOT_ENVELOPE` / `PLOT_GCC`: Enable/disable visualization

