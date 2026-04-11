"""
ACOUSTIC PROXIMITY CLASSIFIER - PRACTICAL USAGE GUIDE

This script demonstrates how to use the trained proximity classifier
to determine if a hydrophone is nearby (≤20ft) or far (>20ft).
"""

import pandas as pd
import numpy as np
import joblib

# ============================================================================
# STEP 1: LOAD THE TRAINED MODEL
# ============================================================================
print("Loading trained model...")
config = joblib.load('proximity_classifier_final.pkl')

model = config['model']
scaler = config['scaler']
optimal_features = config['features']

print(f"✓ Model loaded successfully")
print(f"  Type: {config['model_type']}")
print(f"  Accuracy: {config['accuracy']:.2%}±{config['accuracy_std']:.2%}")
print(f"  Features: {len(optimal_features)}")

# ============================================================================
# STEP 2: PREPARE NEW DATA FOR PREDICTION
# ============================================================================
print("\n" + "="*80)
print("PREPARING DATA FOR CLASSIFICATION")
print("="*80)

# Load the analysis data
csv_path = r'c:\Users\Saagar\Desktop\Acoustics\acoustics-v3\analysis\analysis_2026-04-10--04-22-31.csv'
df = pd.read_csv(csv_path)

# Filter for H0 data
h0_data = df[df['CLOSEST_HYDROPHONE'] == 'H0'].copy()

# Handle distance parsing
def parse_distance(dist_str):
    try:
        return int(str(dist_str).replace('FT', ''))
    except:
        return np.nan

h0_data['distance_ft'] = h0_data['DISTANCE'].apply(parse_distance)
h0_data['is_nearby_actual'] = (h0_data['distance_ft'] <= 20).astype(int)

# Get all features from the data
h0_feature_cols = [col for col in df.columns if col.startswith('H0_') and 
                   col not in ['H0_IS_NEARBY', 'H0_VALID', 'H0_REASON', 'H0_TOA']]
X_full = h0_data[h0_feature_cols].copy()

# Handle missing values
valid_idx = X_full.notna().all(axis=1)
X_valid = X_full[valid_idx].copy()

# Select ONLY the optimal features
X_selected = X_valid[optimal_features].copy()

print(f"\n✓ Data prepared:")
print(f"  Total records: {len(h0_data)}")
print(f"  Valid records (no missing): {len(X_selected)}")
print(f"  Features used: {len(optimal_features)}/{len(h0_feature_cols)}")

# ============================================================================
# STEP 3: NORMALIZE THE DATA
# ============================================================================
print("\nNormalizing features using pre-trained scaler...")
X_scaled = scaler.transform(X_selected)
print(f"✓ Data normalized (mean≈0, std≈1)")

# ============================================================================
# STEP 4: MAKE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("MAKING PREDICTIONS")
print("="*80)

# Get predictions
predictions = model.predict(X_scaled)  # 1=Nearby, 0=Far
probabilities = model.predict_proba(X_scaled)  # Confidence scores

# Get actual distances for comparison
actual_distances = h0_data.loc[valid_idx, 'distance_ft'].values
actual_nearby = h0_data.loc[valid_idx, 'is_nearby_actual'].values

print(f"\n✓ Predictions complete: {len(predictions)} records classified")

# ============================================================================
# STEP 5: ANALYZE ACCURACY & RESULTS
# ============================================================================
print("\n" + "="*80)
print("ACCURACY ANALYSIS")
print("="*80)

accuracy = np.mean(predictions == actual_nearby)
tp = np.sum((predictions == 1) & (actual_nearby == 1))  # True Positives
tn = np.sum((predictions == 0) & (actual_nearby == 0))  # True Negatives
fp = np.sum((predictions == 1) & (actual_nearby == 0))  # False Positives
fn = np.sum((predictions == 0) & (actual_nearby == 1))  # False Negatives

sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"\nConfusion Matrix:")
print(f"  True Positives (Nearby, Correctly): {tp}")
print(f"  True Negatives (Far, Correctly): {tn}")
print(f"  False Positives (Far, Predicted Nearby): {fp}")
print(f"  False Negatives (Nearby, Predicted Far): {fn}")
print(f"\nSensitivity (Recall - True Positive Rate): {sensitivity:.4f}")
print(f"Specificity (True Negative Rate): {specificity:.4f}")

# ============================================================================
# STEP 6: DETAILED PREDICTIONS WITH CONFIDENCE
# ============================================================================
print("\n" + "="*80)
print("SAMPLE PREDICTIONS (First 10 Records)")
print("="*80)

results_df = pd.DataFrame({
    'Distance_ft': actual_distances,
    'Actual_Class': ['NEARBY' if x == 1 else 'FAR' for x in actual_nearby],
    'Predicted_Class': ['NEARBY' if x == 1 else 'FAR' for x in predictions],
    'Confidence_Nearby': probabilities[:, 1],
    'Confidence_Far': probabilities[:, 0],
    'Correct': predictions == actual_nearby
})

print("\n" + results_df.head(10).to_string(index=False))
print(f"\n... and {len(results_df)-10} more records")

# ============================================================================
# STEP 7: CONFIDENCE DISTRIBUTION
# ============================================================================
print("\n" + "="*80)
print("CONFIDENCE SCORE DISTRIBUTION")
print("="*80)

confidence_nearby = probabilities[:, 1]

print(f"\nConfidence for NEARBY predictions (should be >0.5):")
print(f"  Min: {confidence_nearby.min():.4f}")
print(f"  Max: {confidence_nearby.max():.4f}")
print(f"  Mean: {confidence_nearby.mean():.4f}")
print(f"  Std: {confidence_nearby.std():.4f}")

high_confidence = np.sum(confidence_nearby > 0.8)
medium_confidence = np.sum((confidence_nearby >= 0.5) & (confidence_nearby <= 0.8))
low_confidence = np.sum(confidence_nearby < 0.5)

print(f"\nConfidence levels:")
print(f"  High (>0.8): {high_confidence} ({high_confidence/len(confidence_nearby)*100:.1f}%)")
print(f"  Medium (0.5-0.8): {medium_confidence} ({medium_confidence/len(confidence_nearby)*100:.1f}%)")
print(f"  Low (<0.5): {low_confidence} ({low_confidence/len(confidence_nearby)*100:.1f}%)")

# ============================================================================
# STEP 8: DISTANCE VS CONFIDENCE
# ============================================================================
print("\n" + "="*80)
print("DISTANCE VS PREDICTION CONFIDENCE")
print("="*80)

# Group by actual distance and analyze
distance_bins = [0, 10, 20, 30, 40, 50, 100]
for i in range(len(distance_bins)-1):
    mask = (actual_distances >= distance_bins[i]) & (actual_distances < distance_bins[i+1])
    if mask.sum() > 0:
        bin_conf = confidence_nearby[mask]
        print(f"\nDistance {distance_bins[i]}-{distance_bins[i+1]} ft: {mask.sum()} samples")
        print(f"  Avg confidence: {bin_conf.mean():.4f}")
        print(f"  Accuracy: {np.mean(predictions[mask] == actual_nearby[mask]):.4f}")

# ============================================================================
# STEP 9: FEATURE IMPORTANCE REMINDER
# ============================================================================
print("\n" + "="*80)
print("MODEL FEATURES (In order of importance)")
print("="*80)

feature_importance = {
    'H0_RAW_spectral_flatness': 0.316,
    'H0_FILTERED_spectral_centroid_hz': 0.152,
    'H0_FILTERED_time_to_secondary_peak_ms': 0.095,
    'H0_RAW_rise_time_ms': 0.075,
    'H0_FILTERED_rise_time_ms': 0.052,
    'H0_RAW_fwhm_ms': 0.046,
    'H0_RAW_spectral_centroid_hz': 0.046,
    'H0_FILTERED_late_window_energy': 0.045,
    'H0_RAW_peak_amplitude': 0.042,
    'H0_RAW_secondary_peak_count': 0.039,
    'H0_FILTERED_peak_floor_ratio': 0.038,
    'H0_RAW_time_to_secondary_peak_ms': 0.022,
    'H0_FILTERED_total_energy': 0.017,
    'H0_RAW_total_energy': 0.015,
}

for i, (feat, imp) in enumerate(feature_importance.items(), 1):
    print(f"  {i:2d}. {feat:45s} {imp:6.1%}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"""
Model Performance:    86.29% accuracy
Features Used:        14 out of 52
Improvement vs Ping:  +29% over ping width baseline

Classification Rule:
  ≤20 feet  → NEARBY (Prediction = 1)
  >20 feet  → FAR    (Prediction = 0)

Key Feature:          H0_RAW_spectral_flatness (31.6% importance)

Ready to Deploy! ✓
""")
