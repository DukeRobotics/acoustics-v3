"""
Train model with tighter threshold: nearby = <= 10ft (instead of <= 20ft)
Evaluate performance at the new decision boundary
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load and prepare data
csv_path = r'c:\Users\Saagar\Desktop\Acoustics\acoustics-v3\analysis\analysis_2026-04-10--04-22-31.csv'
df = pd.read_csv(csv_path)
h0_data = df[df['CLOSEST_HYDROPHONE'] == 'H0'].copy()

def parse_distance(dist_str):
    try:
        return int(str(dist_str).replace('FT', ''))
    except:
        return np.nan

h0_data['distance_ft'] = h0_data['DISTANCE'].apply(parse_distance)

# TIGHTER THRESHOLD: 10ft instead of 20ft
h0_data['is_nearby'] = (h0_data['distance_ft'] <= 10).astype(int)

h0_feature_cols = [col for col in df.columns if col.startswith('H0_') and col not in 
                   ['H0_IS_NEARBY', 'H0_VALID', 'H0_REASON', 'H0_TOA']]
X = h0_data[h0_feature_cols].copy()
y = h0_data['is_nearby'].copy()

# Keep the same 7 features
OPTIMAL_FEATURES = [
    'H0_RAW_spectral_flatness',
    'H0_FILTERED_spectral_centroid_hz',
    'H0_FILTERED_time_to_secondary_peak_ms',
    'H0_RAW_rise_time_ms',
    'H0_RAW_spectral_centroid_hz',
    'H0_FILTERED_rise_time_ms',
    'H0_FILTERED_fwhm_ms',
]

# Select only optimal features first
X = X[OPTIMAL_FEATURES]

# Handle missing values
missing_pct = (X.isna().sum() / len(X)) * 100
valid_idx = X.notna().all(axis=1)
X = X[valid_idx]
y = y[valid_idx]

print("="*100)
print("MODEL TRAINING: TIGHTER THRESHOLD (nearby = <= 10ft)")
print("="*100)

print(f"\nDataset: {len(X)} samples, {len(OPTIMAL_FEATURES)} features")
print(f"Target distribution:")
print(f"  Nearby (<= 10ft):  {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")
print(f"  Far (> 10ft):      {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print(f"\nOptimal Features ({len(OPTIMAL_FEATURES)}):")
print("-" * 100)
for i, feat in enumerate(OPTIMAL_FEATURES, 1):
    print(f"  {i}. {feat}")

# Cross-validation evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*100)
print("5-FOLD STRATIFIED CROSS-VALIDATION")
print("="*100)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
rf_scores = cross_val_score(rf_model, X_scaled, y, cv=cv, scoring='accuracy')

print(f"\nRandom Forest (7 features, 10ft threshold):")
print(f"  Mean Accuracy: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
print(f"  Fold scores: {[f'{s:.4f}' for s in rf_scores]}")

# Train final model on full dataset
print("\n" + "="*100)
print("TRAINING FINAL MODEL ON FULL DATASET")
print("="*100)

best_model = rf_model
best_model.fit(X_scaled, y)

# Get feature importance
feature_importance = pd.DataFrame({
    'feature': OPTIMAL_FEATURES,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nFeature Importance:")
for i, (idx, row) in enumerate(feature_importance.iterrows(), 1):
    pct = (row['importance'] / feature_importance['importance'].sum()) * 100
    print(f"  {i}. {row['feature']:50s} {row['importance']:.4f} ({pct:.1f}%)")

# Test on full data
y_pred = best_model.predict(X_scaled)
y_pred_proba = best_model.predict_proba(X_scaled)[:, 1]

cm = confusion_matrix(y, y_pred)
accuracy = (y_pred == y).mean()
auc = roc_auc_score(y, y_pred_proba)

print("\n" + "="*100)
print("FULL DATASET PERFORMANCE")
print("="*100)
print(f"\nAccuracy: {accuracy:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"\nConfusion Matrix:")
print(f"  True Negatives (correctly identified FAR):     {cm[0,0]:3d}")
print(f"  False Positives (incorrectly called NEARBY):   {cm[0,1]:3d}")
print(f"  False Negatives (incorrectly called FAR):      {cm[1,0]:3d}")
print(f"  True Positives (correctly identified NEARBY):  {cm[1,1]:3d}")

print("\nClassification Report:")
print(classification_report(y, y_pred, target_names=['far (>10ft)', 'nearby (<=10ft)']))

# Performance by distance
print("\n" + "="*100)
print("PERFORMANCE BY DISTANCE")
print("="*100)

distances = sorted(h0_data[valid_idx]['distance_ft'].unique())
print(f"\n{'DISTANCE':>12s} {'SAMPLES':>8s} {'CLASS':>10s} {'CORRECT':>8s} {'ACCURACY':>10s}")
print("-" * 100)

for dist in distances:
    mask = h0_data[valid_idx]['distance_ft'] == dist
    if mask.sum() > 0:
        mask = mask.values
        acc = (y_pred[mask] == y.values[mask]).mean()
        n_correct = (y_pred[mask] == y.values[mask]).sum()
        n_total = mask.sum()
        nearby = "NEARBY" if dist <= 10 else "FAR"
        print(f"{dist:>11.0f}ft {n_total:>8d} {nearby:>10s} {n_correct:>8d} {acc:>10.2%}")

# Save model
model_path = r'c:\Users\Saagar\Desktop\Acoustics\acoustics-v3\proximity_classifier_10ft_threshold.pkl'
model_package = {
    'model': best_model,
    'scaler': scaler,
    'features': OPTIMAL_FEATURES,
    'accuracy_cv': rf_scores.mean(),
    'accuracy_full': accuracy,
    'model_type': 'Random Forest',
    'threshold_ft': 10,
}

joblib.dump(model_package, model_path)
print(f"\nModel saved to: {model_path}")

print("\n" + "="*100)
print("COMPARISON: 10ft vs 20ft Threshold")
print("="*100)
print(f"""
10ft Threshold (TIGHTER):
  - Cross-validation accuracy: {rf_scores.mean():.2%}
  - Full dataset accuracy: {accuracy:.2%}
  - Nearby samples: {(y==1).sum()} (more restricted)
  - Far samples: {(y==0).sum()}

20ft Threshold (ORIGINAL):
  - Cross-validation accuracy: 87.33%
  - Full dataset accuracy: 99.79%
  - Nearby samples: 287 (wider range)
  - Far samples: 194

The tighter 10ft threshold makes classification harder because:
- Fewer samples in "nearby" class (smaller positive set)
- Less separation between 10-20ft and >20ft regions
- More samples near the decision boundary
""")

print("="*100)
