"""
Optimal Proximity Classifier for Acoustic Data
Final Model: Gradient Boosting on 14 carefully selected features
Achieved Accuracy: 86.29% (vs 69.66% with ping width alone)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import pickle
import joblib

# ============================================================================
# OPTIMAL FEATURE SET (from RFE analysis)
# ============================================================================
OPTIMAL_FEATURES = [
    'H0_FILTERED_late_window_energy',
    'H0_FILTERED_peak_floor_ratio',
    'H0_FILTERED_rise_time_ms',
    'H0_FILTERED_spectral_centroid_hz',
    'H0_FILTERED_time_to_secondary_peak_ms',
    'H0_FILTERED_total_energy',
    'H0_RAW_fwhm_ms',
    'H0_RAW_peak_amplitude',
    'H0_RAW_rise_time_ms',
    'H0_RAW_secondary_peak_count',
    'H0_RAW_spectral_centroid_hz',
    'H0_RAW_spectral_flatness',
    'H0_RAW_time_to_secondary_peak_ms',
    'H0_RAW_total_energy',
]

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
csv_path = r'c:\Users\Saagar\Desktop\Acoustics\acoustics-v3\analysis\analysis_2026-04-10--04-22-31.csv'
df = pd.read_csv(csv_path)

h0_data = df[df['CLOSEST_HYDROPHONE'] == 'H0'].copy()

def parse_distance(dist_str):
    try:
        return int(str(dist_str).replace('FT', ''))
    except:
        return np.nan

h0_data['distance_ft'] = h0_data['DISTANCE'].apply(parse_distance)
h0_data['is_nearby'] = (h0_data['distance_ft'] <= 20).astype(int)

h0_feature_cols = [col for col in df.columns if col.startswith('H0_') and col not in 
                   ['H0_IS_NEARBY', 'H0_VALID', 'H0_REASON', 'H0_TOA']]
X = h0_data[h0_feature_cols].copy()
y = h0_data['is_nearby'].copy()

# Handle missing values
missing_pct = (X.isna().sum() / len(X)) * 100
high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
X = X.drop(columns=high_missing_cols)
valid_idx = X.notna().all(axis=1)
X = X[valid_idx]
y = y[valid_idx]

# Select optimal features
X_optimal = X[OPTIMAL_FEATURES].copy()

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_optimal)
X_scaled = pd.DataFrame(X_scaled, columns=OPTIMAL_FEATURES)

print("="*80)
print("ACOUSTIC PROXIMITY CLASSIFIER - FINAL MODEL")
print("="*80)
print(f"\nDataset: {len(X_scaled)} samples")
print(f"Features: {len(OPTIMAL_FEATURES)} selected features")
print(f"Class distribution - Nearby: {(y==1).sum()}, Far: {(y==0).sum()}")

# ============================================================================
# MODEL TRAINING AND VALIDATION
# ============================================================================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    'Gradient Boosting': {
        'model': GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                           random_state=42, subsample=0.8),
        'params': {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 5}
    },
    'Random Forest': {
        'model': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        'params': {'n_estimators': 100, 'max_depth': 10}
    }
}

print("\n" + "="*80)
print("MODEL PERFORMANCE (5-Fold Cross-Validation)")
print("="*80)

results = {}
for name, config in models.items():
    scores = cross_val_score(config['model'], X_scaled, y, cv=cv, scoring='accuracy')
    mean_score = scores.mean()
    std_score = scores.std()
    results[name] = {'mean': mean_score, 'std': std_score, 'scores': scores}
    
    print(f"\n{name}:")
    print(f"  Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")
    print(f"  Fold scores: {', '.join([f'{s:.4f}' for s in scores])}")

best_model_name = max(results.items(), key=lambda x: x[1]['mean'])[0]
best_model_config = models[best_model_name]

print("\n" + "="*80)
print(f"SELECTED MODEL: {best_model_name}")
print("="*80)
print(f"Best accuracy: {results[best_model_name]['mean']:.4f}")
print(f"Model hyperparameters: {best_model_config['params']}")

# ============================================================================
# TRAIN FINAL MODEL
# ============================================================================
final_model = best_model_config['model']
final_model.fit(X_scaled, y)

# Get feature importances
feature_importances = pd.DataFrame({
    'feature': OPTIMAL_FEATURES,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*80)
print("FEATURE IMPORTANCE IN FINAL MODEL")
print("="*80)
print(feature_importances.to_string(index=False))

# ============================================================================
# SAVE MODEL AND CONFIGURATION
# ============================================================================
model_config = {
    'model_type': best_model_name,
    'features': OPTIMAL_FEATURES,
    'scaler': scaler,
    'model': final_model,
    'accuracy': results[best_model_name]['mean'],
    'accuracy_std': results[best_model_name]['std'],
}

# Save using joblib (better for sklearn objects)
joblib.dump(model_config, 'proximity_classifier_final.pkl')
print(f"\nModel saved to: proximity_classifier_final.pkl")

# ============================================================================
# PERFORMANCE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPARISON WITH BASELINES")
print("="*80)

# Ping width baseline
ping_width_data = h0_data[['H0 PING_WIDTH']].iloc[valid_idx].values
scaler_ping = StandardScaler()
ping_scaled = scaler_ping.fit_transform(ping_width_data)
ping_scores = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42),
                               ping_scaled, y, cv=cv, scoring='accuracy')

# All features baseline
X_all = X.copy()
X_all_scaled = StandardScaler().fit_transform(X_all)
all_scores = cross_val_score(GradientBoostingClassifier(n_estimators=100, random_state=42),
                              X_all_scaled, y, cv=cv, scoring='accuracy')

print(f"\nPing Width Only (baseline):     {ping_scores.mean():.4f} (+/- {ping_scores.std():.4f})")
print(f"All 52 features:                {all_scores.mean():.4f} (+/- {all_scores.std():.4f})")
print(f"Optimal 14 features:            {results[best_model_name]['mean']:.4f} (+/- {results[best_model_name]['std']:.4f})")

improvement = (results[best_model_name]['mean'] - ping_scores.mean()) / ping_scores.mean() * 100
print(f"\nImprovement over ping width only: +{improvement:.1f}%")

# ============================================================================
# USAGE INSTRUCTIONS
# ============================================================================
print("\n" + "="*80)
print("USAGE INSTRUCTIONS")
print("="*80)
print("""
To use this model in production:

1. Load the model:
   import joblib
   config = joblib.load('proximity_classifier_final.pkl')
   model = config['model']
   scaler = config['scaler']
   features = config['features']

2. Prepare your data:
   X_new = your_data[features]  # Select only these features
   X_scaled = scaler.transform(X_new)

3. Make predictions:
   predictions = model.predict(X_scaled)  # 1 = Nearby, 0 = Far
   probabilities = model.predict_proba(X_scaled)  # Get confidence

4. Interpret predictions:
   - prediction = 1: Device is NEARBY (distance <= 20 ft)
   - prediction = 0: Device is FAR (distance > 20 ft)
""")

print("="*80)
