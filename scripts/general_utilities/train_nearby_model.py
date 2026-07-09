import pandas as pd
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# CONFIG
CSV_PATH = 'analysis/analysis_2026-04-12--14-13-52.csv'
MODEL_SAVE_PATH = 'artifacts/proximity_classifier_10ft_threshold_2026-04-12--23-04-00.pkl'
THRESHOLD_FT = 10
N_ESTIMATORS = 100
MAX_DEPTH = 10

FEATURES = [
    'H0 RAW_spectral_flatness',
    'H0 FILTERED_spectral_centroid_hz',
    'H0 FILTERED_time_to_secondary_peak_ms',
    'H0 RAW_rise_time_ms',
]

# Load data
df = pd.read_csv(CSV_PATH)
df = df[df['ALL_VALID'] == True]
X = df[FEATURES].copy()
y = (pd.to_numeric(df['DISTANCE'].str.replace('FT', ''), errors='coerce') <= THRESHOLD_FT).astype(int)

print(f"Dataset: {len(X)} samples")
print(f"Nearby (≤{THRESHOLD_FT}ft): {(y==1).sum()}, Far (>{THRESHOLD_FT}ft): {(y==0).sum()}\n")

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = RandomForestClassifier(n_estimators=N_ESTIMATORS, max_depth=MAX_DEPTH, random_state=42, n_jobs=-1)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print(f"CV Accuracy: {cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
for i, score in enumerate(cv_scores, 1):
    print(f"  Fold {i}: {score:.2%}")
print()

# Train on full data
model.fit(X, y)

# Feature importance
imp = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)
print("Feature Importance:")
for feat, score in imp:
    print(f"  {feat}: {score:.4f}")
print()

# Save
model_pkg = {
    'model': model,
    'features': FEATURES,
    'threshold_ft': THRESHOLD_FT,
}
joblib.dump(model_pkg, MODEL_SAVE_PATH)
print(f"Model saved: {MODEL_SAVE_PATH}")
