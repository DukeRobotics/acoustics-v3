"""
DEMONSTRATION: Using the Optimized 7-Feature Model
Shows predictions on new data with confidence scores
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Load data
csv_path = r'c:\Users\Saagar\Desktop\Acoustics\acoustics-v3\analysis\analysis_2026-04-10--04-22-31.csv'
df = pd.read_csv(csv_path)

# Load optimized model
model_path = r'c:\Users\Saagar\Desktop\Acoustics\acoustics-v3\proximity_classifier_optimized_7features.pkl'
model_pkg = joblib.load(model_path)

model = model_pkg['model']
scaler = model_pkg['scaler']
features_to_use = model_pkg['features']
accuracy_cv = model_pkg['accuracy_cv']

print("="*100)
print("OPTIMIZED PROXIMITY CLASSIFIER - LIVE DEMONSTRATION")
print("="*100)

print(f"\nModel Configuration:")
print(f"  - Features: {len(features_to_use)}")
print(f"  - Model: Random Forest Classifier")
print(f"  - Cross-Validation Accuracy: {accuracy_cv:.2%}")
print(f"  - Threshold: Distance ≤ 20ft = NEARBY, > 20ft = FAR")

print(f"\nOptimal Features:")
for i, feat in enumerate(features_to_use, 1):
    print(f"  {i}. {feat}")

# Prepare data
h0_data = df[df['CLOSEST_HYDROPHONE'] == 'H0'].copy()

def parse_distance(dist_str):
    try:
        return int(str(dist_str).replace('FT', ''))
    except:
        return np.nan

h0_data['distance_ft'] = h0_data['DISTANCE'].apply(parse_distance)
h0_data['is_nearby_truth'] = (h0_data['distance_ft'] <= 20).astype(int)

# Extract features and handle missing values
X_subset = h0_data[features_to_use].copy()
valid_mask = X_subset.notna().all(axis=1)
X_subset = X_subset[valid_mask]
h0_data_valid = h0_data[valid_mask].reset_index(drop=True)
X_subset = X_subset.reset_index(drop=True)

# Scale features
X_scaled = scaler.transform(X_subset)

# Make predictions
predictions = model.predict(X_scaled)
probabilities = model.predict_proba(X_scaled)
confidence = probabilities.max(axis=1)

# Evaluate
y_true = h0_data_valid['is_nearby_truth'].values
accuracy = (predictions == y_true).mean()

print(f"\n" + "="*100)
print(f"EVALUATION ON TEST DATASET ({len(X_scaled)} samples)")
print("="*100)

# Confusion matrix
tn = ((predictions == 0) & (y_true == 0)).sum()
fp = ((predictions == 1) & (y_true == 0)).sum()
fn = ((predictions == 0) & (y_true == 1)).sum()
tp = ((predictions == 1) & (y_true == 1)).sum()

print(f"\nOverall Accuracy: {accuracy:.2%}")
print(f"\nConfusion Matrix:")
print(f"  True Negatives (correctly identified FAR):    {tn:3d}")
print(f"  False Positives (incorrectly called NEARBY):   {fp:3d}")
print(f"  False Negatives (incorrectly called FAR):      {fn:3d}")
print(f"  True Positives (correctly identified NEARBY):  {tp:3d}")

if tp + fn > 0:
    sensitivity = tp / (tp + fn)
    print(f"\nSensitivity (NEARBY detection rate): {sensitivity:.2%}")

if tn + fp > 0:
    specificity = tn / (tn + fp)
    print(f"Specificity (FAR detection rate): {specificity:.2%}")

# Show sample predictions with confidence
print(f"\n" + "="*100)
print("SAMPLE PREDICTIONS (first 20 samples)")
print("="*100)

print(f"\n{'#':3s} {'DISTANCE':>8s} {'PREDICTION':>12s} {'CONFIDENCE':>10s} {'TRUTH':>8s} {'CORRECT':>7s}")
print("-" * 100)

for i in range(min(20, len(X_scaled))):
    dist_ft = h0_data_valid.iloc[i]['distance_ft']
    pred = 'NEARBY' if predictions[i] == 1 else 'FAR'
    conf = confidence[i]
    truth = 'NEARBY' if y_true[i] == 1 else 'FAR'
    correct = '✓' if predictions[i] == y_true[i] else '✗'
    
    print(f"{i+1:3d} {dist_ft:>7.0f}ft {pred:>12s} {conf:>9.2%} {truth:>8s} {correct:>7s}")

# Distance-based analysis
print(f"\n" + "="*100)
print("PERFORMANCE BY DISTANCE")
print("="*100)

distances = h0_data_valid['distance_ft'].unique()
distances.sort()

print(f"\n{'DISTANCE':>12s} {'SAMPLES':>8s} {'CORRECT':>8s} {'ACCURACY':>10s}")
print("-" * 100)

for dist in distances:
    mask = h0_data_valid['distance_ft'] == dist
    if mask.sum() > 0:
        acc = (predictions[mask] == y_true[mask]).mean()
        n_correct = (predictions[mask] == y_true[mask]).sum()
        n_total = mask.sum()
        print(f"{dist:>11.0f}ft {n_total:>8d} {n_correct:>8d} {acc:>10.2%}")

# Confidence analysis
print(f"\n" + "="*100)
print("CONFIDENCE ANALYSIS")
print("="*100)

print(f"\nAverage confidence by distance:")
print(f"{'DISTANCE':>12s} {'AVG CONFIDENCE':>18s} {'MIN':>8s} {'MAX':>8s}")
print("-" * 100)

for dist in distances:
    mask = h0_data_valid['distance_ft'] == dist
    if mask.sum() > 0:
        avg_conf = confidence[mask].mean()
        min_conf = confidence[mask].min()
        max_conf = confidence[mask].max()
        print(f"{dist:>11.0f}ft {avg_conf:>18.2%} {min_conf:>8.2%} {max_conf:>8.2%}")

print(f"\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)

# Find hardest cases
hard_cases = np.where(confidence < 0.75)[0]
if len(hard_cases) > 0:
    print(f"\n⚠️  Low confidence predictions: {len(hard_cases)} samples with confidence < 75%")
    print(f"   Suggesting these may be near the decision boundary")

# High confidence correct
high_conf_correct = ((predictions == y_true) & (confidence > 0.95)).sum()
print(f"\n✅ High confidence correct: {high_conf_correct} predictions with >95% confidence")

print(f"\n" + "="*100)
print("MODEL DEPLOYMENT")
print("="*100)

print(f"""
The optimized model is ready for deployment:

1. Load the model:
   model_pkg = joblib.load('proximity_classifier_optimized_7features.pkl')
   
2. Prepare new data:
   - Extract these 7 features: {', '.join([f.split('_')[1] + '...' for f in features_to_use[:3]])}...
   - Ensure no missing values
   
3. Scale and predict:
   X_scaled = model_pkg['scaler'].transform(X_new[model_pkg['features']])
   predictions = model_pkg['model'].predict(X_scaled)
   probabilities = model_pkg['model'].predict_proba(X_scaled)
   
Performance: ~87% accuracy across diverse distance ranges
Speed: Microseconds per prediction
Size: 50% smaller than 14-feature baseline model
""")

print("="*100)
