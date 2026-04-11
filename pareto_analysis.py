"""
PARETO FRONTIER ANALYSIS
Find the best accuracy achievable for each feature count (1-30)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
h0_data['is_nearby'] = (h0_data['distance_ft'] <= 20).astype(int)

h0_feature_cols = [col for col in df.columns if col.startswith('H0_') and col not in 
                   ['H0_IS_NEARBY', 'H0_VALID', 'H0_REASON', 'H0_TOA']]
X = h0_data[h0_feature_cols].copy()
y = h0_data['is_nearby'].copy()

missing_pct = (X.isna().sum() / len(X)) * 100
high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
X = X.drop(columns=high_missing_cols)
valid_idx = X.notna().all(axis=1)
X = X[valid_idx]
y = y[valid_idx]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("="*100)
print("PARETO FRONTIER ANALYSIS: Accuracy vs Feature Count")
print("="*100)
print(f"Dataset: {len(X_scaled)} samples, {len(X_scaled.columns)} features\n")

# Get feature importance rankings from multiple models
print("Computing feature importance rankings...")
print("-" * 100)

# Method 1: Random Forest importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_scaled, y)
rf_importance = pd.DataFrame({
    'feature': X_scaled.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

# Method 2: Gradient Boosting importance
gb = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5, learning_rate=0.1)
gb.fit(X_scaled, y)
gb_importance = pd.DataFrame({
    'feature': X_scaled.columns,
    'importance': gb.feature_importances_
}).sort_values('importance', ascending=False)

# Method 3: Permutation importance (Random Forest)
from sklearn.inspection import permutation_importance
perm_result = permutation_importance(rf, X_scaled, y, n_repeats=10, random_state=42, n_jobs=-1)
perm_importance = pd.DataFrame({
    'feature': X_scaled.columns,
    'importance': perm_result.importances_mean
}).sort_values('importance', ascending=False)

print("Feature importance methods calculated\n")

# Test each method
results_by_method = {}

methods = {
    'Random Forest Importance': rf_importance['feature'].tolist(),
    'Gradient Boosting Importance': gb_importance['feature'].tolist(),
    'Permutation Importance': perm_importance['feature'].tolist(),
}

print("\nTesting feature count 1-30 for each method:")
print("-" * 100)

for method_name, feature_order in methods.items():
    print(f"\n{method_name}:")
    results = []
    
    for n_feat in range(1, 31):
        selected_features = feature_order[:n_feat]
        X_sub = X_scaled[selected_features]
        
        # Test with each model
        rf_score = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                                   X_sub, y, cv=cv, scoring='accuracy').mean()
        gb_score = cross_val_score(GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1),
                                   X_sub, y, cv=cv, scoring='accuracy').mean()
        lr_score = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                                   X_sub, y, cv=cv, scoring='accuracy').mean()
        
        best_score = max(rf_score, gb_score, lr_score)
        best_model = ['RF', 'GB', 'LR'][np.argmax([rf_score, gb_score, lr_score])]
        
        results.append({
            'n_features': n_feat,
            'rf_score': rf_score,
            'gb_score': gb_score,
            'lr_score': lr_score,
            'best_score': best_score,
            'best_model': best_model,
            'features': selected_features
        })
        
        if n_feat <= 20 or n_feat % 5 == 0:
            print(f"  {n_feat:2d} features: RF={rf_score:.4f} GB={gb_score:.4f} LR={lr_score:.4f} Best={best_score:.4f} ({best_model})")
    
    results_by_method[method_name] = results

# Find Pareto frontier
print("\n" + "="*100)
print("PARETO FRONTIER - Best accuracy achievable for each feature count")
print("="*100)

pareto = {}
for n_feat in range(1, 31):
    best_across_all = 0
    best_method = ''
    best_features = []
    
    for method_name, results in results_by_method.items():
        for r in results:
            if r['n_features'] == n_feat and r['best_score'] > best_across_all:
                best_across_all = r['best_score']
                best_method = method_name
                best_features = r['features']
    
    if best_across_all > 0:
        pareto[n_feat] = {
            'score': best_across_all,
            'method': best_method,
            'features': best_features
        }
        print(f"  {n_feat:2d} features: {best_across_all:.4f} ({best_method})")

# Find key thresholds
print("\n" + "="*100)
print("KEY INSIGHTS")
print("="*100)

# Best accuracy achievable
best_overall = max(pareto.items(), key=lambda x: x[1]['score'])
print(f"\n1. BEST OVERALL: {best_overall[1]['score']:.4f} accuracy with {best_overall[0]} features")
print(f"   Method: {best_overall[1]['method']}")

# Find feature counts that beat baseline (86.29%)
baseline = 0.8629
over_baseline = [(n, data) for n, data in pareto.items() if data['score'] > baseline]
if over_baseline:
    print(f"\n2. BEATING BASELINE (86.29%): {len(over_baseline)} configurations")
    for n, data in sorted(over_baseline):
        print(f"   {n:2d} features: {data['score']:.4f}")
    
    min_beating = min(over_baseline, key=lambda x: x[0])
    print(f"\n   Minimum features to beat baseline: {min_beating[0]} features at {min_beating[1]['score']:.4f}")
else:
    print(f"\n2. BEATING BASELINE (86.29%): None found")
    
    # Find closest
    closest = min(pareto.items(), key=lambda x: abs(x[1]['score'] - baseline))
    print(f"   Closest: {closest[0]} features at {closest[1]['score']:.4f} (only -{(baseline - closest[1]['score'])*100:.2f}% difference)")

# 85% threshold (only 1.29% below baseline)
threshold_85 = 0.85
over_85 = [(n, data) for n, data in pareto.items() if data['score'] > threshold_85]
if over_85:
    min_for_85 = min(over_85, key=lambda x: x[0])
    print(f"\n3. ACHIEVING ≥85% ACCURACY: Minimum {min_for_85[0]} features ({min_for_85[1]['score']:.4f})")

# 80% threshold (5.29% below baseline)
threshold_80 = 0.80
over_80 = [(n, data) for n, data in pareto.items() if data['score'] > threshold_80]
if over_80:
    min_for_80 = min(over_80, key=lambda x: x[0])
    print(f"\n4. ACHIEVING ≥80% ACCURACY: Minimum {min_for_80[0]} features ({min_for_80[1]['score']:.4f})")

print("\n" + "="*100)
