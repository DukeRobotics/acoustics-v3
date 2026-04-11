import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

# Read the CSV file
csv_path = r'c:\Users\Saagar\Desktop\Acoustics\acoustics-v3\analysis\analysis_2026-04-10--04-22-31.csv'
df = pd.read_csv(csv_path)

# Filter for H0 data only
h0_data = df[df['CLOSEST_HYDROPHONE'] == 'H0'].copy()

# Create binary target
def parse_distance(dist_str):
    try:
        return int(str(dist_str).replace('FT', ''))
    except:
        return np.nan

h0_data['distance_ft'] = h0_data['DISTANCE'].apply(parse_distance)
h0_data['is_nearby'] = (h0_data['distance_ft'] <= 20).astype(int)

# Get features
h0_feature_cols = [col for col in df.columns if col.startswith('H0_') and col not in ['H0_IS_NEARBY', 'H0_VALID', 'H0_REASON', 'H0_TOA']]
X = h0_data[h0_feature_cols].copy()
y = h0_data['is_nearby'].copy()

# Handle missing values
missing_pct = (X.isna().sum() / len(X)) * 100
high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
X = X.drop(columns=high_missing_cols)
valid_idx = X.notna().all(axis=1)
X = X[valid_idx]
y = y[valid_idx]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("Dataset prepared:")
print(f"Samples: {len(X_scaled)}, Features: {len(X_scaled.columns)}")
print(f"Class 0 (Far): {(y==0).sum()}, Class 1 (Nearby): {(y==1).sum()}")

# Feature importance analysis
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_scaled, y)

feature_importance = pd.DataFrame({
    'feature': X_scaled.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*80)
print("AGGRESSIVE FEATURE REDUCTION SEARCH")
print("="*80)

best_overall = {'accuracy': 0, 'n_features': 0, 'features': [], 'model': ''}

# Test increasingly smaller feature sets
for n_features in range(1, len(X_scaled.columns) + 1):
    top_features = feature_importance.head(n_features)['feature'].tolist()
    X_subset = X_scaled[top_features]
    
    # Test multiple models
    rf_score = cross_val_score(rf, X_subset, y, cv=cv, scoring='accuracy').mean()
    gb = GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=5)
    gb_score = cross_val_score(gb, X_subset, y, cv=cv, scoring='accuracy').mean()
    et = ExtraTreesClassifier(n_estimators=100, random_state=42, max_depth=10)
    et_score = cross_val_score(et, X_subset, y, cv=cv, scoring='accuracy').mean()
    
    best_model_score = max(rf_score, gb_score, et_score)
    best_model_name = ['RF', 'GB', 'ET'][[rf_score, gb_score, et_score].index(best_model_score)]
    
    if best_model_score > best_overall['accuracy']:
        best_overall = {
            'accuracy': best_model_score,
            'n_features': n_features,
            'features': top_features,
            'model': best_model_name
        }
    
    print(f"N={n_features:2d}: RF={rf_score:.4f} | GB={gb_score:.4f} | ET={et_score:.4f} | Best={best_model_score:.4f} ({best_model_name})", end='')
    if best_model_score > 0.85:
        print(" ✓✓✓", end='')
    print()

print("\n" + "="*80)
print("BEST MINIMAL SET FOUND")
print("="*80)
print(f"\nAccuracy: {best_overall['accuracy']:.4f}")
print(f"Number of features: {best_overall['n_features']}")
print(f"Model: {best_overall['model']}")
print(f"\nOptimal features:")
for i, feat in enumerate(best_overall['features'], 1):
    print(f"  {i}. {feat}")

# Test even smaller with manual curation based on correlation
print("\n" + "="*80)
print("MANUAL FEATURE SET TUNING (Top correlated features)")
print("="*80)

# Calculate correlations
correlations = []
for col in X_scaled.columns:
    corr = X_scaled[col].corr(y)
    correlations.append({'feature': col, 'correlation': abs(corr)})

corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)

# Test top k features from correlation
best_corr = {'accuracy': 0, 'n_features': 0, 'features': []}
for n_features in range(1, 21):
    top_corr_features = corr_df.head(n_features)['feature'].tolist()
    X_corr = X_scaled[top_corr_features]
    
    rf_score = cross_val_score(rf, X_corr, y, cv=cv, scoring='accuracy').mean()
    gb_score = cross_val_score(gb, X_corr, y, cv=cv, scoring='accuracy').mean()
    
    best_score = max(rf_score, gb_score)
    
    print(f"Top {n_features} features by correlation: {best_score:.4f}")
    
    if best_score > best_corr['accuracy']:
        best_corr = {
            'accuracy': best_score,
            'n_features': n_features,
            'features': top_corr_features
        }

print(f"\nBest correlation-based set: {best_corr['accuracy']:.4f} ({best_corr['n_features']} features)")
print("Features:")
for i, feat in enumerate(best_corr['features'], 1):
    print(f"  {i}. {feat}")

# Test combination of top importance and top corr
print("\n" + "="*80)
print("HYBRID APPROACH (Top Importance + Top Correlated)")
print("="*80)

top5_importance = set(feature_importance.head(5)['feature'].tolist())
top5_corr = set(corr_df.head(5)['feature'].tolist())
hybrid_features = list(top5_importance.union(top5_corr))

print(f"\nHybrid feature set ({len(hybrid_features)} features):")
hybrid_set_combined = list(set(hybrid_features))
for i, feat in enumerate(sorted(hybrid_set_combined), 1):
    print(f"  {i}. {feat}")

X_hybrid = X_scaled[hybrid_set_combined]
rf_score = cross_val_score(rf, X_hybrid, y, cv=cv, scoring='accuracy').mean()
gb_score = cross_val_score(gb, X_hybrid, y, cv=cv, scoring='accuracy').mean()
print(f"\nHybrid RF accuracy: {rf_score:.4f}")
print(f"Hybrid GB accuracy: {gb_score:.4f}")

# FINAL SUMMARY
print("\n" + "="*80)
print("FINAL RECOMMENDATIONS")
print("="*80)

baseline_ping_width = pd.Series(h0_data['H0 PING_WIDTH'].values)[valid_idx].values.reshape(-1, 1)
scaler_ping = StandardScaler()
X_ping = scaler_ping.fit_transform(baseline_ping_width)
ping_acc = cross_val_score(rf, X_ping, y, cv=cv, scoring='accuracy').mean()

print(f"\nBaseline (Ping Width only):     {ping_acc:.4f}")
print(f"All 52 features:                0.8360")
print(f"Optimal importance-based set:   {best_overall['accuracy']:.4f} ({best_overall['n_features']} features)")
print(f"Optimal correlation-based set:  {best_corr['accuracy']:.4f} ({best_corr['n_features']} features)")
print(f"Hybrid set:                     {max(rf_score, gb_score):.4f} ({len(hybrid_set_combined)} features)")

print(f"\n\nREC #1: Use {best_overall['n_features']} features for {best_overall['accuracy']:.4f} accuracy")
print(f"REC #2: Use {best_corr['n_features']} features for {best_corr['accuracy']:.4f} accuracy (simpler)")
print(f"REC #3: Try hybrid {len(hybrid_set_combined)} features for best generalization")
