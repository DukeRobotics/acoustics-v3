"""
AGGRESSIVE FEATURE OPTIMIZATION
Goal: Fewer features, higher accuracy
Methods: XGBoost, LightGBM, Boruta, Sequential Selection, Permutation Importance
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, SequentialFeatureSelector
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except:
    HAS_XGBOOST = False
    print("XGBoost not installed, skipping")

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except:
    HAS_LIGHTGBM = False
    print("LightGBM not installed, skipping")

try:
    from boruta import BorutaPy
    HAS_BORUTA = True
except:
    HAS_BORUTA = False
    print("Boruta not installed, skipping")

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

print("="*80)
print("AGGRESSIVE FEATURE OPTIMIZATION")
print("="*80)
print(f"Dataset: {len(X_scaled)} samples, {len(X_scaled.columns)} features\n")

results = {}

# ============================================================================
# METHOD 1: XGBoost with feature importance
# ============================================================================
if HAS_XGBOOST:
    print("METHOD 1: XGBoost Feature Importance")
    print("-" * 80)
    
    best_xgb_score = 0
    best_xgb_features = []
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                                   random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_scaled, y)
    
    feature_importance = pd.DataFrame({
        'feature': X_scaled.columns,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for n_feat in range(1, 26):
        top_features = feature_importance.head(n_feat)['feature'].tolist()
        X_sub = X_scaled[top_features]
        score = cross_val_score(xgb_model, X_sub, y, cv=cv, scoring='accuracy').mean()
        print(f"  {n_feat:2d} features: {score:.4f}")
        
        if score > best_xgb_score:
            best_xgb_score = score
            best_xgb_features = top_features
    
    results['XGBoost'] = {'score': best_xgb_score, 'n_features': len(best_xgb_features), 
                          'features': best_xgb_features}
    print(f"  Best: {best_xgb_score:.4f} with {len(best_xgb_features)} features\n")

# ============================================================================
# METHOD 2: LightGBM with feature importance
# ============================================================================
if HAS_LIGHTGBM:
    print("METHOD 2: LightGBM Feature Importance")
    print("-" * 80)
    
    best_lgb_score = 0
    best_lgb_features = []
    
    lgb_model = lgb.LGBMClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                                    random_state=42, verbose=-1)
    lgb_model.fit(X_scaled, y)
    
    feature_importance = pd.DataFrame({
        'feature': X_scaled.columns,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for n_feat in range(1, 26):
        top_features = feature_importance.head(n_feat)['feature'].tolist()
        X_sub = X_scaled[top_features]
        score = cross_val_score(lgb_model, X_sub, y, cv=cv, scoring='accuracy').mean()
        print(f"  {n_feat:2d} features: {score:.4f}")
        
        if score > best_lgb_score:
            best_lgb_score = score
            best_lgb_features = top_features
    
    results['LightGBM'] = {'score': best_lgb_score, 'n_features': len(best_lgb_features), 
                           'features': best_lgb_features}
    print(f"  Best: {best_lgb_score:.4f} with {len(best_lgb_features)} features\n")

# ============================================================================
# METHOD 3: SelectFromModel (threshold-based)
# ============================================================================
print("METHOD 3: SelectFromModel (Threshold-based)")
print("-" * 80)

best_sfm_score = 0
best_sfm_features = []
best_sfm_thresh = 0

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_scaled, y)

# Get feature importance and test different thresholds
feature_importance = pd.DataFrame({
    'feature': X_scaled.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

for n_feat in [5, 8, 10, 12, 15, 20, 25]:
    top_features = feature_importance.head(n_feat)['feature'].tolist()
    X_sub = X_scaled[top_features]
    score = cross_val_score(rf, X_sub, y, cv=cv, scoring='accuracy').mean()
    print(f"  Top {n_feat:2d} features: {score:.4f}")
    
    if score > best_sfm_score:
        best_sfm_score = score
        best_sfm_features = top_features
        best_sfm_thresh = n_feat

results['SelectFromModel'] = {'score': best_sfm_score, 'n_features': len(best_sfm_features), 
                              'features': best_sfm_features}
print(f"  Best: {best_sfm_score:.4f} with {len(best_sfm_features)} features\n")

# ============================================================================
# METHOD 4: Sequential Forward Selection
# ============================================================================
print("METHOD 4: Sequential Forward Selection (SFS)")
print("-" * 80)

rf_sfs = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
sfs = SequentialFeatureSelector(rf_sfs, n_features_to_select='auto', 
                                 direction='forward', cv=cv, scoring='accuracy', n_jobs=-1)
sfs.fit(X_scaled, y)

sfs_features = X_scaled.columns[sfs.get_support()].tolist()
X_sfs = X_scaled[sfs_features]
sfs_score = cross_val_score(rf_sfs, X_sfs, y, cv=cv, scoring='accuracy').mean()

print(f"  Selected {len(sfs_features)} features")
print(f"  Accuracy: {sfs_score:.4f}")
for i, feat in enumerate(sfs_features, 1):
    print(f"    {i}. {feat}")

results['SFS'] = {'score': sfs_score, 'n_features': len(sfs_features), 'features': sfs_features}
print()

# ============================================================================
# METHOD 5: Sequential Backward Selection
# ============================================================================
print("METHOD 5: Sequential Backward Selection (SBS)")
print("-" * 80)

sbs = SequentialFeatureSelector(rf_sfs, n_features_to_select='auto', 
                                 direction='backward', cv=cv, scoring='accuracy', n_jobs=-1)
sbs.fit(X_scaled, y)

sbs_features = X_scaled.columns[sbs.get_support()].tolist()
X_sbs = X_scaled[sbs_features]
sbs_score = cross_val_score(rf_sfs, X_sbs, y, cv=cv, scoring='accuracy').mean()

print(f"  Selected {len(sbs_features)} features")
print(f"  Accuracy: {sbs_score:.4f}")
for i, feat in enumerate(sbs_features, 1):
    print(f"    {i}. {feat}")

results['SBS'] = {'score': sbs_score, 'n_features': len(sbs_features), 'features': sbs_features}
print()

# ============================================================================
# METHOD 6: Boruta Algorithm (if available)
# ============================================================================
if HAS_BORUTA:
    print("METHOD 6: Boruta Algorithm (Feature Importance Wrapper)")
    print("-" * 80)
    
    rf_boruta = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
    boruta = BorutaPy(rf_boruta, n_estimators='auto', random_state=42)
    boruta.fit(X_scaled.values, y.values)
    
    boruta_features = X_scaled.columns[boruta.support_].tolist()
    X_boruta = X_scaled[boruta_features]
    boruta_score = cross_val_score(rf_boruta, X_boruta, y, cv=cv, scoring='accuracy').mean()
    
    print(f"  Selected {len(boruta_features)} features")
    print(f"  Accuracy: {boruta_score:.4f}")
    for i, feat in enumerate(boruta_features, 1):
        print(f"    {i}. {feat}")
    
    results['Boruta'] = {'score': boruta_score, 'n_features': len(boruta_features), 
                         'features': boruta_features}
    print()

# ============================================================================
# FINAL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("FINAL COMPARISON - SORTED BY ACCURACY")
print("="*80)

for method in sorted(results.items(), key=lambda x: x[1]['score'], reverse=True):
    name, data = method
    print(f"{name:20s} | Accuracy: {data['score']:.4f} | Features: {data['n_features']:2d}")

best_method = max(results.items(), key=lambda x: x[1]['score'])
print(f"\n🏆 BEST: {best_method[0]} - {best_method[1]['score']:.4f} accuracy with {best_method[1]['n_features']} features")

print(f"\nBest features:")
for i, feat in enumerate(best_method[1]['features'], 1):
    print(f"  {i}. {feat}")

# ============================================================================
# AGGRESSIVE TESTING: Try fewer features
# ============================================================================
print("\n" + "="*80)
print("ULTRA-AGGRESSIVE: Testing 1-10 features with best method")
print("="*80)

best_of_best = {'score': 0, 'n_features': 0, 'features': []}
best_model_type = None

for n_feat in range(1, 11):
    top_features = best_method[1]['features'][:n_feat]
    X_tiny = X_scaled[top_features]
    
    # Test multiple models
    rf_score = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                               X_tiny, y, cv=cv, scoring='accuracy').mean()
    gb_score = cross_val_score(GradientBoostingClassifier(n_estimators=100, random_state=42),
                               X_tiny, y, cv=cv, scoring='accuracy').mean()
    lr_score = cross_val_score(LogisticRegression(max_iter=1000, random_state=42),
                               X_tiny, y, cv=cv, scoring='accuracy').mean()
    
    best_score = max(rf_score, gb_score, lr_score)
    best_model = ['RF', 'GB', 'LR'][np.argmax([rf_score, gb_score, lr_score])]
    
    print(f"  {n_feat} features: RF={rf_score:.4f} | GB={gb_score:.4f} | LR={lr_score:.4f} | Best={best_score:.4f} ({best_model})")
    
    if best_score > best_of_best['score']:
        best_of_best = {'score': best_score, 'n_features': n_feat, 'features': top_features, 'model': best_model}

print(f"\n🎯 ULTRA-BEST: {best_of_best['score']:.4f} accuracy with {best_of_best['n_features']} features using {best_of_best['model']}")
print("Features:")
for i, feat in enumerate(best_of_best['features'], 1):
    print(f"  {i}. {feat}")

print("\n" + "="*80)
