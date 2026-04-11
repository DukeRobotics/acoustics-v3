import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import RFE, mutual_info_classif, SelectKBest, f_classif
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Read and prepare data
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

h0_feature_cols = [col for col in df.columns if col.startswith('H0_') and col not in ['H0_IS_NEARBY', 'H0_VALID', 'H0_REASON', 'H0_TOA']]
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

print("Dataset prepared:")
print(f"Samples: {len(X_scaled)}, Features: {len(X_scaled.columns)}")
print(f"Class distribution: {y.value_counts().to_dict()}\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ============================================================================
# METHOD 1: L1 REGULARIZATION (LASSO LOGISTIC REGRESSION)
# ============================================================================
print("="*80)
print("METHOD 1: L1 REGULARIZATION (LASSO LOGISTIC REGRESSION)")
print("="*80)

# Find optimal C parameter using cross-validation
Cs = np.logspace(-4, 4, 20)
best_c = None
best_score = 0
best_l1_features = []

for C in Cs:
    try:
        clf = LogisticRegression(penalty='l1', solver='liblinear', C=C, random_state=42, max_iter=1000)
        score = cross_val_score(clf, X_scaled, y, cv=cv, scoring='accuracy').mean()
        
        # Fit to get feature coefficients
        clf.fit(X_scaled, y)
        non_zero_features = np.where(clf.coef_[0] != 0)[0]
        
        if score > best_score:
            best_score = score
            best_c = C
            best_l1_features = X_scaled.columns[non_zero_features].tolist()
        
        print(f"C={C:.4f}: Accuracy={score:.4f}, Features selected={len(non_zero_features)}")
    except:
        pass

print(f"\nBest L1 result: C={best_c}, Accuracy={best_score:.4f}, Features={len(best_l1_features)}")
print("Selected features:")
for i, feat in enumerate(sorted(best_l1_features), 1):
    print(f"  {i}. {feat}")

# ============================================================================
# METHOD 2: LASSO FOR RANKING (CONTINUOUS TARGET)
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: LASSO FEATURE RANKING")
print("="*80)

lasso_cv = LassoCV(cv=5, random_state=42, max_iter=5000)
lasso_cv.fit(X_scaled, y)

lasso_importance = pd.DataFrame({
    'feature': X_scaled.columns,
    'coefficient': np.abs(lasso_cv.coef_)
}).sort_values('coefficient', ascending=False)

print(f"\nLasso Alpha: {lasso_cv.alpha_:.6f}")
print("Top 20 features by Lasso coefficient:")
print(lasso_importance.head(20)[['feature', 'coefficient']].to_string(index=False))

# Test Lasso-selected features
for n_feat in [5, 8, 10, 12, 14, 16]:
    lasso_features = lasso_importance.head(n_feat)['feature'].tolist()
    X_lasso = X_scaled[lasso_features]
    rf_score = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                               X_lasso, y, cv=cv, scoring='accuracy').mean()
    print(f"  {n_feat} Lasso features: RF accuracy = {rf_score:.4f}")

# ============================================================================
# METHOD 3: MUTUAL INFORMATION (Information-theoretic approach)
# ============================================================================
print("\n" + "="*80)
print("METHOD 3: MUTUAL INFORMATION")
print("="*80)

mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
mi_df = pd.DataFrame({
    'feature': X_scaled.columns,
    'mi_score': mi_scores
}).sort_values('mi_score', ascending=False)

print("\nTop 20 features by Mutual Information:")
print(mi_df.head(20)[['feature', 'mi_score']].to_string(index=False))

# Test MI-selected features
best_mi_score = 0
best_mi_features = []
for n_feat in range(1, 21):
    mi_features = mi_df.head(n_feat)['feature'].tolist()
    X_mi = X_scaled[mi_features]
    rf_score = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                               X_mi, y, cv=cv, scoring='accuracy').mean()
    print(f"  {n_feat} MI features: RF accuracy = {rf_score:.4f}")
    if rf_score > best_mi_score:
        best_mi_score = rf_score
        best_mi_features = mi_features

# ============================================================================
# METHOD 4: RFE WITH MULTIPLE MODELS
# ============================================================================
print("\n" + "="*80)
print("METHOD 4: RECURSIVE FEATURE ELIMINATION (RFE)")
print("="*80)

estimators = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
}

rfe_results = {}
for est_name, estimator in estimators.items():
    print(f"\nRFE with {est_name}:")
    best_rfe_score = 0
    best_rfe_features = []
    
    for n_features in [5, 7, 10, 12, 14, 16, 18]:
        rfe = RFE(estimator, n_features_to_select=n_features)
        rfe.fit(X_scaled, y)
        selected_features = X_scaled.columns[rfe.support_].tolist()
        
        X_rfe = X_scaled[selected_features]
        rf_score = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                                   X_rfe, y, cv=cv, scoring='accuracy').mean()
        
        print(f"  {n_features} features: {rf_score:.4f}")
        
        if rf_score > best_rfe_score:
            best_rfe_score = rf_score
            best_rfe_features = selected_features
    
    rfe_results[est_name] = {'score': best_rfe_score, 'features': best_rfe_features}

# ============================================================================
# METHOD 5: F-STATISTIC SELECTION (ANOVA)
# ============================================================================
print("\n" + "="*80)
print("METHOD 5: F-STATISTIC (ANOVA) FEATURE SELECTION")
print("="*80)

f_scores, _ = f_classif(X_scaled, y)
f_df = pd.DataFrame({
    'feature': X_scaled.columns,
    'f_score': f_scores
}).sort_values('f_score', ascending=False)

print("\nTop 20 features by F-statistic:")
print(f_df.head(20)[['feature', 'f_score']].to_string(index=False))

best_f_score = 0
best_f_features = []
for n_feat in range(1, 21):
    f_features = f_df.head(n_feat)['feature'].tolist()
    X_f = X_scaled[f_features]
    rf_score = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                               X_f, y, cv=cv, scoring='accuracy').mean()
    print(f"  {n_feat} F-stat features: RF accuracy = {rf_score:.4f}")
    if rf_score > best_f_score:
        best_f_score = rf_score
        best_f_features = f_features

# ============================================================================
# COMPREHENSIVE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE COMPARISON OF ALL METHODS")
print("="*80)

summary = {
    'L1 Regularization': {
        'accuracy': best_score,
        'n_features': len(best_l1_features),
        'features': best_l1_features
    },
    'Lasso': {
        'accuracy': cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                                    X_scaled[lasso_importance.head(14)['feature'].tolist()],
                                    y, cv=cv, scoring='accuracy').mean(),
        'n_features': 14,
        'features': lasso_importance.head(14)['feature'].tolist()
    },
    'Mutual Information': {
        'accuracy': best_mi_score,
        'n_features': len(best_mi_features),
        'features': best_mi_features
    },
    'RFE (LogisticRegression)': {
        'accuracy': rfe_results['LogisticRegression']['score'],
        'n_features': len(rfe_results['LogisticRegression']['features']),
        'features': rfe_results['LogisticRegression']['features']
    },
    'RFE (RandomForest)': {
        'accuracy': rfe_results['RandomForest']['score'],
        'n_features': len(rfe_results['RandomForest']['features']),
        'features': rfe_results['RandomForest']['features']
    },
    'F-Statistic': {
        'accuracy': best_f_score,
        'n_features': len(best_f_features),
        'features': best_f_features
    }
}

print("\nMethod Comparison:")
print("-" * 80)
for method, results in sorted(summary.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    print(f"{method:30s} | Accuracy: {results['accuracy']:.4f} | Features: {results['n_features']:2d}")

best_method = max(summary.items(), key=lambda x: x[1]['accuracy'])
print("\n" + "="*80)
print(f"BEST METHOD: {best_method[0]}")
print("="*80)
print(f"Accuracy: {best_method[1]['accuracy']:.4f}")
print(f"Number of features: {best_method[1]['n_features']}")
print("\nFeatures:")
for i, feat in enumerate(sorted(best_method[1]['features']), 1):
    print(f"  {i}. {feat}")

# ============================================================================
# FINAL TESTING WITH DIFFERENT MODELS
# ============================================================================
print("\n" + "="*80)
print("FINAL VALIDATION WITH MULTIPLE MODELS")
print("="*80)

best_features = best_method[1]['features']
X_best = X_scaled[best_features]

models = {
    'Logistic Regression (L1)': LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=42, max_iter=1000),
    'Logistic Regression (L2)': LogisticRegression(penalty='l2', solver='lbfgs', C=1.0, random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
}

print(f"\nUsing best {len(best_features)} features:")
for model_name, model in models.items():
    score = cross_val_score(model, X_best, y, cv=cv, scoring='accuracy').mean()
    print(f"  {model_name:30s}: {score:.4f}")

# Baseline comparison
print("\nBaseline accuracies (all 52 features):")
for model_name, model in models.items():
    score = cross_val_score(model, X_scaled, y, cv=cv, scoring='accuracy').mean()
    print(f"  {model_name:30s}: {score:.4f}")

ping_width_data = h0_data[['H0 PING_WIDTH']].iloc[valid_idx].values
scaler_ping = StandardScaler()
ping_scaled = scaler_ping.fit_transform(ping_width_data)
ping_score = cross_val_score(RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
                              ping_scaled, y, cv=cv, scoring='accuracy').mean()
print(f"\nPing Width Only baseline: {ping_score:.4f}")
