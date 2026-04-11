import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import (
    mutual_info_classif, 
    RFE, 
    SelectKBest,
    f_classif
)
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Read the CSV file
csv_path = r'c:\Users\Saagar\Desktop\Acoustics\acoustics-v3\analysis\analysis_2026-04-10--04-22-31.csv'
df = pd.read_csv(csv_path)

print("Data loaded. Shape:", df.shape)
print("\nDataframe info:")
print(df.info())

# Filter for H0 data only
h0_data = df[df['CLOSEST_HYDROPHONE'] == 'H0'].copy()
print(f"\nH0 samples: {len(h0_data)}")

# Extract distance and convert to binary target (≤20ft = nearby)
def parse_distance(dist_str):
    """Parse distance string like '0FT', '20FT', etc."""
    if pd.isna(dist_str):
        return np.nan
    try:
        return int(str(dist_str).replace('FT', ''))
    except:
        return np.nan

h0_data['distance_ft'] = h0_data['DISTANCE'].apply(parse_distance)
h0_data['is_nearby'] = (h0_data['distance_ft'] <= 20).astype(int)

print(f"\nClass distribution:")
print(h0_data['is_nearby'].value_counts())
print(f"Nearby: {(h0_data['is_nearby'] == 1).sum()}, Far: {(h0_data['is_nearby'] == 0).sum()}")

# Get all H0 feature columns (but exclude H0_IS_NEARBY, H0_VALID, H0_REASON, H0_TOA which are metadata)
h0_feature_cols = [col for col in df.columns if col.startswith('H0_') and col not in ['H0_IS_NEARBY', 'H0_VALID', 'H0_REASON', 'H0_TOA']]
print(f"\nTotal H0 features: {len(h0_feature_cols)}")
print(f"Features: {h0_feature_cols[:10]}...") # Show first 10

# Extract features and target
X = h0_data[h0_feature_cols].copy()
y = h0_data['is_nearby'].copy()

# Add ping width as a separate feature check
if 'H0 PING_WIDTH' in h0_data.columns:
    print("\nH0 PING_WIDTH column exists")

# Handle missing values - drop rows and columns with too many NaNs
print(f"\nMissing values per feature:")
missing_pct = (X.isna().sum() / len(X)) * 100
print(missing_pct[missing_pct > 0])

# Drop columns with >50% missing values
high_missing_cols = missing_pct[missing_pct > 50].index.tolist()
print(f"\nDropping {len(high_missing_cols)} columns with >50% missing values")
X = X.drop(columns=high_missing_cols)

# Drop rows with any remaining NaN values
valid_idx = X.notna().all(axis=1)
X = X[valid_idx]
y = y[valid_idx]

print(f"Final shape - X: {X.shape}, y: {y.shape}")
print(f"Final class distribution: {y.value_counts().to_dict()}")

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("\n" + "="*80)
print("STEP 1: CORRELATION ANALYSIS")
print("="*80)

# Calculate correlations with target
correlations = []
for col in X_scaled.columns:
    corr, p_val = pearsonr(X_scaled[col], y)
    correlations.append({'feature': col, 'correlation': abs(corr), 'p_value': p_val})

corr_df = pd.DataFrame(correlations).sort_values('correlation', ascending=False)
print("\nTop 20 most correlated features with target:")
print(corr_df.head(20).to_string())

# Check for multicollinearity
print("\n" + "="*80)
print("STEP 2: FEATURE IMPORTANCE (Random Forest)")
print("="*80)

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_scaled, y)

feature_importance = pd.DataFrame({
    'feature': X_scaled.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 most important features:")
print(feature_importance.head(20).to_string())

# Baseline model accuracy with all features
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
baseline_score = cross_val_score(rf, X_scaled, y, cv=cv, scoring='accuracy').mean()
print(f"\nBaseline accuracy (all {len(X_scaled.columns)} features): {baseline_score:.4f}")

print("\n" + "="*80)
print("STEP 3: ITERATIVE FEATURE SELECTION")
print("="*80)

# Test different numbers of features
results = []

for n_features in range(1, min(16, len(X_scaled.columns) + 1)):
    # Select top n features by importance
    top_features = feature_importance.head(n_features)['feature'].tolist()
    X_subset = X_scaled[top_features]
    
    # Test with multiple classifiers
    rf_score = cross_val_score(rf, X_subset, y, cv=cv, scoring='accuracy').mean()
    
    gb = GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=5)
    gb_score = cross_val_score(gb, X_subset, y, cv=cv, scoring='accuracy').mean()
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_score = cross_val_score(lr, X_subset, y, cv=cv, scoring='accuracy').mean()
    
    best_score = max(rf_score, gb_score, lr_score)
    best_model = ['RF', 'GB', 'LR'][[rf_score, gb_score, lr_score].index(best_score)]
    
    results.append({
        'n_features': n_features,
        'features': top_features,
        'rf_accuracy': rf_score,
        'gb_accuracy': gb_score,
        'lr_accuracy': lr_score,
        'best_accuracy': best_score,
        'best_model': best_model
    })
    
    print(f"Features: {n_features:2d} | RF: {rf_score:.4f} | GB: {gb_score:.4f} | LR: {lr_score:.4f} | Best: {best_score:.4f} ({best_model})")

results_df = pd.DataFrame(results)
best_result = results_df.loc[results_df['best_accuracy'].idxmax()]

print(f"\n\nOPTIMAL RESULT:")
print(f"Best accuracy: {best_result['best_accuracy']:.4f} using {int(best_result['n_features'])} features with {best_result['best_model']}")
print(f"\nOptimal features:")
for i, feat in enumerate(best_result['features'], 1):
    print(f"  {i}. {feat}")

# Compare with single feature (ping width baseline)
print("\n" + "="*80)
print("STEP 4: BASELINE COMPARISON (PING_WIDTH ONLY)")
print("="*80)

# Ping width from original data, not from features - use correct column name
X_ping_raw = h0_data[['H0 PING_WIDTH']].copy()
X_ping_raw = X_ping_raw.iloc[valid_idx].reset_index(drop=True)

# Check if ping width has NaNs and handle them
if X_ping_raw.isna().any().any():
    print(f"Ping width has {X_ping_raw.isna().sum()[0]} NaN values")
    valid_ping = X_ping_raw.notna().all(axis=1)
    X_ping_subset = X_ping_raw[valid_ping]
    y_ping_subset = pd.Series(y)[valid_ping]
else:
    X_ping_subset = X_ping_raw
    y_ping_subset = y

# Scale ping width
scaler_ping = StandardScaler()
X_ping_scaled = scaler_ping.fit_transform(X_ping_subset)

ping_rf = cross_val_score(rf, X_ping_scaled, y_ping_subset, cv=cv, scoring='accuracy').mean()
print(f"Ping width only accuracy: {ping_rf:.4f}")
print(f"Improvement with optimal set: {best_result['best_accuracy'] - ping_rf:.4f}")

print("\n" + "="*80)
print("STEP 5: RECURSIVE FEATURE ELIMINATION (RFE)")
print("="*80)

# Try RFE to find feature subset
rfe_results = []
for n_features in range(1, min(16, len(X_scaled.columns) + 1)):
    rfe = RFE(RandomForestClassifier(n_estimators=50, random_state=42), 
              n_features_to_select=n_features)
    rfe.fit(X_scaled, y)
    
    selected_features = X_scaled.columns[rfe.support_].tolist()
    X_rfe = X_scaled[selected_features]
    
    rf_score = cross_val_score(rf, X_rfe, y, cv=cv, scoring='accuracy').mean()
    
    rfe_results.append({
        'n_features': n_features,
        'features': selected_features,
        'accuracy': rf_score
    })
    
    print(f"RFE Features: {n_features:2d} | Accuracy: {rf_score:.4f}")

rfe_results_df = pd.DataFrame(rfe_results)
best_rfe = rfe_results_df.loc[rfe_results_df['accuracy'].idxmax()]

print(f"\nBest RFE result: {best_rfe['accuracy']:.4f} with {int(best_rfe['n_features'])} features")
print(f"RFE features:")
for i, feat in enumerate(best_rfe['features'], 1):
    print(f"  {i}. {feat}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Ping Width Only:        {ping_rf:.4f}")
print(f"Top N Features Method:  {best_result['best_accuracy']:.4f} ({int(best_result['n_features'])} features)")
print(f"RFE Method:             {best_rfe['accuracy']:.4f} ({int(best_rfe['n_features'])} features)")
print(f"\nBest overall: {max(ping_rf, best_result['best_accuracy'], best_rfe['accuracy']):.4f}")
