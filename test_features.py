"""Quick test of parser on 2 epochs only."""
import os
import sys

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from controller import analyze_one_sample

path = "data/2.22.2026/H0_Closest_0FT_2026-02-22--15-29-08"

print("Testing FeatureAnalyzer with 2 epochs...\n")

for i, folder in enumerate([d.name for d in os.scandir(path) if d.is_dir()][:2]):
    print(f"Processing {folder}...")
    try:
        is_nearby, is_valid, toa_results, nearby_results, feature_results = analyze_one_sample(
            os.path.join(path, folder)
        )
        
        print(f"  is_nearby={is_nearby}, is_valid={is_valid}")
        print(f"  toa_results: {len(toa_results)} hydros")
        print(f"  nearby_results: {len(nearby_results)} hydros")
        print(f"  feature_results: {len(feature_results)} hydros")
        
        if feature_results:
            first_features = feature_results[0]
            print(f"    H{first_features.get('hydrophone_idx')}: {len(first_features)-1} features")
            # Print all features
            feature_names = sorted([k for k in first_features.keys() if k != 'hydrophone_idx'])
            print(f"    All {len(feature_names)} features:")
            for fname in feature_names:
                fval = first_features[fname]
                # Check for invalid values
                if isinstance(fval, float):
                    import math
                    if math.isnan(fval) or math.isinf(fval):
                        print(f"      {fname}: {fval} ⚠️ INVALID")
                    else:
                        print(f"      {fname}: {fval:.4f}")
                else:
                    print(f"      {fname}: {fval}")
        print()
    except Exception as e:
        print(f"  ERROR: {e}\n")
        import traceback
        traceback.print_exc()
