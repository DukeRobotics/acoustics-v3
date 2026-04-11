#!/usr/bin/env python3
"""Quick test of parser on one directory only."""
import os
import sys
import csv

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from controller import analyze_one_sample

path = "data/2.22.2026/H0_Closest_0FT_2026-02-22--15-29-08"

print("Testing parser feature extraction with 1 directory...\n")

# Collect feature names from first sample
feature_names_set = set()
first_sample = None

# Just process first 3 epochs for testing
epoch_dirs = sorted([d.name for d in os.scandir(path) if d.is_dir()])[:3]

for folder in epoch_dirs:
    print(f"Processing {folder}...")
    is_nearby, is_valid, toa_results, nearby_results, feature_results = analyze_one_sample(
        os.path.join(path, folder)
    )
    
    # Collect feature names from first sample
    if first_sample is None and feature_results:
        first_sample = feature_results[0]
        for key in first_sample.keys():
            if key != 'hydrophone_idx':
                feature_names_set.add(key)
    
    print(f"  is_nearby={is_nearby}, is_valid={is_valid}")
    if feature_results:
        print(f"  feature_results: {len(feature_results)} hydros")
        for feat_dict in feature_results:
            h_idx = feat_dict.get('hydrophone_idx', -1)
            n_features = len([k for k in feat_dict.keys() if k != 'hydrophone_idx'])
            print(f"    H{h_idx}: {n_features} features")
    print()

print(f"\nTotal unique features found across all hydros: {len(feature_names_set)}")
print("\nSample feature names:")
for fname in sorted(feature_names_set)[:10]:
    print(f"  {fname}")

# Check expected column structure for CSV
print("\n\nExpected CSV structure:")
print("  Base columns: Sample, ping_time, latitude, longitude, is_nearby, IS_NEARBY, [TOA/NEARBY fields]")
print(f"  Feature columns per hydro: 52 (26 FILTERED_ + 26 RAW_) × 4 hydros = 208 total")
print(f"  Total expected columns: ~50 base + 208 feature = ~258 columns")
