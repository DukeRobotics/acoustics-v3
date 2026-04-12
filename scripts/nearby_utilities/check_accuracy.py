"""Check model accuracy against distance-based ground truth."""
import pandas as pd
import sys
from pathlib import Path


def check_accuracy(csv_path):
    df = pd.read_csv(csv_path)
    
    # Parse distance and set ground truth
    df['dist_ft'] = pd.to_numeric(df['DISTANCE'].str.replace('FT', ''), errors='coerce')
    df['truth'] = df['dist_ft'] <= 10
    
    # Filter: valid distance and ALL_VALID == True
    valid = (df['dist_ft'].notna()) & (df['ALL_VALID'] == True)
    df = df[valid]
    
    if len(df) == 0:
        print("No valid rows found")
        return
    
    # Accuracy
    matches = df['truth'] == df['IS_NEARBY']
    print(f"\n{'='*60}")
    print(f"Accuracy: {matches.mean():.2%} ({matches.sum()}/{len(df)})")
    print(f"{'='*60}")
    
    # Confusion matrix
    print(f"\nConfusion Matrix:")
    tp = ((df['truth']) & (df['IS_NEARBY'])).sum()
    tn = ((~df['truth']) & (~df['IS_NEARBY'])).sum()
    fp = ((~df['truth']) & (df['IS_NEARBY'])).sum()
    fn = ((df['truth']) & (~df['IS_NEARBY'])).sum()
    print(f"  TP: {tp:3d}  TN: {tn:3d}  FP: {fp:3d}  FN: {fn:3d}")
    
    # Per-distance accuracy
    print(f"\nBy Distance:")
    for dist in sorted(df['dist_ft'].unique()):
        m = df['dist_ft'] == dist
        acc = (df[m]['truth'] == df[m]['IS_NEARBY']).mean()
        label = "nearby" if dist <= 10 else "far"
        print(f"  {dist:.0f}ft ({label}): {acc:.2%} ({m.sum()} samples)")


if __name__ == "__main__":
    path = "analysis/analysis_2026-04-11--13-31-12.csv"
    check_accuracy(path)
