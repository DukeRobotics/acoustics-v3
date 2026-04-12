"""Check model accuracy against ground truth (distance-based nearby definition)."""
import pandas as pd
import sys
from pathlib import Path


def check_accuracy(csv_path):
    """Load CSV and calculate accuracy.
    
    Args:
        csv_path: Path to analysis CSV file
    """
    df = pd.read_csv(csv_path)
    
    # Parse distance
    def parse_distance(dist_str):
        try:
            return int(str(dist_str).replace('FT', ''))
        except:
            return None
    
    df['distance_ft'] = df['DISTANCE'].apply(parse_distance)
    
    # Ground truth: nearby if <= 10ft
    df['ground_truth'] = df['distance_ft'] <= 10
    
    # Model prediction
    df['model_prediction'] = df['IS_NEARBY']
    
    # Calculate accuracy
    valid_rows = df['distance_ft'].notna()
    df_valid = df[valid_rows]
    
    if len(df_valid) == 0:
        print("No valid rows found")
        return
    
    accuracy = (df_valid['ground_truth'] == df_valid['model_prediction']).mean()
    correct = (df_valid['ground_truth'] == df_valid['model_prediction']).sum()
    total = len(df_valid)
    
    # Confusion matrix
    tp = ((df_valid['ground_truth'] == True) & (df_valid['model_prediction'] == True)).sum()
    tn = ((df_valid['ground_truth'] == False) & (df_valid['model_prediction'] == False)).sum()
    fp = ((df_valid['ground_truth'] == False) & (df_valid['model_prediction'] == True)).sum()
    fn = ((df_valid['ground_truth'] == True) & (df_valid['model_prediction'] == False)).sum()
    
    print(f"\n{'='*60}")
    print(f"Accuracy Check: {Path(csv_path).name}")
    print(f"{'='*60}")
    print(f"\nDataset: {total} samples")
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives (nearby, predicted nearby):    {tp:3d}")
    print(f"  True Negatives (far, predicted far):          {tn:3d}")
    print(f"  False Positives (far, predicted nearby):      {fp:3d}")
    print(f"  False Negatives (nearby, predicted far):      {fn:3d}")
    
    # Per-distance accuracy
    print(f"\nAccuracy by Distance:")
    for dist in sorted(df_valid['distance_ft'].unique()):
        mask = df_valid['distance_ft'] == dist
        acc = (df_valid[mask]['ground_truth'] == df_valid[mask]['model_prediction']).mean()
        count = mask.sum()
        label = "nearby" if dist <= 10 else "far"
        print(f"  {dist:3.0f}ft ({label:6s}): {acc:.2%} ({count} samples)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_accuracy.py <csv_file>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    if not Path(csv_path).exists():
        print(f"File not found: {csv_path}")
        sys.exit(1)
    
    check_accuracy(csv_path)
