import csv
import sys

csv_file = "analysis/analysis_2026-04-09--02-47-04.csv"
threshold = 10

tp = tn = fp = fn = valid = total = skipped = 0

with open(csv_file) as f:
    reader = csv.DictReader(f)
    for row in reader:
        total += 1
        if row['ALL_VALID'].lower() != 'true':
            skipped += 1
            continue
        
        valid += 1
        distance = float(row['DISTANCE'].replace('FT', ''))
        is_nearby = row['IS_NEARBY'].lower() == 'true'
        actually_nearby = distance <= threshold
        
        if actually_nearby and is_nearby:
            tp += 1
        elif not actually_nearby and not is_nearby:
            tn += 1
        elif not actually_nearby and is_nearby:
            fp += 1
        else:
            fn += 1

accuracy = (tp + tn) / valid if valid > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nTotal samples: {total} | Skipped: {skipped} | Valid: {valid} | Threshold: {threshold} ft")
print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
print(f"Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%} | F1: {f1:.4f}\n")
