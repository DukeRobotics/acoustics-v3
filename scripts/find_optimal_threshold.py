import csv

rows = []
total = skipped = 0

with open("analysis/analysis_2026-04-09--02-47-04.csv") as f:
    for row in csv.DictReader(f):
        total += 1
        if row['ALL_VALID'].lower() != 'true':
            skipped += 1
            continue
        try:
            distance = float(row['DISTANCE'].replace('FT', ''))
            ping_width = float(row['H0 PING_WIDTH'])
            rows.append((ping_width, distance <= 20))
        except:
            skipped += 1

valid = len(rows)
best_accuracy = -1
best_threshold = best_metrics = None

for threshold in sorted(set(pw[0] for pw in rows)):
    tp = tn = fp = fn = 0
    for pw, gt in rows:
        pred = pw <= threshold
        if gt and pred: tp += 1
        elif not gt and not pred: tn += 1
        elif not gt and pred: fp += 1
        else: fn += 1
    
    acc = (tp + tn) / valid
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_threshold = threshold
        best_metrics = (tp, tn, fp, fn, acc, prec, rec, f1)

tp, tn, fp, fn, acc, prec, rec, f1 = best_metrics
print(f"\nOptimal H0 PING_WIDTH threshold: {best_threshold}")
print(f"Total samples: {total} | Skipped: {total - valid} | Valid: {valid} | Distance Threshold: 20 ft")
print(f"TP: {tp} | TN: {tn} | FP: {fp} | FN: {fn}")
print(f"Accuracy: {acc:.2%} | Precision: {prec:.2%} | Recall: {rec:.2%} | F1: {f1:.4f}\n")
