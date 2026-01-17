
"""
Load a trained hydrophone model checkpoint and output class probabilities for new data.

Usage:
    python src/nn_predict.py --ckpt artifacts/hydrophone_model.pt --csv data/nn_sample_data.csv --out probs.csv

The input CSV must have columns:
    [label ,] Envelope H1, Envelope H2, Envelope H3

The output CSV will include per-class probabilities and the predicted class.
"""

import argparse
import csv
from typing import List, Tuple

import torch
from torch import nn


# Model must match training architecture
class MLPProb(nn.Module):
    """3 hidden layers, always outputs probabilities (Softmax)."""

    def __init__(self, in_dim=3, num_classes=4, p_drop=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(64, num_classes),
            nn.Softmax(dim=1)   # Always output probabilities
        )

    def forward(self, x):
        return self.model(x)


def read_csv_features(path: str) -> Tuple[torch.Tensor, List[str], List[List[str]]]:
    """
    Read CSV and return (X, header, rows). Keeps original rows for output passthrough.
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        raise ValueError("CSV is empty.")
    header = [h.strip() for h in rows[0]]

    try:
        idx1 = header.index("Envelope H1")
        idx2 = header.index("Envelope H2")
        idx3 = header.index("Envelope H3")
    except ValueError as e:
        raise ValueError("Input CSV must include columns: Envelope H1, Envelope H2, Envelope H3") from e

    Xlist = []
    for r in rows[1:]:
        # Skip blank lines quietly
        if not r or all((c.strip() == "" for c in r)):
            continue
        try:
            Xlist.append([float(r[idx1]), float(r[idx2]), float(r[idx3])])
        except Exception as e:
            raise ValueError(f"Failed to parse numeric features on row: {r}") from e

    if not Xlist:
        raise ValueError("No feature rows found in CSV.")
    X = torch.tensor(Xlist, dtype=torch.float32)
    return X, header, rows


def write_output_csv(out_path: str, header: List[str], rows: List[List[str]], probs: torch.Tensor, preds: torch.Tensor):
    """
    Write the original rows plus per-class probabilities and predicted class.
    If the input had a header, we extend it; otherwise we create one.
    """
    # Determine number of classes from probs shape
    num_classes = probs.shape[1]
    prob_headers = [f"prob_{c}" for c in range(num_classes)]
    out_header = header.copy()
    for ph in prob_headers:
        if ph not in out_header:
            out_header.append(ph)
    if "pred" not in out_header:
        out_header.append("pred")

    # Build output rows (skip header row from input)
    out_rows = [out_header]
    ri = 0
    for r in rows[1:]:
        if not r or all((c.strip() == "" for c in r)):
            continue
        extended = r.copy()
        # Append probabilities
        for c in range(num_classes):
            extended.append(f"{probs[ri, c].item():.6f}")
        # Append predicted class
        extended.append(str(int(preds[ri].item())))
        out_rows.append(extended)
        ri += 1

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(out_rows)


def infer_num_classes_from_state_dict(state_dict: dict) -> int:
    """
    Infer the number of output classes by finding the last Linear layer's weight shape.
    """
    # Find the last weight tensor in the sequential module (should be the classifier layer)
    linear_out_feats = None
    for k, v in state_dict.items():
        if k.endswith(".weight") and v.ndim == 2:
            linear_out_feats = v.shape[0]
    if linear_out_feats is None:
        # Fallback to 4 (the training script default)
        linear_out_feats = 4
    return int(linear_out_feats)


def main():
    parser = argparse.ArgumentParser(description="Hydrophone probability inference")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file (from training script)")
    parser.add_argument("--csv", required=True, help="Path to input CSV with Envelope H1..3")
    parser.add_argument("--out", default="predictions.csv", help="Path to output CSV")
    parser.add_argument("--batch", type=int, default=256, help="Batch size for inference")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt["model_state_dict"]

    # Build model
    num_classes = infer_num_classes_from_state_dict(state_dict)
    model = MLPProb(in_dim=3, num_classes=num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Read input
    X, header, rows = read_csv_features(args.csv)
    X = X.to(device)

    # Inference in batches
    all_probs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], args.batch):
            xb = X[i:i+args.batch]
            pb = model(xb)  # already probabilities
            all_probs.append(pb)
    probs = torch.cat(all_probs, dim=0)
    preds = probs.argmax(dim=1)

    # Save to CSV
    write_output_csv(args.out, header, rows, probs.cpu(), preds.cpu())
    print(f"✅ Wrote predictions to: {args.out}")

    # Also print a small preview
    preview_n = min(5, probs.shape[0])
    print("\nPreview (first {} rows):".format(preview_n))
    for i in range(preview_n):
        p = ", ".join(f"{probs[i, c].item():.4f}" for c in range(num_classes))
        print(f"row {i}: pred={int(preds[i].item())} | probs=[{p}]")


if __name__ == "__main__":
    main()