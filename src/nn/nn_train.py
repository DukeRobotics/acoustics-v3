"""

"""


import argparse
import os
import random
import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


def set_seed(seed: int = 13):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class HydrophoneDataset(Dataset):
    """
    PyTorch Dataset for hydrophone **time-difference** features relative to a
    reference hydrophone (hydrophone_0), which is fixed at 0.0 and intentionally
    excluded from the CSV.

    Expected CSV columns (header required):
        Envelope H1, Envelope H2, Envelope H3, Truth

    Notes
    -----
    - Features are raw Δt values (e.g., seconds) like: t_i - t_0, with t_0 ≡ 0.0.
    - By default, **no normalization** is applied to preserve physical meaning.
    - Optional per-channel calibration offsets can be supplied (e.g., to remove
      constant delays from wire lengths): X := X - offsets.
    """

    def __init__(
        self,
        csv_path: str,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize the dataset from a CSV file.

        Parameters
        ----------
        csv_path : str
            Path to CSV with columns: Envelope H1, Envelope H2, Envelope H3, Truth.
        dtype : torch.dtype
            Torch dtype for features (default: torch.float32).

        Attributes
        ----------
        X : torch.Tensor
            Feature tensor of shape (N, 3) with raw (or calibrated) Δt values.
        y : torch.Tensor
            Long tensor of shape (N,) with class labels.
        stats : dict
            Basic per-channel stats (min/max/median) for diagnostics.
        """

        # --- Read CSV ---
        with open(csv_path, "r", encoding="utf-8") as f:
            rows = [r for r in csv.reader(f) if any(cell.strip() for cell in r)]
        if not rows:
            raise ValueError("CSV is empty.")
        header = [h.strip() for h in rows[0]]

        try:
            idx1 = header.index("Envelope H1")
            idx2 = header.index("Envelope H2")
            idx3 = header.index("Envelope H3")
            idxy = header.index("Truth")
        except ValueError as e:
            raise ValueError(
                "CSV must include header columns: Envelope H1, Envelope H2, Envelope H3, Truth"
            ) from e

        feats, labels = [], []
        for r in rows[1:]:
            if not r or all(c.strip() == "" for c in r):
                continue

            feats.append([float(r[idx2])])
            # feats.append([float(r[idx2]), float(r[idx3])])
            # feats.append([float(r[idx1]), float(r[idx2]), float(r[idx3])])
            labels.append(int(float(r[idxy])))

        X = torch.tensor(feats, dtype=dtype)
        y = torch.tensor(labels, dtype=torch.long)

        # --- Store stats for debugging (no normalization applied) ---
        self.stats = {
            "min": X.min(dim=0).values,
            "max": X.max(dim=0).values,
            "median": X.median(dim=0).values,
        }

        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Returns
        -------
        int
            Number of samples (rows) in the dataset.
        """
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        """
        Retrieve a single (features, label) pair.

        Parameters
        ----------
        idx : int
            Row index.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            - features: shape (3,), raw (or calibrated) Δt values
            - label: scalar integer class id
        """
        return self.X[idx], self.y[idx]


# Model
class MLPProb(nn.Module):
    """
    NN for hydrophone classification tasks that outputs probabilities for each 
    class using a Softmax activation at the output layer.

    This model is designed for input features derived from hydrophone time
    differences (e.g., Δt₁₀, Δt₂₀, Δt₃₀), where each feature represents the
    arrival time difference between a hydrophone and a fixed reference sensor.
    The network learns to map these temporal differences to a categorical class
    (e.g., source direction or event type).

    Architecture:
        Input Layer:      `in_dim` (default 3)
        Hidden Layer 1:   64 neurons, ReLU activation, Dropout(p_drop)
        Hidden Layer 2:   128 neurons, ReLU activation, Dropout(p_drop)
        Hidden Layer 3:   64 neurons, ReLU activation, Dropout(p_drop)
        Output Layer:     `num_classes` neurons, Softmax activation (probabilities)

    Parameters
    ----------
    in_dim : int, optional
        Number of input features (default: 3). Typically the number of hydrophones minus one.
    num_classes : int, optional
        Number of output classes (default: 4).
    p_drop : float, optional
        Dropout probability between layers (default: 0.2).

    Attributes
    ----------
    model : nn.Sequential
        A sequential container implementing the layer stack.
    """
    # def __init__(self, in_dim=1, num_classes=4, p_drop=0.2):
    # def __init__(self, in_dim=2, num_classes=4, p_drop=0.2):
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
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_dim).

        Returns
        -------
        torch.Tensor
            Tensor of shape (batch_size, num_classes) containing class probabilities
            that sum to 1 along `dim=1`.
        """
        return self.model(x)


def accuracy(probs, y):
    """Calculate accuracy of model"""
    preds = probs.argmax(dim=1)
    return (preds == y).float().mean().item()


def accuracy_with_pass(probs, y, confidence_threshold=0.7):
    """Calculate accuracy only for confident predictions"""
    max_prob, preds = torch.max(probs, dim=1)
    confident_mask = max_prob >= confidence_threshold
    
    if confident_mask.sum() == 0:
        return 0.0, 0.0  # No confident predictions
    
    # Only count confident predictions
    confident_correct = (preds[confident_mask] == y[confident_mask]).sum().float()
    confident_total = confident_mask.sum().float()
    
    accuracy = confident_correct / confident_total
    coverage = confident_total / len(y)  # What fraction we actually predicted
    
    return accuracy.item(), coverage.item()


def main():
    default_path = "Scripts/Test_set/0_2.csv"
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=False, help="Path to hydrophone CSV file", default=default_path)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--save_dir", type=str, default="artifacts")
    args = parser.parse_args()

    set_seed(13)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    ds = HydrophoneDataset(args.csv)
    n = len(ds)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)

    # Build model
    model = MLPProb(in_dim=1, num_classes=4, p_drop=args.dropout).to(device)
    # model = MLPProb(in_dim=2, num_classes=4, p_drop=args.dropout).to(device)
    # model = MLPProb(in_dim=3, num_classes=4, p_drop=args.dropout).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.NLLLoss()  # using log(probs)

    # Training loop
    best_val_loss = float("inf")
    best_acc = 0
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss, total_acc = 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()

            probs = model(X)
            log_probs = torch.log(probs + 1e-8)
            loss = loss_fn(log_probs, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)
            total_acc += accuracy(probs, y) * X.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        train_acc = total_acc / len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss, val_acc = 0, 0
        val_conf_acc_50, val_coverage_50 = 0, 0
        val_conf_acc_80, val_coverage_80 = 0, 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                probs = model(X)
                log_probs = torch.log(probs + 1e-8)
                loss = loss_fn(log_probs, y)
                val_loss += loss.item() * X.size(0)
                val_acc += accuracy(probs, y) * X.size(0)
                
                # Confidence-aware metrics
                acc_50, cov_50 = accuracy_with_pass(probs, y, confidence_threshold=0.5)
                acc_80, cov_80 = accuracy_with_pass(probs, y, confidence_threshold=0.8)
                val_conf_acc_50 += acc_50 * X.size(0)
                val_coverage_50 += cov_50 * X.size(0)
                val_conf_acc_80 += acc_80 * X.size(0)
                val_coverage_80 += cov_80 * X.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc /= len(val_loader.dataset)
        val_conf_acc_50 /= len(val_loader.dataset)
        val_coverage_50 /= len(val_loader.dataset)
        val_conf_acc_80 /= len(val_loader.dataset)
        val_coverage_80 /= len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = val_acc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        print(f"Epoch {epoch+1:03d}: "
              f"train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f} | "
              f"50%: acc={val_conf_acc_50:.3f} cov={val_coverage_50:.3f} | "
              f"80%: acc={val_conf_acc_80:.3f} cov={val_coverage_80:.3f}")

    # Save model + normalization stats
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "hydrophone_model.pt")

    # Calculate final confidence metrics for best model
    model.load_state_dict(best_state)
    model.eval()
    final_conf_acc_50, final_coverage_50 = 0, 0
    final_conf_acc_80, final_coverage_80 = 0, 0
    
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            probs = model(X)
            acc_50, cov_50 = accuracy_with_pass(probs, y, confidence_threshold=0.5)
            acc_80, cov_80 = accuracy_with_pass(probs, y, confidence_threshold=0.8)
            final_conf_acc_50 += acc_50 * X.size(0)
            final_coverage_50 += cov_50 * X.size(0)
            final_conf_acc_80 += acc_80 * X.size(0)
            final_coverage_80 += cov_80 * X.size(0)
    
    final_conf_acc_50 /= len(val_loader.dataset)
    final_coverage_50 /= len(val_loader.dataset)
    final_conf_acc_80 /= len(val_loader.dataset)
    final_coverage_80 /= len(val_loader.dataset)

    print(f"BEST: val_loss={best_val_loss:.4f} acc={best_acc:.3f} | "
          f"50%: acc={final_conf_acc_50:.3f} cov={final_coverage_50:.3f} | "
          f"80%: acc={final_conf_acc_80:.3f} cov={final_coverage_80:.3f}")

    torch.save({
        "model_state_dict": best_state,
    }, save_path)

    print(f"\n✅ Model saved to: {save_path}")


if __name__ == "__main__":
    main()