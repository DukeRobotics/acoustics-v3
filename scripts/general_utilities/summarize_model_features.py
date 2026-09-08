"""Summarize nearby and far model features from an analysis CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_PATH = REPO_ROOT / "analysis/analysis_2026-04-12--14-13-52.csv"
DEFAULT_TABLE_PATH = REPO_ROOT / "analysis/feature_group_summary.csv"
DEFAULT_PLOT_PATH = REPO_ROOT / "analysis/feature_group_scatter.png"
DEFAULT_DISTRIBUTION_PATH = REPO_ROOT / "analysis/feature_group_distributions.png"
DEFAULT_BOUNDARY_PATH = REPO_ROOT / "analysis/model_decision_regions.png"
DEFAULT_MODEL_PATH = (
    REPO_ROOT / "scripts/artifacts/proximity_classifier_10ft_threshold_2026-04-12--23-04-00.pkl"
)

FEATURE_COLUMNS = {
    "Raw spectral flatness": "H0 RAW_spectral_flatness",
    "Filtered spectral centroid (Hz)": "H0 FILTERED_spectral_centroid_hz",
    "Raw rise time (ms)": "H0 RAW_rise_time_ms",
}
FEATURE_LABELS = list(FEATURE_COLUMNS.keys())
FEATURE_NAMES = list(FEATURE_COLUMNS.values())


def load_valid_features(csv_path: Path) -> pd.DataFrame:
    """Load valid rows and assign nearby/far ground-truth groups."""
    data = pd.read_csv(csv_path)
    data["distance_ft"] = pd.to_numeric(
        data["DISTANCE"].astype(str).str.replace("FT", "", regex=False),
        errors="coerce",
    )
    valid = data[(data["ALL_VALID"] == True) & data["distance_ft"].notna()].copy()
    valid["group"] = valid["distance_ft"].le(10).map({True: "Nearby", False: "Far"})

    for column in FEATURE_COLUMNS.values():
        valid[column] = pd.to_numeric(valid[column], errors="coerce")

    return valid.dropna(subset=list(FEATURE_COLUMNS.values()))


def build_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate count, mean, median, and standard deviation by group."""
    summary = data.groupby("group")[list(FEATURE_COLUMNS.values())].agg(
        ["count", "mean", "median", "std"]
    )
    summary.columns = [f"{feature} {stat}" for feature, stat in summary.columns]
    return summary.reset_index()


def plot_features(data: pd.DataFrame, output_path: Path) -> None:
    """Save feature-versus-distance scatter plots colored by class."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {"Nearby": "tab:blue", "Far": "tab:orange"}

    for axis, (label, column) in zip(axes, FEATURE_COLUMNS.items()):
        for group, group_data in data.groupby("group"):
            axis.scatter(
                group_data["distance_ft"],
                group_data[column],
                s=24,
                alpha=0.7,
                color=colors[group],
                label=group,
            )
        axis.set_title(label)
        axis.set_xlabel("Distance (ft)")
        axis.set_ylabel("Feature value")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    figure.suptitle("Model features by source distance", fontsize=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_distributions(data: pd.DataFrame, output_path: Path) -> None:
    """Save class distributions with boxplots and overlaid raw observations."""
    figure, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = {"Nearby": "tab:blue", "Far": "tab:orange"}
    rng = np.random.default_rng(42)

    for axis, label, column in zip(axes, FEATURE_LABELS, FEATURE_NAMES):
        groups = [
            data.loc[data["group"] == "Nearby", column],
            data.loc[data["group"] == "Far", column],
        ]
        axis.boxplot(groups, positions=[1, 2], widths=0.45, patch_artist=False)
        for position, group in enumerate(("Nearby", "Far"), start=1):
            values = data.loc[data["group"] == group, column].to_numpy()
            jitter = rng.uniform(-0.12, 0.12, len(values))
            axis.scatter(
                position + jitter,
                values,
                s=18,
                alpha=0.55,
                color=colors[group],
                label=group,
            )
        axis.set_title(label)
        axis.set_xticks([1, 2], ["Nearby", "Far"])
        axis.set_ylabel("Feature value")
        axis.grid(True, axis="y", alpha=0.3)
        axis.legend(loc="best")

    figure.suptitle("Nearby versus far feature distributions", fontsize=15)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_decision_regions(
    data: pd.DataFrame,
    model_path: Path,
    output_path: Path,
) -> None:
    """Save pairwise model probability regions with the third feature fixed."""
    package = joblib.load(model_path)
    model = package["model"]
    model_features = package["features"]
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1

    feature_data = data[FEATURE_NAMES].rename(
        columns=dict(zip(FEATURE_NAMES, model_features))
    )
    medians = feature_data.median()
    pairs = [(0, 1), (0, 2), (1, 2)]
    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    colors = {"Nearby": "tab:blue", "Far": "tab:orange"}
    nearby_class = list(model.classes_).index(1)

    for axis, (x_index, y_index) in zip(axes, pairs):
        x_name = model_features[x_index]
        y_name = model_features[y_index]
        x_values = feature_data[x_name]
        y_values = feature_data[y_name]
        x_padding = (x_values.max() - x_values.min()) * 0.08
        y_padding = (y_values.max() - y_values.min()) * 0.08
        x_grid = np.linspace(x_values.min() - x_padding, x_values.max() + x_padding, 180)
        y_grid = np.linspace(y_values.min() - y_padding, y_values.max() + y_padding, 180)
        grid_x, grid_y = np.meshgrid(x_grid, y_grid)
        grid = pd.DataFrame({feature: medians[feature] for feature in model_features}, index=grid_x.ravel())
        grid[x_name] = grid_x.ravel()
        grid[y_name] = grid_y.ravel()
        nearby_probability = model.predict_proba(grid)[:, nearby_class].reshape(grid_x.shape)

        axis.contourf(
            grid_x,
            grid_y,
            nearby_probability,
            levels=np.linspace(0, 1, 11),
            cmap="RdYlBu",
            alpha=0.42,
        )
        axis.contour(
            grid_x,
            grid_y,
            nearby_probability,
            levels=[0.5],
            colors="black",
            linewidths=1.5,
        )
        for group, group_data in data.groupby("group"):
            axis.scatter(
                group_data[FEATURE_NAMES[x_index]],
                group_data[FEATURE_NAMES[y_index]],
                s=22,
                alpha=0.7,
                color=colors[group],
                label=group,
            )
        third_index = ({0, 1, 2} - {x_index, y_index}).pop()
        axis.set_title(
            f"{FEATURE_LABELS[x_index]} vs {FEATURE_LABELS[y_index]}\n"
            f"{FEATURE_LABELS[third_index]} fixed at median"
        )
        axis.set_xlabel(FEATURE_LABELS[x_index])
        axis.set_ylabel(FEATURE_LABELS[y_index])
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    figure.suptitle(
        "Random Forest decision regions\n"
        "Blue background = nearby probability; orange background = far probability",
        fontsize=15,
    )
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    """Parse input and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH)
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE_PATH)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--distributions", type=Path, default=DEFAULT_DISTRIBUTION_PATH)
    parser.add_argument("--boundaries", type=Path, default=DEFAULT_BOUNDARY_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    """Build and save the feature summary table and scatter plot."""
    args = parse_args()
    data = load_valid_features(args.csv)
    if not args.model.exists():
        raise FileNotFoundError(args.model)
    summary = build_summary(data)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.table, index=False, float_format="%.6f")
    plot_features(data, args.plot)
    plot_distributions(data, args.distributions)
    plot_decision_regions(data, args.model, args.boundaries)

    print(f"Rows used: {len(data)}")
    print(summary.to_string(index=False))
    print(f"Saved table to {args.table}")
    print(f"Saved plot to {args.plot}")
    print(f"Saved distributions to {args.distributions}")
    print(f"Saved decision regions to {args.boundaries}")


if __name__ == "__main__":
    main()
