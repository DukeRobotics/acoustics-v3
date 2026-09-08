"""Plot nearby and far hydrophone waveforms, spectra, and model features."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq

SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyzers import NearbyAnalyzer
from hydrophones.hydrophone_array import HydrophoneArray


DEFAULT_NEAR_PATH = (
    REPO_ROOT
    / "data/2.22.2026/H0_Closest_0FT_2026-02-22--15-29-08"
    / "H0_Closest_0FT_epoch_1"
)
DEFAULT_FAR_PATH = (
    REPO_ROOT
    / "data/2.22.2026/H0_Closest_40FT_2026-02-22--15-53-23"
    / "H0_Closest_40FT_epoch_0"
)
DEFAULT_MODEL_PATH = (
    SCRIPT_DIR / "artifacts/proximity_classifier_10ft_threshold_2026-04-12--23-04-00.pkl"
)
DEFAULT_OUTPUT_PATH = REPO_ROOT / "analysis/feature_comparison_0ft_vs_40ft.png"

SEARCH_BAND_MIN = 30000
SEARCH_BAND_MAX = 34000
FILTER_ORDER = 6
MAX_PLOT_POINTS = 100_000
RISE_WINDOW_START_SECONDS = -0.002
RISE_WINDOW_END_SECONDS = 0.018


def load_hydrophone(capture_path: Path):
    """Load H0 from a Logic 2 capture directory."""
    array = HydrophoneArray(sampling_freq=781250, selected=[True, False, False, False])
    array.load_from_path(str(capture_path))
    hydrophone = array.hydrophones[0]
    if hydrophone.signal is None or hydrophone.times is None:
        raise ValueError(f"No H0 signal found in {capture_path}")
    return hydrophone


def analyze_capture(hydrophone, analyzer: NearbyAnalyzer) -> dict:
    """Calculate the plotted waveforms, spectra, envelope, and model features."""
    signal = hydrophone.signal
    sampling_frequency = 1 / hydrophone.sampling_period
    filtered = analyzer.apply_bandpass(
        signal,
        sampling_frequency,
        band_min=SEARCH_BAND_MIN,
        band_max=SEARCH_BAND_MAX,
    )
    envelope = np.abs(hilbert(signal))
    peak_index = int(np.argmax(envelope))
    rise_threshold = envelope[peak_index] * 0.1
    rise_start_index = next(
        (index for index in range(peak_index, -1, -1) if envelope[index] < rise_threshold),
        peak_index,
    )

    frequencies = fftfreq(len(signal), 1 / sampling_frequency)
    positive = frequencies >= 0
    raw_magnitude = np.abs(fft(signal))
    filtered_magnitude = np.abs(fft(filtered))
    flatness = analyzer._spectral_flatness(signal)
    centroid = analyzer._spectral_centroid(filtered, sampling_frequency)
    rise_time_ms = analyzer._rise_time(envelope, sampling_frequency, peak_index)
    feature_values = {
        "RAW_spectral_flatness": flatness,
        "FILTERED_spectral_centroid_hz": centroid,
        "RAW_rise_time_ms": rise_time_ms,
    }
    model_input = pd.DataFrame(
        [[
            feature_values["RAW_spectral_flatness"],
            feature_values["FILTERED_spectral_centroid_hz"],
            feature_values["RAW_rise_time_ms"],
        ]],
        columns=analyzer.features,
    )
    prediction = bool(analyzer.model.predict(model_input)[0])
    confidence = float(np.max(analyzer.model.predict_proba(model_input)[0]))

    return {
        "times": hydrophone.times,
        "signal": signal,
        "filtered": filtered,
        "envelope": envelope,
        "frequencies": frequencies[positive],
        "raw_magnitude": raw_magnitude[positive],
        "filtered_magnitude": filtered_magnitude[positive],
        "peak_index": peak_index,
        "rise_start_index": rise_start_index,
        "flatness": flatness,
        "centroid": centroid,
        "rise_time_ms": rise_time_ms,
        "prediction": prediction,
        "confidence": confidence,
    }


def plot_signal(ax, times, signal, color, label):
    """Plot a signal with downsampling for responsive figures."""
    stride = max(1, int(np.ceil(len(times) / MAX_PLOT_POINTS)))
    ax.plot(times[::stride], signal[::stride], color=color, label=label)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_combined_waveform(ax, data: dict, case_name: str) -> None:
    """Plot raw and filtered waveforms together with the model features."""
    times = data["times"]
    plot_signal(ax, times, data["signal"], "gray", "Raw signal")
    plot_signal(ax, times, data["filtered"], "blue", "Filtered signal")
    prediction_label = "NEARBY" if data["prediction"] else "FAR"
    ax.set_title(
        f"{case_name} | Detection: {prediction_label} "
        f"({data['confidence']:.1%})"
    )
    ax.set_ylabel("Amplitude")
    ax.text(
        0.02,
        0.97,
        (
            "MODEL FEATURES\n"
            f"Raw flatness: {data['flatness']:.4f}\n"
            f"Filtered centroid: {data['centroid']:.1f} Hz\n"
            f"Raw rise time: {data['rise_time_ms']:.3f} ms"
        ),
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85},
    )


def plot_rise_time(ax, data: dict) -> None:
    """Plot the envelope around the rise interval on a relative time axis."""
    times = data["times"]
    rise_start = times[data["rise_start_index"]]
    relative_times = times - rise_start
    window = (
        (relative_times >= RISE_WINDOW_START_SECONDS)
        & (relative_times <= RISE_WINDOW_END_SECONDS)
    )
    stride = max(1, int(np.ceil(np.count_nonzero(window) / MAX_PLOT_POINTS)))
    ax.plot(
        relative_times[window][::stride] * 1000,
        data["envelope"][window][::stride],
        color="darkblue",
        label="Raw Hilbert envelope",
    )
    rise_time_ms = data["rise_time_ms"]
    ax.axvline(0, color="orange", linestyle="--", label="Rise start")
    ax.axvline(
        rise_time_ms,
        color="red",
        linestyle="--",
        label=f"Rise end / peak: {rise_time_ms:.3f} ms",
    )
    ax.axvspan(0, rise_time_ms, color="orange", alpha=0.12, label="Rise interval")
    ax.set_xlim(RISE_WINDOW_START_SECONDS * 1000, RISE_WINDOW_END_SECONDS * 1000)
    ax.set_ylabel("Envelope")
    ax.set_xlabel("Time from rise start (ms)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")


def plot_comparison(nearby_data: dict, far_data: dict, output_path: Path) -> None:
    """Render and save the nearby-versus-far comparison figure."""
    cases = [("Nearby: 0 ft", nearby_data), ("Far: 40 ft", far_data)]
    figure, axes = plt.subplots(3, 2, figsize=(16, 12))
    row_titles = [
        "Raw + filtered waveform",
        "Raw + filtered FFT magnitude",
        "Rise-time zoom",
    ]

    for column, (case_name, data) in enumerate(cases):
        plot_combined_waveform(axes[0, column], data, case_name)

        axes[1, column].plot(
            data["frequencies"],
            data["raw_magnitude"],
            color="gray",
            label="Raw FFT",
        )
        axes[1, column].plot(
            data["frequencies"],
            data["filtered_magnitude"],
            color="blue",
            label="Filtered FFT",
        )
        axes[1, column].set_ylabel("Magnitude")
        axes[1, column].set_xlim(0, 100000)
        axes[1, column].grid(True, alpha=0.3)
        axes[1, column].legend(loc="best")

        axes[1, column].axvline(
            SEARCH_BAND_MIN,
            color="red",
            linestyle="--",
            alpha=0.5,
            label="Filter range",
        )
        axes[1, column].axvline(SEARCH_BAND_MAX, color="red", linestyle="--", alpha=0.5)
        axes[1, column].axvline(
            data["centroid"],
            color="green",
            linestyle=":",
            linewidth=2,
            label=f"Centroid: {data['centroid']:.1f} Hz",
        )

        plot_rise_time(axes[2, column], data)

    for row, title in enumerate(row_titles):
        axes[row, 0].annotate(
            title,
            xy=(0, 0.5),
            xytext=(-axes[row, 0].yaxis.labelpad - 15, 0),
            xycoords=axes[row, 0].yaxis.label,
            textcoords="offset points",
            size="large",
            ha="right",
            va="center",
            rotation=90,
        )

    figure.suptitle("H0 feature comparison: 0 ft versus 40 ft", fontsize=16)
    figure.tight_layout(rect=(0.04, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    print(f"Saved comparison plot to {output_path}")
    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse optional capture, model, and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--near", type=Path, default=DEFAULT_NEAR_PATH)
    parser.add_argument("--far", type=Path, default=DEFAULT_FAR_PATH)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    """Load both captures, calculate features, and save the comparison plot."""
    args = parse_args()
    for path in (args.near, args.far, args.model):
        if not path.exists():
            raise FileNotFoundError(path)

    analyzer = NearbyAnalyzer(
        model_path=str(args.model),
        filter_order=FILTER_ORDER,
        search_band_min=SEARCH_BAND_MIN,
        search_band_max=SEARCH_BAND_MAX,
        plot_results_flag=False,
    )
    nearby_data = analyze_capture(load_hydrophone(args.near), analyzer)
    far_data = analyze_capture(load_hydrophone(args.far), analyzer)
    plot_comparison(nearby_data, far_data, args.output)


if __name__ == "__main__":
    main()
