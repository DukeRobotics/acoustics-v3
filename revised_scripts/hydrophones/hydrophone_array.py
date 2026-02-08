"""Hydrophone array module for multi-sensor data processing."""
from __future__ import annotations

import os
import struct

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.fft import fft, fftfreq

from hydrophones import hydrophone as hydrophone_module


class HydrophoneArray:
    """Array of hydrophone sensors with signal processing capabilities."""
    def __init__(
        self,
        search_band_min: float = 25000,
        search_band_max: float = 40000,
        sampling_freq: float = 781250,
        selected: list[bool] | None = None,
        apply_narrow_band_filter: bool = True
    ):
        self.search_band_min = search_band_min
        self.search_band_max = search_band_max
        self.sampling_freq = sampling_freq
        self.sampling_period = 1 / sampling_freq

        self.selected = selected if selected is not None else [True] * 4

        self.hydrophones = [
            hydrophone_module.Hydrophone(),
            hydrophone_module.Hydrophone(),
            hydrophone_module.Hydrophone(),
            hydrophone_module.Hydrophone()
        ]
        self.apply_narrow_band_filter = apply_narrow_band_filter
        self.narrow_band_width = 100

    def load_from_path(self, path: str) -> None:
        """Load hydrophone data from a file (bin or csv format)."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".bin":
            self._load_from_bin(path)
        elif ext == ".csv":
            self._load_from_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Expected .bin or .csv")

        self._apply_bandpass()

    def _load_from_csv(self, path: str) -> None:
        self._reset_hydrophones()

        skip_rows = 0
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                parts = line.strip().split(',')
                try:
                    float(parts[0])
                    skip_rows = i
                    break
                except ValueError:
                    continue

        data = pd.read_csv(path, skiprows=skip_rows, header=None)

        times = data.iloc[:, 0].to_numpy()
        for idx, hydro in enumerate(self.hydrophones):
            self._update_hydrophone(
                hydro, times, data.iloc[:, idx + 1].to_numpy()
            )

    def _load_from_bin(self, path: str) -> None:
        self._reset_hydrophones()

        with open(path, "rb") as f:
            # Read header: 8 bytes uint64, 4 bytes uint32, 8 bytes double
            header = f.read(8 + 4 + 8)
            num_samples, num_channels, sample_period = struct.unpack(
                "<QId", header
            )

            # read all float32 samples
            total_floats = num_samples * num_channels
            float_bytes = f.read(total_floats * 4)
            data = np.frombuffer(float_bytes, dtype="<f4")
            data = data.reshape((num_channels, num_samples))

        times = np.arange(num_samples, dtype=np.float64) * sample_period
        for idx, hydro in enumerate(self.hydrophones):
            self._update_hydrophone(hydro, times, data[idx])

    def _update_hydrophone(self, hydro, times, signal):
        hydro.times = times
        hydro.signal = signal
        hydro.freqs = fftfreq(len(signal), self.sampling_period)
        hydro.frequency = fft(signal)

    def _reset_hydrophones(self):
        for hydro in self.hydrophones:
            hydro.reset()

    def _apply_bandpass(self):
        if self.apply_narrow_band_filter:
            peak_freqs = []
            for hydro, is_selected in zip(self.hydrophones, self.selected):
                if is_selected:
                    freq_mask = (
                        (hydro.freqs >= self.search_band_min) &
                        (hydro.freqs <= self.search_band_max)
                    )
                    band_freqs = hydro.freqs[freq_mask]
                    band_magnitude = np.abs(hydro.frequency[freq_mask])

                    peak_idx = np.argmax(band_magnitude)
                    peak_freq = band_freqs[peak_idx]
                    peak_freqs.append(peak_freq)

            center_freq = np.mean(peak_freqs)

            for hydro, is_selected in zip(self.hydrophones, self.selected):
                if is_selected:
                    self._bandpass_signal(
                        hydro,
                        band_min=center_freq - self.narrow_band_width,
                        band_max=center_freq + self.narrow_band_width
                    )
        else:
            for hydro, is_selected in zip(self.hydrophones, self.selected):
                if is_selected:
                    self._bandpass_signal(hydro)

    def _bandpass_signal(self, hydro, band_min=None, band_max=None, order=16):
        if band_min is None:
            band_min = self.search_band_min
        if band_max is None:
            band_max = self.search_band_max

        sos = butter(
            order,
            [band_min, band_max],
            fs=self.sampling_freq,
            btype='band',
            output='sos'
        )
        hydro.filtered_signal = sosfilt(sos, hydro.signal)
        hydro.filtered_frequency = fft(hydro.filtered_signal)

    def plot_hydrophones(self, option="signal"):
        """Plot hydrophone data with specified visualization option."""
        num_subplots = sum(self.selected)
        _, axes = plt.subplots(num_subplots, 1, figsize=(10, 10), sharex=True)
        if num_subplots == 1:
            axes = [axes]

        plot_idx = 0
        for i, (hydro, is_selected) in enumerate(
            zip(self.hydrophones, self.selected)
        ):
            if is_selected:
                if option == "signal":
                    x_label = "Signal"
                    y_label = "Time (s)"
                    x_axis = hydro.times
                    y_axis = hydro.signal
                elif option == "filtered_signal":
                    x_label = "Signal"
                    y_label = "Time (s)"
                    x_axis = hydro.times
                    y_axis = hydro.filtered_signal
                elif option == "frequency":
                    x_label = "Frequency"
                    y_label = "Frequency Content"
                    x_axis = hydro.freqs
                    y_axis = np.abs(hydro.frequency)
                elif option == "filtered_frequency":
                    x_label = "Filtered Frequency"
                    y_label = "Frequency Content"
                    x_axis = hydro.freqs
                    y_axis = np.abs(hydro.filtered_frequency)
                else:
                    print("not a valid option")
                    return

                axes[plot_idx].plot(x_axis, y_axis, label="Original")
                axes[plot_idx].set_xlabel(x_label)
                axes[plot_idx].set_ylabel(y_label)
                axes[plot_idx].legend(loc="best")
                axes[plot_idx].grid(True)
                axes[plot_idx].set_title(f"Hydrophone {i+1}")
                plot_idx += 1

        plt.tight_layout()
        plt.show()
