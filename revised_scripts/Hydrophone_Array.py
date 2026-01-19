from __future__ import annotations

from typing import List, Optional
import os
import csv

import numpy as np
import pandas as pd
import struct
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, sosfilt
from scipy.fft import fft, ifft, fftfreq
import time


import Hydrophone

class HydrophoneArray:
    def __init__(
        self,
        search_band_min: float = 25000,
        search_band_max: float = 40000,
        sampling_freq: float = 781250,
        selected: list[bool] = [True, True, True, True],
        apply_narrow_band_filter = True
    ):
        self.search_band_min = search_band_min
        self.search_band_max = search_band_max
        self.sampling_freq = sampling_freq
        self.sampling_period = 1 / sampling_freq

        self.selected = selected

        self.hydrophones=[Hydrophone.Hydrophone(), Hydrophone.Hydrophone(), 
                          Hydrophone.Hydrophone(), Hydrophone.Hydrophone()]
        self.apply_narrow_band_filter = apply_narrow_band_filter
        self.narrow_band_width = 100

    def load_from_path(self, path: str)-> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".bin":
            self.load_from_bin(path)
        elif ext == ".csv":
            self.load_from_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Expected .bin or .csv")

        self.apply_bandpass()

    def load_from_csv(self, path: str) -> None:
        self._reset_hydrophones()

        skip_rows = 0
        with open(path, 'r') as f:
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
        for idx, hydrophone in enumerate(self.hydrophones):
            self._update_hydrophone(hydrophone, times, data.iloc[:, idx + 1].to_numpy())

    def load_from_bin(self, path: str) -> None:
        self._reset_hydrophones()

        with open(path, "rb") as f:
            # Read header: 8 bytes uint64, 4 bytes uint32, 8 bytes double (little-endian)
            header = f.read(8 + 4 + 8)
            num_samples, num_channels, sample_period = struct.unpack("<QId", header)

            # read all float32 samples
            total_floats = num_samples * num_channels
            float_bytes = f.read(total_floats * 4)
            data = np.frombuffer(float_bytes, dtype="<f4")
            data = data.reshape((num_channels, num_samples))

        times = np.arange(num_samples, dtype=np.float64) * sample_period
        for idx, hydrophone in enumerate(self.hydrophones):
            self._update_hydrophone(hydrophone, times, data[idx])

    def _update_hydrophone(self, hydrophone, times, signal):
        hydrophone.times = times
        hydrophone.signal = signal
        hydrophone.freqs = fftfreq(len(signal), self.sampling_period)
        hydrophone.frequency = fft(signal)

    def _reset_hydrophones(self):
        for hydrophone in self.hydrophones:
            hydrophone.reset()

    def apply_bandpass(self):
        if self.apply_narrow_band_filter:
            peak_freqs = []
            for hydrophone, is_selected in zip(self.hydrophones, self.selected):
                if is_selected:
                    freq_mask = (hydrophone.freqs >= self.search_band_min) & (hydrophone.freqs <= self.search_band_max)
                    band_freqs = hydrophone.freqs[freq_mask]
                    band_magnitude = np.abs(hydrophone.frequency[freq_mask])
                    
                    peak_idx = np.argmax(band_magnitude)
                    peak_freq = band_freqs[peak_idx]
                    peak_freqs.append(peak_freq)
            
            center_freq = np.mean(peak_freqs)
        
            for hydrophone, is_selected in zip(self.hydrophones, self.selected):
                if is_selected:
                    self._bandpass_signal(hydrophone, 
                                    band_min=center_freq - self.narrow_band_width,
                                    band_max=center_freq + self.narrow_band_width)
        else:
            for hydrophone, is_selected in zip(self.hydrophones, self.selected):
                if is_selected:
                    self._bandpass_signal(hydrophone)

    def _bandpass_signal(self, hydrophone, band_min=None, band_max=None, order=16):
        if band_min is None:
            band_min = self.search_band_min
        if band_max is None:
            band_max = self.search_band_max
        
        sos = butter(order, [band_min, band_max], 
                     fs=self.sampling_freq, btype='band', output='sos')
        hydrophone.filtered_signal = sosfilt(sos, hydrophone.signal)
        hydrophone.filtered_frequency = fft(hydrophone.filtered_signal)

    def plot_hydrophones(self, option = "signal"):
        num_subplots = sum(self.selected)      
        _, axes = plt.subplots(num_subplots, 1, figsize=(10, 10), sharex=True)
        if num_subplots == 1:
            axes = [axes]

        plot_idx = 0
        for i, (hydrophone,is_selected) in enumerate(zip(self.hydrophones, self.selected)):
            if is_selected:
                if (option == "signal"):
                    x_label = "Signal"
                    y_label = "Time (s)"
                    x_axis = hydrophone.times
                    y_axis = hydrophone.signal
                elif (option == "filtered_signal"):
                    x_label = "Signal"
                    y_label = "Time (s)"
                    x_axis = hydrophone.times
                    y_axis = hydrophone.filtered_signal
                elif (option == "frequency"):
                    x_label = "Frequency"
                    y_label = "Frequency Content"
                    x_axis = hydrophone.freqs
                    y_axis = np.abs(hydrophone.frequency)
                elif (option == "filtered_frequency"):
                    x_label = "Filtered Frequency"
                    y_label = "Frequency Content"
                    x_axis = hydrophone.freqs
                    y_axis = np.abs(hydrophone.filtered_frequency)
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
