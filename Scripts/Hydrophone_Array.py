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
    """Manages an array of hydrophones with signal processing and time-of-arrival detection capabilities."""

    def __init__(
        self,
        sampling_freq: float = 781250,
        search_band_min: float = 25000,
        search_band_max: float = 40000,
        bandwidth: float = 25.0,
        enable_data_sample: bool = False,
        data_sample_out_dir: str = "",
    ):
        self.search_band_min = search_band_min
        self.search_band_max = search_band_max
        self.bandwidth = float(bandwidth)
        self.sampling_freq = float(sampling_freq)
        self.dt = 1 / sampling_freq

        self.hydrophones: List[Hydrophone.Hydrophone] = [
            Hydrophone.Hydrophone() for _ in range(4)
        ]

        self.threshold_factor = 0.7

        self.last_data = ""

        self.enable_data_sample = enable_data_sample 
        self.data_sample_out_dir = data_sample_out_dir
        if self.enable_data_sample:
            self.data_sample_path = self.setup_data_sample()

    def setup_data_sample(self):
        out_dir = self.data_sample_out_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        # timestamped filename to avoid clobbering existing files
        ts = time.strftime('%Y-%m-%d--%H-%M-%S')
        filename = f"data_sample_{ts}.csv"
        path = os.path.join(out_dir, filename)

        headers = [
            "Truth",
            "Envelope",
            "Envelope H0",
            "Envelope H1",
            "Envelope H2",
            "Envelope H3",
            "GCC",
            "GCC H0",
            "GCC H1",
            "GCC H2",
            "GCC H3",
        ]
        with open(path, mode="w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)

        return path

    def data_sample(self, truth = None):
        if truth is None:
            if self.last_data:
                base = os.path.basename(self.last_data)
                prefix = base.split("_", 1)[0]
                if prefix.isdigit():
                    val = int(prefix)
                    if 0 <= val <= 7:
                        truth = val

        # Determine earliest hydrophone by envelope (smallest toa_time)
        envelope_candidates = [
            (i, hp.toa_time)
            for i, hp in enumerate(self.hydrophones)
            if getattr(hp, "toa_time", None) is not None and getattr(hp, "found_peak", False)
        ]
        if envelope_candidates:
            envelope_first = min(envelope_candidates, key=lambda x: x[1])[0]
        else:
            envelope_first = ""

        # Determine earliest hydrophone by GCC (smallest gcc_tdoa)
        gcc_candidates = [
            (i, hp.gcc_tdoa)
            for i, hp in enumerate(self.hydrophones)
            if getattr(hp, "gcc_tdoa", None) is not None
        ]
        if gcc_candidates:
            gcc_first = min(gcc_candidates, key=lambda x: x[1])[0]
        else:
            gcc_first = ""

        row = [
            truth,
            envelope_first,
            self.hydrophones[0].toa_time - self.hydrophones[0].toa_time,
            self.hydrophones[1].toa_time - self.hydrophones[0].toa_time,
            self.hydrophones[2].toa_time - self.hydrophones[0].toa_time,
            self.hydrophones[3].toa_time - self.hydrophones[0].toa_time,
            gcc_first,
            self.hydrophones[0].gcc_tdoa,
            self.hydrophones[1].gcc_tdoa,
            self.hydrophones[2].gcc_tdoa,
            self.hydrophones[3].gcc_tdoa,
        ]

        # Ensure data sample file exists (create header if missing)
        if not getattr(self, "data_sample_path", None):
            self.data_sample_path = self.setup_data_sample()

        # Append row to the CSV (don't overwrite header)
        with open(self.data_sample_path, mode="a", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

    # Goal: Load time-voltage data from a file into hydrophone array
    # Return: None
    def load_from_path(self, path: str)-> None:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".bin":
            self.last_data = path
            return self.load_from_bin(path)
        elif ext == ".csv":
            self.last_data = path
            return self.load_from_csv(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Expected .bin or .csv")
        
    # Goal: Load time-voltage data from a CSV file into hydrophone array
    # How: Detects and skips header rows, then populates each hydrophone with time and voltage data
    # Return: None
    def load_from_csv(self, path: str) -> None:
        self.reset_selected()

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
        
        self.hydrophones[0].times = times
        self.hydrophones[0].voltages = data.iloc[:, 1].to_numpy()

        # for idx, hydrophone in enumerate(self.hydrophones):
        #     hydrophone.times = times
        #     hydrophone.voltages = data.iloc[:, idx + 1].to_numpy()

    # Goal: Load time-voltage data from a binary file into hydrophone array
    # How: Parses binary header to extract sample count, then reads voltage samples for each hydrophone channel
    # Return: None
    def load_from_bin(self, path: str) -> None:
        self.reset_selected()

        with open(path, "rb") as f:
            # Read header: 8 bytes uint64, 4 bytes uint32, 8 bytes double (little-endian)
            header = f.read(8 + 4 + 8)
            num_samples, num_channels, sample_period = struct.unpack("<QId", header)

            # read all float32 samples
            total_floats = num_samples * num_channels
            float_bytes = f.read(total_floats * 4)
            data = np.frombuffer(float_bytes, dtype="<f4")  # little-endian float32
            data = data.reshape((num_channels, num_samples))

        # Create time base
        times = np.arange(num_samples, dtype=np.float64) * sample_period
        for idx, hydrophone in enumerate(self.hydrophones):
            hydrophone.times = times
            hydrophone.voltages = data[idx]

    # Goal: Normalize selection mask to match hydrophone array length
    # How: Returns all-True mask if None, otherwise adjusts mask length to match hydrophone count
    # Return: List[bool] with length equal to number of hydrophones
    def _normalize_selection(self, selected: Optional[List[bool]]) -> List[bool]:
        if selected is None:
            return [True] * len(self.hydrophones)
        if len(selected) != len(self.hydrophones):
            return list(selected[:len(self.hydrophones)]) + [False] * max(0, len(self.hydrophones) - len(selected))
        return selected

    # Goal: Plot time-series envelope data for selected hydrophones
    # How: Creates subplots for each selected hydrophone and displays their envelope detection results
    # Return: None (displays matplotlib figure)
    def plot_selected_envelope(self, selected: Optional[List[bool]] = None, show_frequency_domain: bool = False) -> None:
        selected = self._normalize_selection(selected)
        num_subplots = sum(selected)
        
        if show_frequency_domain:
            # Create two columns: time domain and frequency domain
            fig, axes = plt.subplots(num_subplots, 2, figsize=(16, 10), sharex=False)
            if num_subplots == 1:
                axes = axes.reshape(1, -1)  # Ensure 2D array
        else:
            # Original single column layout
            fig, axes = plt.subplots(num_subplots, 1, figsize=(10, 10), sharex=True)
            if num_subplots == 1:
                axes = [axes]

        plot_idx = 0
        for i, (hydrophone, is_selected) in enumerate(zip(self.hydrophones, selected)):
            if is_selected:
                if show_frequency_domain:
                    # Plot time domain in left column
                    self._plot_hydrophone_envelope(hydrophone, axes[plot_idx, 0])
                    axes[plot_idx, 0].set_title(f"Hydrophone {i} - Time Domain")
                    
                    # Plot frequency domain in right column
                    self._plot_hydrophone_frequency(hydrophone, axes[plot_idx, 1])
                    axes[plot_idx, 1].set_title(f"Hydrophone {i} - Frequency Domain")
                else:
                    # Original time domain only
                    self._plot_hydrophone_envelope(hydrophone, axes[plot_idx])
                    axes[plot_idx].set_title(f"Hydrophone {i} - ToA Detection")
                plot_idx += 1

        plt.tight_layout()
        plt.show()

    # Goal: Plot frequency domain analysis for a single hydrophone
    # How: Displays FFT of original, filtered, and envelope signals with frequency bands marked
    # Return: None (modifies matplotlib axis in place)
    def _plot_hydrophone_frequency(self, hydrophone: Hydrophone.Hydrophone, ax) -> None:
        if hydrophone.found_peak is False or hydrophone.voltages is None:
            ax.text(0.5, 0.5, "No data loaded", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Frequency Domain (No Data)")
            ax.axis("off")
            return

        # Compute FFT of original signal
        n = len(hydrophone.voltages)
        fft_original = np.abs(fft(hydrophone.voltages))
        freqs = fftfreq(n, self.dt)
        
        # Keep only positive frequencies (first half)
        positive_freqs = freqs[:n//2]  # Keep in Hz first
        positive_fft_original = fft_original[:n//2]
        
        # Plot original signal spectrum
        ax.plot(positive_freqs / 1000, 20*np.log10(positive_fft_original + 1e-10), 
                label="Original", alpha=0.7, linewidth=1)
        
        # Plot filtered signal spectrum if available
        if getattr(hydrophone, "filtered_signal", None) is not None:
            fft_filtered = np.abs(fft(hydrophone.filtered_signal))
            positive_fft_filtered = fft_filtered[:n//2]
            ax.plot(positive_freqs / 1000, 20*np.log10(positive_fft_filtered + 1e-10), 
                    label="Filtered", alpha=0.8, linewidth=1.5)
        
        # Plot envelope spectrum if available
        if getattr(hydrophone, "envelope", None) is not None:
            fft_envelope = np.abs(fft(hydrophone.envelope))
            positive_fft_envelope = fft_envelope[:n//2]
            ax.plot(positive_freqs / 1000, 20*np.log10(positive_fft_envelope + 1e-10), 
                    label="Envelope", linestyle="--", alpha=0.8, linewidth=1.5)
        
        # Mark signal band
        ax.axvspan(self.search_band_min/1000, self.search_band_max/1000, 
                   alpha=0.2, color='green', label=f'Signal Band ({self.search_band_min/1000:.0f}-{self.search_band_max/1000:.0f} kHz)')
        
        # Mark narrow band if used
        if hasattr(self, 'last_center_freq') and self.last_center_freq is not None:
            low_freq = (self.last_center_freq - self.bandwidth/2) / 1000
            high_freq = (self.last_center_freq + self.bandwidth/2) / 1000
            ax.axvspan(low_freq, high_freq, alpha=0.3, color='orange', 
                      label=f'Narrow Band ({low_freq:.1f}-{high_freq:.1f} kHz)')
        
        ax.set_xlabel("Frequency (kHz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_xlim(0, min(100, max(positive_freqs / 1000)))  # Limit to 100 kHz or max freq
        ax.legend(loc="best", fontsize='small')
        ax.grid(True, alpha=0.3)

    # Goal: Plot envelope detection results for a single hydrophone
    # How: Displays original signal, filtered signal, envelope, and time-of-arrival marker
    # Return: None (modifies matplotlib axis in place)
    @staticmethod
    def _plot_hydrophone_envelope(hydrophone: Hydrophone.Hydrophone, ax) -> None:
        if hydrophone.found_peak is False:
            ax.text(0.5, 0.5, "No data loaded", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("ToA Detection (No Data)")
            ax.axis("off")
            return

        ax.plot(hydrophone.times, hydrophone.voltages, label="Original")
        if getattr(hydrophone, "filtered_signal", None) is not None:
            ax.plot(hydrophone.times, hydrophone.filtered_signal, label="Filtered")
        if getattr(hydrophone, "envelope", None) is not None:
            ax.plot(hydrophone.times, hydrophone.envelope, label="Envelope", linestyle="--")
        if getattr(hydrophone, "toa_time", None) is not None:
            ax.axvline(hydrophone.toa_time, color="r", linestyle=":", label=f"ToA = {hydrophone.toa_time:.6f}s")

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Voltage")
        ax.set_title("ToA Detection")
        ax.legend(loc="best")
        ax.grid(True)

    # Goal: Print time-of-arrival for all hydrophones sorted by detection time
    # How: Sorts hydrophones by TOA and prints each with detection status
    # Return: None (prints to console)
    def print_envelope_toas(self) -> None:
        sorted_hydrophones = sorted(
            enumerate(self.hydrophones),
            key=lambda ih: (not ih[1].found_peak, ih[1].toa_time if ih[1].toa_time is not None else float('inf'))
        )

        for i, hydrophone in sorted_hydrophones:
            if hydrophone.toa_time is not None:
                print(f"Hydrophone {i} saw ping at {hydrophone.toa_time:.6f}s (found_peak={hydrophone.found_peak})")
            else:
                print(f"Hydrophone {i} saw ping at N/A (found_peak={hydrophone.found_peak})")

    # Goal: Find the peak frequency in the signal band across all hydrophones
    # How: Computes FFT for each hydrophone, finds peaks in search band, returns average
    # Return: float - center frequency in Hz
    def find_signal_center_frequency(self, selected: Optional[List[bool]] = None) -> float:
        selected = self._normalize_selection(selected)
        peak_frequencies = []
        
        for hydrophone, is_selected in zip(self.hydrophones, selected):
            if not is_selected or hydrophone.voltages is None:
                continue
                
            # Compute FFT - only need positive frequencies
            fft_vals = np.abs(fft(hydrophone.voltages))
            n = len(fft_vals)
            freqs = np.fft.fftfreq(n, self.dt)
            
            # Keep only positive frequencies (first half)
            positive_freqs = freqs[:n//2]
            positive_fft = fft_vals[:n//2]
            
            # Find peak in signal band (25-40 kHz)
            band_mask = (positive_freqs >= self.search_band_min) & (positive_freqs <= self.search_band_max)
            if not np.any(band_mask):
                continue
                
            # Get peak frequency in the band
            band_fft = positive_fft[band_mask]
            band_freqs = positive_freqs[band_mask]
            peak_idx = np.argmax(band_fft)
            peak_freq = band_freqs[peak_idx]
            peak_frequencies.append(peak_freq)
        
        if not peak_frequencies:
            print("Warning: No peak frequencies found, using band center")
            return (self.search_band_min + self.search_band_max) / 2
        
        center_freq = np.mean(peak_frequencies)
        print(f"Found signal center frequency: {center_freq:.1f} Hz")
        
        # Store for frequency plotting
        self.last_center_freq = center_freq
        
        return center_freq

    # Goal: Apply bandpass filter to hydrophone signal using Butterworth filter
    # How: Uses scipy.signal.butter to create a Butterworth bandpass filter
    # Return: np.ndarray - filtered signal (also stores in hydrophone object)
    def apply_bandpass_filter(
        self,
        hydrophone: Hydrophone.Hydrophone,
        order: int = 16,
        center_freq: Optional[float] = None,
        use_narrow_band: bool = False
    ) -> np.ndarray:
        
        if use_narrow_band and center_freq is not None:
            # Use narrow bandpass filter around center frequency
            low_freq = center_freq - self.bandwidth / 2
            high_freq = center_freq + self.bandwidth / 2

            sos = butter(order, [low_freq, high_freq], fs=self.sampling_freq, btype='band', output='sos')
        else:
            # Use original wide bandpass filter
            sos = butter(order, [self.search_band_min, self.search_band_max], fs=self.sampling_freq, btype='band', output='sos')
        
        # Apply filter
        filtered_signal = sosfilt(sos, hydrophone.voltages)
        
        # Store results in hydrophone object
        hydrophone.filtered_signal = filtered_signal
        hydrophone.found_peak = True
        
        return filtered_signal
                     
    # Goal: Apply spectral whitening to filtered signal
    # How: Normalizes magnitude spectrum while preserving phase, trims edges to avoid artifacts
    # Return: np.ndarray - whitened signal
    def apply_whitening(self, filtered_signal: np.ndarray, trim_percent: float = 0.05, whitening_strength: float = 0.3) -> np.ndarray:
        fft_signal = fft(filtered_signal)
        magnitude = np.abs(fft_signal)
        phase = np.angle(fft_signal)
        
        # Gentle whitening: blend between original magnitude and flat magnitude
        magnitude_white = np.ones_like(magnitude)
        magnitude_blended = (1 - whitening_strength) * magnitude + whitening_strength * magnitude_white * np.mean(magnitude)
        
        # Reconstruct with blended magnitude, original phase
        fft_white = magnitude_blended * np.exp(1j * phase)
        whitened_signal = np.real(ifft(fft_white))
        
        # Trim edges to remove artifacts
        trim_samples = int(len(whitened_signal) * trim_percent)
        if trim_samples > 0:
            whitened_signal[:trim_samples] = filtered_signal[:trim_samples]
            whitened_signal[-trim_samples:] = filtered_signal[-trim_samples:]
        
        return whitened_signal

    # Goal: Estimate time-of-arrival using envelope detection method
    # How: Applies bandpass filter, computes Hilbert envelope, and detects TOA via threshold crossing
    # Return: None (modifies hydrophone object with TOA data)
    def estimate_toa_by_envelope(self, hydrophone: Hydrophone.Hydrophone, center_freq: Optional[float] = None, use_narrow_band: bool = True, apply_whitening: bool = True) -> None:
        if hydrophone.times is None or hydrophone.voltages is None:
            raise RuntimeError("Load data first with load_from_csv().")

        filtered_signal = self.apply_bandpass_filter(hydrophone, center_freq=center_freq, use_narrow_band=use_narrow_band)

        if apply_whitening:
            whitened_signal = self.apply_whitening(filtered_signal)
            envelope = np.abs(hilbert(whitened_signal))
        else:
            envelope = np.abs(hilbert(filtered_signal))

        # Use more robust threshold based on signal statistics
        envelope_mean = np.mean(envelope)
        envelope_std = np.std(envelope)
        threshold = envelope_mean + 5 * envelope_std  # 3-sigma threshold
        
        toa_candidates = np.where(envelope > threshold)[0]
        if len(toa_candidates) == 0:
            threshold = float(self.threshold_factor) * float(np.max(envelope))
            toa_idx = np.argmax(envelope > threshold)
        else:
            toa_idx = toa_candidates[0]  # First crossing
        
        toa_time = hydrophone.times[toa_idx]

        toa_idx_peak = int(np.argmax(envelope))
        toa_idx = toa_idx_peak
        toa_peak = hydrophone.times[toa_idx_peak]

        hydrophone.found_peak = True
        hydrophone.toa_idx = toa_idx
        hydrophone.toa_time = toa_time
        hydrophone.toa_peak = toa_peak
        hydrophone.envelope = envelope

    # Goal: Run envelope-based TOA estimation on selected hydrophones
    # How: Iterates through selected hydrophones and applies envelope detection to each
    # Return: None (modifies selected hydrophone objects)
    def estimate_selected_by_envelope(self, selected: Optional[List[bool]] = None, use_narrow_band: bool = True, apply_whitening: bool = True) -> None:
        selected = self._normalize_selection(selected)
        
        # Find center frequency if using narrow band filtering
        center_freq = None
        if use_narrow_band:
            center_freq = self.find_signal_center_frequency(selected)
        
        for hydrophone, is_selected in zip(self.hydrophones, selected):
            if is_selected:
                self.estimate_toa_by_envelope(hydrophone, center_freq=center_freq, use_narrow_band=use_narrow_band, apply_whitening=apply_whitening)

    # Goal: Reset selected hydrophones to clear previously loaded data
    # How: Calls reset method on each selected hydrophone object
    # Return: None (modifies hydrophone objects)
    def reset_selected(self, selected: Optional[List[bool]] = None) -> None:
        selected = self._normalize_selection(selected)
        for hydrophone, is_selected in zip(self.hydrophones, selected):
            if is_selected:
                hydrophone.reset()

    # Goal: Compute GCC-PHAT cross-correlation between two signals for time delay estimation
    # How: Uses FFT-based cross-correlation with phase transform weighting for noise robustness
    # Return: Tuple of (cross_correlation array, lag index, time delay in seconds)
    def compute_gcc_phat(self, signal1: np.ndarray, signal2: np.ndarray) -> tuple[np.ndarray, int, float]:
        fft1 = fft(signal1)
        fft2 = fft(signal2)

        cross_spectrum = fft1 * np.conj(fft2)

        epsilon = 1e-10
        phat_weighted = cross_spectrum / (np.abs(cross_spectrum) + epsilon)

        gcc = np.real(ifft(phat_weighted))

        peak_idx = np.argmax(gcc)

        n = len(gcc)
        if peak_idx > n // 2:
            lag = peak_idx - n
        else:
            lag = peak_idx

        tdoa = lag * self.dt

        return gcc, lag, tdoa

    # Goal: Compute signal quality metrics for diagnostic purposes
    # How: Calculates peak sharpness and coherence measures
    # Return: Tuple of (peak_sharpness, coherence_score)
    def compute_signal_quality(self, gcc: np.ndarray, signal1: np.ndarray, signal2: np.ndarray) -> tuple[float, float]:
        # Peak sharpness: ratio of peak to mean
        peak_value = np.max(gcc)
        mean_value = np.mean(gcc)
        peak_sharpness = peak_value / (mean_value + 1e-10)
        
        # Simple coherence: correlation coefficient between signals
        correlation = np.corrcoef(signal1, signal2)[0, 1]
        coherence_score = abs(correlation)
        
        return peak_sharpness, coherence_score

    # Goal: Estimate time difference of arrival using GCC-PHAT for selected hydrophones
    # How: Computes all pairwise GCC correlations and uses voting to get robust estimates
    # Return: None (modifies hydrophone objects with TDOA data)
    def estimate_selected_by_gcc(self, selected: Optional[List[bool]] = None, use_narrow_band: bool = True, use_multi_reference: bool = True) -> None:
        selected = self._normalize_selection(selected)
        selected_indices = [i for i, is_selected in enumerate(selected) if is_selected]

        if len(selected_indices) < 2:
            print("Warning: Need at least 2 hydrophones selected for GCC-PHAT TDOA estimation")
            return

        # Find center frequency if using narrow band filtering
        center_freq = None
        if use_narrow_band:
            center_freq = self.find_signal_center_frequency(selected)

        # Apply bandpass filter to all selected hydrophones
        filtered_signals = {}
        for idx in selected_indices:
            hydrophone = self.hydrophones[idx]
            if hydrophone.voltages is None:
                print(f"Warning: Hydrophone {idx} has no voltage data, skipping")
                continue
            filtered_signals[idx] = self.apply_bandpass_filter(hydrophone, center_freq=center_freq, use_narrow_band=use_narrow_band)

        if not use_multi_reference:
            # Use original single reference method
            ref_idx = selected_indices[0]
            ref_filtered = filtered_signals[ref_idx]
            
            for idx in selected_indices:
                if idx not in filtered_signals:
                    continue
                    
                if idx == ref_idx:
                    self.hydrophones[idx].gcc_tdoa = 0.0
                    continue
                    
                gcc, lag, tdoa = self.compute_gcc_phat(ref_filtered, filtered_signals[idx])
                peak_sharpness, coherence = self.compute_signal_quality(gcc, ref_filtered, filtered_signals[idx])
                
                print(f"H{ref_idx}-H{idx}: TDOA={tdoa*1e6:.2f}μs, Sharpness={peak_sharpness:.2f}, Coherence={coherence:.3f}")
                
                self.hydrophones[idx].gcc_tdoa = tdoa
                self.hydrophones[idx].gcc_cc = gcc
        else:
            # Multi-reference approach
            print("\n=== Multi-Reference GCC Analysis ===")
            
            # Compute all pairwise TDOAs
            pairwise_tdoas = {}
            quality_scores = {}
            
            for i, idx1 in enumerate(selected_indices):
                for j, idx2 in enumerate(selected_indices):
                    if i >= j or idx1 not in filtered_signals or idx2 not in filtered_signals:
                        continue
                        
                    gcc, lag, tdoa = self.compute_gcc_phat(filtered_signals[idx1], filtered_signals[idx2])
                    peak_sharpness, coherence = self.compute_signal_quality(gcc, filtered_signals[idx1], filtered_signals[idx2])
                    
                    pair_key = (idx1, idx2)
                    pairwise_tdoas[pair_key] = tdoa
                    quality_scores[pair_key] = (peak_sharpness, coherence)
                    
                    print(f"H{idx1}-H{idx2}: TDOA={tdoa*1e6:.2f}μs, Sharpness={peak_sharpness:.2f}, Coherence={coherence:.3f}")
            
            # Use first hydrophone as reference and compute relative TDOAs
            ref_idx = selected_indices[0]
            
            # Validate triangulation consistency
            print("\n=== Triangulation Validation ===")
            for i in range(len(selected_indices)):
                for j in range(i+1, len(selected_indices)):
                    for k in range(j+1, len(selected_indices)):
                        idx_i, idx_j, idx_k = selected_indices[i], selected_indices[j], selected_indices[k]
                        
                        # Check if we have all three pairs
                        pair_ij = (idx_i, idx_j) if idx_i < idx_j else (idx_j, idx_i)
                        pair_ik = (idx_i, idx_k) if idx_i < idx_k else (idx_k, idx_i)
                        pair_jk = (idx_j, idx_k) if idx_j < idx_k else (idx_k, idx_j)
                        
                        if pair_ij in pairwise_tdoas and pair_ik in pairwise_tdoas and pair_jk in pairwise_tdoas:
                            # Get TDOAs with correct signs
                            tdoa_ij = pairwise_tdoas[pair_ij] if idx_i < idx_j else -pairwise_tdoas[pair_ij]
                            tdoa_ik = pairwise_tdoas[pair_ik] if idx_i < idx_k else -pairwise_tdoas[pair_ik]
                            tdoa_jk = pairwise_tdoas[pair_jk] if idx_j < idx_k else -pairwise_tdoas[pair_jk]
                            
                            # Check triangulation: tdoa_ij + tdoa_jk should ≈ tdoa_ik
                            expected_ik = tdoa_ij + tdoa_jk
                            error = abs(tdoa_ik - expected_ik)
                            error_us = error * 1e6
                            
                            status = "✓" if error_us < 2.0 else "✗"
                            print(f"{status} Triangle H{idx_i}-H{idx_j}-H{idx_k}: error={error_us:.2f}μs")
            
            # Assign final TDOAs using validated measurements
            for idx in selected_indices:
                if idx == ref_idx:
                    self.hydrophones[idx].gcc_tdoa = 0.0
                    continue
                    
                if idx not in filtered_signals:
                    continue
                
                # Find direct measurement with correct sign
                pair_key = (ref_idx, idx) if ref_idx < idx else (idx, ref_idx)
                
                if pair_key in pairwise_tdoas:
                    # Get TDOA with correct sign
                    tdoa = pairwise_tdoas[pair_key] 
                    if ref_idx > idx:  # We need to flip if ref comes after target
                        tdoa = -tdoa
                    
                    self.hydrophones[idx].gcc_tdoa = tdoa
                    
                    # Store the GCC for plotting (recompute with correct order)
                    gcc, _, _ = self.compute_gcc_phat(filtered_signals[ref_idx], filtered_signals[idx])
                    self.hydrophones[idx].gcc_cc = gcc
                    
                    # Cross-validate with other measurements if available
                    validations = []
                    for other_idx in selected_indices:
                        if other_idx == ref_idx or other_idx == idx:
                            continue
                        
                        # Check if we can compute ref->idx via ref->other->idx
                        pair_ref_other = (ref_idx, other_idx) if ref_idx < other_idx else (other_idx, ref_idx)
                        pair_other_idx = (other_idx, idx) if other_idx < idx else (idx, other_idx)
                        
                        if pair_ref_other in pairwise_tdoas and pair_other_idx in pairwise_tdoas:
                            tdoa_ref_other = pairwise_tdoas[pair_ref_other]
                            if ref_idx > other_idx:
                                tdoa_ref_other = -tdoa_ref_other
                                
                            tdoa_other_idx = pairwise_tdoas[pair_other_idx]
                            if other_idx > idx:
                                tdoa_other_idx = -tdoa_other_idx
                            
                            indirect_tdoa = tdoa_ref_other + tdoa_other_idx
                            validations.append(indirect_tdoa)
                    
                    if validations:
                        avg_validation = np.mean(validations)
                        validation_error = abs(tdoa - avg_validation) * 1e6
                        status = "✓" if validation_error < 2.0 else "⚠"
                        print(f"{status} H{ref_idx}->H{idx}: direct={tdoa*1e6:.2f}μs, indirect={avg_validation*1e6:.2f}μs, error={validation_error:.2f}μs")
                else:
                    print(f"Warning: No direct measurement for H{ref_idx}-H{idx}")
            
            print("\n=== Quality Summary ===")
            avg_sharpness = np.mean([scores[0] for scores in quality_scores.values()])
            avg_coherence = np.mean([scores[1] for scores in quality_scores.values()])
            print(f"Average Peak Sharpness: {avg_sharpness:.2f}")
            print(f"Average Coherence: {avg_coherence:.3f}")
            
            # Detect outliers (simple threshold-based)
            outliers = []
            for pair, (sharpness, coherence) in quality_scores.items():
                if sharpness < avg_sharpness * 0.5 or coherence < 0.3:
                    outliers.append(pair)
            
            if outliers:
                print(f"Potential outlier pairs: {outliers}")
            else:
                print("No obvious outliers detected")
        

    # Goal: Print TDOA results from GCC-PHAT estimation
    # How: Iterates through selected hydrophones and displays their time delay values
    # Return: None (prints to console)
    def print_gcc_tdoa(self, selected: Optional[List[bool]] = None) -> None:
        selected = self._normalize_selection(selected)
        for i, (hydrophone, is_selected) in enumerate(zip(self.hydrophones, selected)):
            if is_selected:
                if hydrophone.gcc_tdoa is not None:
                    print(f"Hydrophone {i}: TDOA = {hydrophone.gcc_tdoa * 1e6:.2f} μs ({hydrophone.gcc_tdoa:.9f} s)")
                else:
                    print(f"Hydrophone {i}: TDOA = N/A")

    # Goal: Plot GCC-PHAT cross-correlation results for selected hydrophones
    # How: Creates subplots showing correlation vs time delay with TDOA markers
    # Return: None (displays matplotlib figure)
    def plot_selected_gcc(self, selected: Optional[List[bool]] = None) -> None:
        selected = self._normalize_selection(selected)
        selected_indices = [i for i, is_selected in enumerate(selected) if is_selected]

        plot_indices = [
            i for i in selected_indices
            if self.hydrophones[i].gcc_cc is not None
        ]

        if not plot_indices:
            print("No GCC-PHAT data to plot")
            return

        num_plots = len(plot_indices)
        fig, axes = plt.subplots(num_plots, 1, figsize=(12, 3 * num_plots), sharex=True)

        if num_plots == 1:
            axes = [axes]

        for plot_idx, hydro_idx in enumerate(plot_indices):
            hydrophone = self.hydrophones[hydro_idx]
            gcc = hydrophone.gcc_cc

            n = len(gcc)
            lags = np.arange(-n // 2, n // 2 + (n % 2))
            time_lags = lags * self.dt * 1e6

            gcc_shifted = np.fft.fftshift(gcc)

            ax = axes[plot_idx]
            ax.plot(time_lags, gcc_shifted, linewidth=1)

            if hydrophone.gcc_tdoa is not None:
                tdoa_us = hydrophone.gcc_tdoa * 1e6
                ax.axvline(tdoa_us, color='r', linestyle='--',
                          label=f'TDOA = {tdoa_us:.2f} μs')

            ax.set_ylabel('Correlation')
            ax.set_title(f'Hydrophone {hydro_idx} GCC-PHAT')
            ax.grid(True, alpha=0.3)
            ax.legend()

        axes[-1].set_xlabel('Time Delay (μs)')
        plt.tight_layout()
        plt.show()
