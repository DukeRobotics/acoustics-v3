"""TOA estimation using Hilbert envelope detection."""
import numpy as np
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq

from .base_analyzer import BaseAnalyzer


class TOAEnvelopeAnalyzer(BaseAnalyzer):
    """TOA estimation using Hilbert envelope detection."""

    def __init__(self, threshold_sigma=5, **kwargs):
        super().__init__(**kwargs)
        self.threshold_sigma = threshold_sigma

    def get_name(self):
        return "TOA Envelope Detection"

    def _analyze_single(self, hydrophone, sampling_freq, center_freq):
        """Analyze one hydrophone using envelope detection."""
        # Determine bandpass range
        if self.use_narrow_band and center_freq:
            band_min = center_freq - self.narrow_band_width
            band_max = center_freq + self.narrow_band_width
        else:
            band_min = self.search_band_min
            band_max = self.search_band_max

        # Apply bandpass filter
        filtered_signal = self.apply_bandpass(
            hydrophone.signal, sampling_freq, band_min, band_max
        )

        # Compute envelope using Hilbert transform
        envelope = np.abs(hilbert(filtered_signal))

        # Detect TOA using threshold
        threshold = (
            np.mean(envelope) +
            self.threshold_sigma * np.std(envelope)
        )
        toa_candidates = np.where(envelope > threshold)[0]
        if len(toa_candidates) > 0:
            toa_idx = toa_candidates[0]  # First crossing
        else:
            toa_idx = np.argmax(envelope)  # Fallback to peak

        # Compute filtered frequency spectrum
        filtered_frequency = fft(filtered_signal)
        filtered_freqs = fftfreq(len(filtered_signal), 1/sampling_freq)

        return {
            'toa_time': hydrophone.times[toa_idx],
            'toa_idx': toa_idx,
            'filtered_signal': filtered_signal,
            'processed_signal': envelope,
            'filtered_frequency': filtered_frequency,
            'filtered_freqs': filtered_freqs,
            'threshold': threshold,
            'band_min': band_min,
            'band_max': band_max
        }

    def _plot_single_signal(self, ax_time, ax_freq, hydrophone, result, idx):
        """Plot envelope over filtered signal with TOA marker."""
        # Time domain plot
        ax_time.plot(
            hydrophone.times, result['filtered_signal'],
            alpha=0.5, label='Filtered Signal', color='blue'
        )
        ax_time.plot(
            hydrophone.times, result['processed_signal'],
            label='Envelope', color='darkblue', linewidth=2
        )
        ax_time.axhline(
            result['threshold'], color='green',
            linestyle=':', alpha=0.5, label='Threshold'
        )
        ax_time.axvline(
            result['toa_time'], color='red',
            linestyle='--', linewidth=2,
            label=f"TOA: {result['toa_time']:.6f}s"
        )

        # Frequency domain plot
        freq_mask = result['filtered_freqs'] >= 0
        freqs = result['filtered_freqs'][freq_mask]
        magnitude = np.abs(result['filtered_frequency'][freq_mask])

        ax_freq.plot(freqs, magnitude, label='Filtered Spectrum', color='blue')
        ax_freq.axvline(
            result['band_min'], color='red',
            linestyle='--', alpha=0.5, label='Filter Range'
        )
        ax_freq.axvline(
            result['band_max'], color='red',
            linestyle='--', alpha=0.5
        )
        ax_freq.set_xlim([0, 100000])  # Focus on relevant frequency range
