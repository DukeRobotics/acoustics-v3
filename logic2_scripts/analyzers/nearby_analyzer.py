from .base_analyzer import BaseAnalyzer
import numpy as np
from scipy.fft import fft, fftfreq

class NearbyAnalyzer(BaseAnalyzer):
    """TOA estimation using Hilbert envelope detection."""

    def __init__(self, threshold, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold

    def get_name(self):
        return "Static Nearby Analyzer"
    
    def print_results(self, analysis_results):
        """Print nearby detection results."""
        super().print_results(analysis_results)
        print(f"\nNearby Detection (threshold: {self.threshold}):")
        for result in analysis_results['results']:
            status = "NEARBY" if result['nearby'] else "NOT NEARBY"
            print(f"  Hydrophone {result['hydrophone_idx']}: {status}")
    
    def _analyze_single(self, hydrophone, sampling_freq, center_freq):
        """Analyze one hydrophone using static threshold."""
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

        # Detect TOA using threshold
        toa_candidates = np.where(filtered_signal > self.threshold)[0]
        nearby = len(toa_candidates) > 0

        # Compute filtered frequency spectrum
        filtered_frequency = fft(filtered_signal)
        filtered_freqs = fftfreq(len(filtered_signal), 1/sampling_freq)
        
        return {
            'nearby': nearby,
            'filtered_signal': filtered_signal,
            'filtered_frequency': filtered_frequency,
            'filtered_freqs': filtered_freqs,
            'threshold': self.threshold,
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
        ax_time.axhline(
            result['threshold'], color='green',
            linestyle=':', alpha=0.5, label='Threshold'
        )
        
        # Indicate if nearby
        status = 'NEARBY' if result['nearby'] else 'NOT NEARBY'
        color = 'green' if result['nearby'] else 'red'
        ax_time.text(
            0.5, 0.95, status,
            transform=ax_time.transAxes,
            fontsize=12, fontweight='bold',
            color=color, ha='center', va='top'
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
