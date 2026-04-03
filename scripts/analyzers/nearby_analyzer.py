"""Nearby detection using ping width analysis."""
import numpy as np
from scipy.fft import fft, fftfreq

from .base_analyzer import BaseAnalyzer


class NearbyAnalyzer(BaseAnalyzer):
    """Nearby presence detection using ping width analysis.
    
    This analyzer determines if a signal source is nearby by measuring the
    ping width (time between first and second threshold crossings at parameterized std).
    """

    def __init__(self, ping_width_threshold=0.01, crossing_std_dev=5, **kwargs):
        """Initialize nearby analyzer.
        
        Args:
            ping_width_threshold: Ping width threshold in seconds (<=threshold = nearby)
            crossing_std_dev: Std deviations above mean for threshold crossings
            **kwargs: Additional arguments passed to BaseAnalyzer
        """
        super().__init__(**kwargs)
        self.ping_width_threshold = ping_width_threshold
        self.crossing_std_dev = crossing_std_dev

    def get_name(self):
        """Return analyzer name.
        
        Returns:
            String identifier for this analyzer
        """
        return "Ping Width Nearby Analyzer"

    def print_results(self, analysis_results):
        """Print nearby detection results.
        
        Args:
            analysis_results: Dictionary returned from analyze_array
        """
        super().print_results(analysis_results)
        print(f"\nNearby Detection (ping width threshold: {self.ping_width_threshold}s):")
        for result in analysis_results['results']:
            status = "NEARBY" if result['nearby'] else "NOT NEARBY"
            delta_t = result.get('delta_t', None)
            delta_t_str = f" (delta_t: {delta_t:.6f}s)" if delta_t is not None else ""
            print(f"  Hydrophone {result['hydrophone_idx']}: {status}{delta_t_str}")

    def _analyze_single(self, hydrophone, sampling_freq):
        """Analyze single hydrophone using ping width detection.
        
        Args:
            hydrophone: Hydrophone object with signal data
            sampling_freq: Sampling frequency in Hz
            
        Returns:
            Dictionary containing:
                - nearby: Boolean indicating if ping width <= threshold
                - delta_t: Ping width in seconds (time between threshold crossings), or None if less than 2 crossings
                - filtered_signal: Bandpass filtered signal
                - filtered_frequency: FFT of filtered signal
                - filtered_freqs: Frequency bins for FFT
                - band_min: Lower frequency bound used (Hz)
                - band_max: Upper frequency bound used (Hz)
        """
        # Apply bandpass filter
        filtered_signal = self.apply_bandpass(
            hydrophone.signal, sampling_freq
        )

        # Compute envelope (absolute value)
        envelope = np.abs(filtered_signal)
        
        # Calculate threshold at parameterized std deviations above mean
        threshold = np.mean(envelope) + self.crossing_std_dev * np.std(envelope)
        
        # Find crossings above threshold
        crossings = np.where(envelope > threshold)[0]
        
        # Calculate ping width (delta_t)
        if len(crossings) >= 2:
            first_crossing = crossings[0]
            second_crossing = crossings[-1]
            delta_t = (second_crossing - first_crossing) / sampling_freq
        else:
            delta_t = None
        
        # Determine if nearby based on ping width threshold
        nearby = delta_t <= self.ping_width_threshold if delta_t is not None else False

        # Compute filtered frequency spectrum
        filtered_frequency = fft(filtered_signal)
        filtered_freqs = fftfreq(len(filtered_signal), 1/sampling_freq)

        return {
            'nearby': nearby,
            'delta_t': delta_t,
            'filtered_signal': filtered_signal,
            'filtered_frequency': filtered_frequency,
            'filtered_freqs': filtered_freqs,
            'threshold': threshold,
            'band_min': self.search_band_min,
            'band_max': self.search_band_max
        }

    def _plot_single_signal(self, ax_time, ax_freq, hydrophone, result, idx):
        """Plot nearby detection results for a single hydrophone.
        
        Args:
            ax_time: Matplotlib axis for time domain plot
            ax_freq: Matplotlib axis for frequency domain plot
            hydrophone: Hydrophone object with signal data
            result: Analysis result dictionary from _analyze_single
            idx: Hydrophone index
        """
        # Time domain plot
        envelope = np.abs(result['filtered_signal'])
        ax_time.plot(
            hydrophone.times, envelope,
            alpha=0.5, label='Envelope', color='blue'
        )
        ax_time.axhline(
            result['threshold'], color='green',
            linestyle=':', alpha=0.5, label='5 Std Threshold'
        )

        # Indicate if nearby with delta_t
        status = 'NEARBY' if result['nearby'] else 'NOT NEARBY'
        color = 'green' if result['nearby'] else 'red'
        delta_t_text = f"{status}\ndelta_t: {result['delta_t']:.6f}s"
        ax_time.text(
            0.5, 0.95, delta_t_text,
            transform=ax_time.transAxes,
            fontsize=10, fontweight='bold',
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
