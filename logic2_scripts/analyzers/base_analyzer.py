"""Base analyzer module for hydrophone signal processing."""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt


class BaseAnalyzer(ABC):
    """Base class for hydrophone signal analyzers."""

    def __init__(
        self,
        search_band_min: float = 25000,
        search_band_max: float = 40000,
        use_narrow_band: bool = True,
        narrow_band_width: float = 100,
        filter_order: int = 8,
        plot_results: bool = False,
        config: dict | None = None
    ):
        self.search_band_min = search_band_min
        self.search_band_max = search_band_max
        self.use_narrow_band = use_narrow_band
        self.narrow_band_width = narrow_band_width
        self.filter_order = filter_order
        self.plot_results_flag = plot_results
        self.config = config or {}

    # ==================== ABSTRACT METHODS ====================

    @abstractmethod
    def _analyze_single(self, hydrophone, sampling_freq, center_freq) -> dict:
        """
        Analyze ONE hydrophone. Return dict with keys:
            - toa_time: float
            - toa_idx: int
            - filtered_signal: np.ndarray
            - processed_signal: np.ndarray (e.g., envelope)
            - (any other analyzer-specific data)
        """

    @abstractmethod
    def _plot_single_signal(self, ax_time, ax_freq, hydrophone, result, idx):
        """Plot ONE hydrophone's analysis on given axes (time and frequency)."""

    @abstractmethod
    def get_name(self) -> str:
        """Return analyzer name."""

    # ==================== PUBLIC ====================

    def analyze_array(self, hydrophone_array, selected: list[bool] | None = None):
        """Analyze all selected hydrophones. Returns dict with results."""
        if selected is None:
            selected = hydrophone_array.selected

        # Find center frequency if needed
        center_freq = None
        if self.use_narrow_band:
            center_freq = self._find_center_frequency(hydrophone_array, selected)

        # Analyze each hydrophone
        results = []
        for idx, (hydro, is_selected) in enumerate(
            zip(hydrophone_array.hydrophones, selected)
        ):
            if is_selected:
                result = self._analyze_single(
                    hydro, hydrophone_array.sampling_freq, center_freq
                )
                result['hydrophone_idx'] = idx
                results.append(result)

        analysis_results = {
            'results': results,
            'center_frequency': center_freq,
            'analyzer': self.get_name()
        }

        if self.plot_results_flag:
            self.plot_results(hydrophone_array, analysis_results, selected)

        return analysis_results

    def print_results(self, analysis_results):
        """Print analysis results. Can be overridden by subclasses."""
        print(f"\n{analysis_results['analyzer']}")
        print(f"Center Frequency: {analysis_results.get('center_frequency', 'N/A'):.2f} Hz" 
              if analysis_results.get('center_frequency') else "Center Frequency: N/A")

    def plot_results(self, hydrophone_array, analysis_results, selected=None):
        """Plot all analyzed hydrophones with filtered signal and frequency."""
        if selected is None:
            selected = hydrophone_array.selected

        results = analysis_results['results']
        num_plots = len(results)

        # Create 2 columns: time domain and frequency domain
        _, axes = plt.subplots(
            num_plots, 2, figsize=(14, 3*num_plots), squeeze=False
        )

        for plot_idx, result in enumerate(results):
            hydro_idx = result['hydrophone_idx']
            hydro = hydrophone_array.hydrophones[hydro_idx]

            # Each analyzer defines how to plot ONE signal
            self._plot_single_signal(
                axes[plot_idx, 0], axes[plot_idx, 1],
                hydro, result, hydro_idx
            )

            # Common formatting
            axes[plot_idx, 0].set_ylabel('Amplitude')
            axes[plot_idx, 0].set_title(f'Hydrophone {hydro_idx} - Time Domain')
            axes[plot_idx, 0].legend(loc='best')
            axes[plot_idx, 0].grid(True, alpha=0.3)

            axes[plot_idx, 1].set_ylabel('Magnitude')
            axes[plot_idx, 1].set_title(f'Hydrophone {hydro_idx} - Frequency Domain')
            axes[plot_idx, 1].legend(loc='best')
            axes[plot_idx, 1].grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel('Time (s)')
        axes[-1, 1].set_xlabel('Frequency (Hz)')
        plt.tight_layout()
        plt.show()

    # ==================== COMMON ====================

    def apply_bandpass(self, signal, sampling_freq, band_min=None, band_max=None):
        """Apply bandpass filter to signal."""
        if band_min is None:
            band_min = self.search_band_min
        if band_max is None:
            band_max = self.search_band_max

        sos = butter(
            self.filter_order,
            [band_min, band_max],
            fs=sampling_freq,
            btype='band',
            output='sos'
        )
        return sosfilt(sos, signal)

    def _find_center_frequency(self, hydrophone_array, selected):
        """Find center frequency from peaks."""
        peak_freqs = []
        for hydro, is_selected in zip(hydrophone_array.hydrophones, selected):
            if is_selected and hydro.freqs is not None:
                freq_mask = (
                    (hydro.freqs >= self.search_band_min) &
                    (hydro.freqs <= self.search_band_max)
                )
                band_freqs = hydro.freqs[freq_mask]
                band_magnitude = np.abs(hydro.frequency[freq_mask])
                peak_idx = np.argmax(band_magnitude)
                peak_freqs.append(band_freqs[peak_idx])
        return float(np.median(peak_freqs))

    def _compute_relative_times(self, results, reference_hydrophone):
        """Compute TOA relative to reference hydrophone."""
        ref_toa = results[reference_hydrophone]['toa_time']
        return [r['toa_time'] - ref_toa for r in results]
