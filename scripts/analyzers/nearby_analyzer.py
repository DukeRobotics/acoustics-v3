"""Feature extraction for all 27 acoustic features."""
from .base_analyzer import BaseAnalyzer
from .feature_analyzer import FeatureAnalyzer


class NearbyAnalyzer(BaseAnalyzer):
    """Extracts all 27 acoustic features using FeatureAnalyzer."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.feature_analyzer = FeatureAnalyzer(**kwargs)

    def get_name(self):
        return "Nearby Analyzer (all features)"

    def print_results(self, analysis_results):
        super().print_results(analysis_results)
        print(f"\nExtracted Features:")
        for result in analysis_results['results']:
            print(f"  Hydrophone {result['hydrophone_idx']}:")
            for key, val in result.items():
                if key != 'hydrophone_idx':
                    print(f"    {key}: {val}")

    def _analyze_single(self, hydrophone, sampling_freq):
        """Extract all 27 features for a single hydrophone.

        Args:
            hydrophone: Hydrophone object with signal data
            sampling_freq: Sampling frequency in Hz

        Returns:
            Dictionary of all 27 extracted feature values
        """
        return self.feature_analyzer._analyze_single(hydrophone, sampling_freq)

    def _plot_single_signal(self, ax_time, ax_freq, hydrophone, result, idx):
        n_features = len(result)
        ax_time.text(0.5, 0.5, f"Hydrophone {idx}\n{n_features} features extracted",
                     ha='center', va='center', transform=ax_time.transAxes,
                     fontsize=11)
        ax_time.axis('off')
        ax_freq.axis('off')
