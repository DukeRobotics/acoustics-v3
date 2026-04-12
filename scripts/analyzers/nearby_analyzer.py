

import joblib
import numpy as np
from scipy.signal import hilbert
from scipy.fft import fft, fftfreq
import scipy.signal as sp_signal
import pandas as pd
from .base_analyzer import BaseAnalyzer


class NearbyAnalyzer(BaseAnalyzer):
    """ML-based nearby detection (≤10ft) using 4 key features."""

    def __init__(self, model_path: str, **kwargs):
        super().__init__(**kwargs)
        pkg = joblib.load(model_path)
        self.model = pkg['model']
        self.features = pkg['features']  # Feature names in order
        print(f"Model features: {self.features}")

    def get_name(self):
        return "ML-based Nearby Detection (10ft)"

    def _analyze_single(self, hydrophone, sampling_freq):
        """Extract 4 features and predict if nearby."""
        signal = hydrophone.signal
        filtered = self.apply_bandpass(signal, sampling_freq)
        
        # Compute envelopes
        env_raw = np.abs(hilbert(signal))
        env_filt = np.abs(hilbert(filtered))
        peak_raw = np.argmax(env_raw)
        peak_filt = np.argmax(env_filt)
        
        # Extract 4 features
        flatness = self._spectral_flatness(signal)
        centroid = self._spectral_centroid(filtered, sampling_freq)
        rise_time = self._rise_time(env_raw, sampling_freq, peak_raw)
        sec_peak = self._secondary_peak(env_filt, sampling_freq, peak_filt, rise_time)
        
        # Prepare for model
        X = pd.DataFrame(
            [[flatness, centroid, sec_peak, rise_time]], 
            columns=self.features
        )
        pred = self.model.predict(X)[0]
        prob = np.max(self.model.predict_proba(X)[0])
        
        return {
            'is_nearby': bool(pred),
            'confidence': float(prob),
            'feature_values': {
                'RAW_spectral_flatness': flatness,
                'FILTERED_spectral_centroid_hz': centroid,
                'FILTERED_time_to_secondary_peak_ms': sec_peak,
                'RAW_rise_time_ms': rise_time,
            }
        }

    def _spectral_flatness(self, sig):
        spec = np.abs(fft(sig))
        geom = np.exp(np.mean(np.log(spec + 1e-10)))
        arith = np.mean(spec)
        return geom / (arith + 1e-10)

    def _spectral_centroid(self, sig, fs):
        spec = np.abs(fft(sig))
        freqs = fftfreq(len(sig), 1/fs)
        pos = freqs >= 0
        return np.sum(freqs[pos] * spec[pos]) / (np.sum(spec[pos]) + 1e-10)

    def _rise_time(self, env, fs, peak):
        thresh = env[peak] * 0.1
        i_start = next((i for i in range(peak, -1, -1) if env[i] < thresh), peak)
        return (peak - i_start) / fs * 1000

    def _secondary_peak(self, env, fs, peak, rise_ms):
        pw = rise_ms / 1000
        start = int(peak + pw * 1.5 * fs)
        end = min(int(peak + pw * 4 * fs), len(env))
        
        if start >= end:
            return 0
        
        peaks, _ = sp_signal.find_peaks(env[start:end], height=np.max(env[start:end]) * 0.1)
        if len(peaks) == 0:
            return 0
        
        return (start + peaks[0] - peak) / fs * 1000

    def _plot_single_signal(self, ax_time, ax_freq, hydrophone, result, idx):
        """Plot filtered signal with prediction."""
        signal = hydrophone.signal
        filtered = self.apply_bandpass(signal, 1 / hydrophone.sampling_period)
        ax_time.plot(filtered)
        nearby_label = "NEARBY" if result['is_nearby'] else "FAR"
        ax_time.set_title(f"H{idx} - {nearby_label} ({result['confidence']:.1%})")

    def print_results(self, analysis_results):
        """Print prediction for each hydrophone."""
        print(f"\n{analysis_results['analyzer']}")
        for r in analysis_results['results']:
            label = "NEARBY" if r['is_nearby'] else "FAR"
            print(f"  H{r.get('hydrophone_idx')}: {label} ({r['confidence']:.1%})")
