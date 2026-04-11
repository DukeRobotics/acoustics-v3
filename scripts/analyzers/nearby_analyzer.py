"""ML-based nearby detection using H0 features."""
import warnings
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.fft import fft, fftfreq
from scipy import signal as scipy_signal

from .base_analyzer import BaseAnalyzer

warnings.filterwarnings('ignore', message='.*does not have valid feature names.*')

_ML_MODEL = None
_ML_SCALER = None
_ML_FEATURES = None


def _load_model():
    global _ML_MODEL, _ML_SCALER, _ML_FEATURES
    if _ML_MODEL is not None:
        return _ML_MODEL, _ML_SCALER, _ML_FEATURES
    
    path = Path(__file__).parent.parent.parent / 'proximity_classifier_10ft_threshold.pkl'
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    
    pkg = joblib.load(str(path))
    _ML_MODEL = pkg['model']
    _ML_SCALER = pkg['scaler']
    _ML_FEATURES = pkg['features']
    return _ML_MODEL, _ML_SCALER, _ML_FEATURES


class NearbyAnalyzer(BaseAnalyzer):
    """ML-based nearby detection (≤10ft) on H0 features."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model, self.scaler, self.optimal_features = _load_model()

    def get_name(self):
        return "ML-based Nearby Analyzer (10ft threshold)"

    def _extract_features(self, signal, sampling_freq):
        """Extract all features from signal."""
        filtered = self.apply_bandpass(signal, sampling_freq)
        
        raw_env = np.abs(scipy_signal.hilbert(signal))
        filt_env = np.abs(scipy_signal.hilbert(filtered))
        
        peak_raw = np.argmax(raw_env)
        peak_filt = np.argmax(filt_env)
        
        filt_temp = self._temporal(filt_env, sampling_freq, peak_filt)
        raw_temp = self._temporal(raw_env, sampling_freq, peak_raw)
        filt_spec = self._spectral(filtered, sampling_freq)
        raw_spec = self._spectral(signal, sampling_freq)
        filt_multi = self._multipath(filt_env, sampling_freq, peak_filt, filt_temp['pulse_width_ms'])
        
        feats = {}
        for k, v in filt_temp.items():
            feats[f"H0_FILTERED_{k}"] = v
        for k, v in filt_spec.items():
            feats[f"H0_FILTERED_{k}"] = v
        for k, v in filt_multi.items():
            feats[f"H0_FILTERED_{k}"] = v
        for k, v in raw_temp.items():
            feats[f"H0_RAW_{k}"] = v
        for k, v in raw_spec.items():
            feats[f"H0_RAW_{k}"] = v
        
        return feats

    def _temporal(self, env, fs, peak):
        peak_val = env[peak]
        thresh_10 = peak_val * 0.1
        thresh_50 = peak_val * 0.5
        
        i_10_rise = next((i for i in range(peak, -1, -1) if env[i] < thresh_10), peak)
        i_10_fall = next((i for i in range(peak, len(env)) if env[i] < thresh_10), peak)
        i_50_before = next((i for i in range(peak, -1, -1) if env[i] < thresh_50), 0)
        i_50_after = next((i for i in range(peak, len(env)) if env[i] < thresh_50 and i > peak), len(env)-1)
        
        return {
            'rise_time_ms': (peak - i_10_rise) / fs * 1000,
            'fall_time_ms': (i_10_fall - peak) / fs * 1000,
            'fwhm_ms': (i_50_after - i_50_before) / fs * 1000,
            'pulse_width_ms': (i_10_fall - i_10_rise) / fs * 1000,
        }

    def _spectral(self, sig, fs):
        spec = np.abs(fft(sig))
        freqs = fftfreq(len(sig), 1/fs)
        pos = freqs >= 0
        f_pos = freqs[pos]
        s_pos = spec[pos]
        
        peak_f = f_pos[np.argmax(s_pos)]
        centroid = np.sum(f_pos * s_pos) / (np.sum(s_pos) + 1e-10)
        geom = np.exp(np.mean(np.log(s_pos + 1e-10)))
        arith = np.mean(s_pos)
        flatness = geom / (arith + 1e-10)
        
        return {
            'spectral_centroid_hz': centroid,
            'spectral_flatness': flatness,
            'peak_frequency_hz': peak_f,
        }

    def _multipath(self, env, fs, peak, pw_ms):
        pw_s = (pw_ms if pw_ms else 0.05) / 1000
        late_start = int(peak + pw_s * 1.5 * fs)
        late_end = min(int(peak + pw_s * 4 * fs), len(env))
        
        if late_start >= late_end:
            return {'time_to_secondary_peak_ms': 0}
        
        late = env[late_start:late_end]
        peaks, props = scipy_signal.find_peaks(late, height=np.max(late) * 0.1)
        
        if len(peaks) == 0:
            return {'time_to_secondary_peak_ms': 0}
        
        t_to_sec = (late_start + peaks[0] - peak) / fs * 1000
        return {'time_to_secondary_peak_ms': t_to_sec}

    def _analyze_single(self, hydrophone, sampling_freq):
        """Not used - we override analyze_array."""
        return {}

    def _plot_single_signal(self, ax_time, ax_freq, hydrophone, result, idx):
        """Not implemented."""
        pass

    def analyze_array(self, hydrophone_array, selected=None):
        """Extract H0 features and run model, return for all hydrophones."""
        if selected is None:
            selected = hydrophone_array.selected
        
        # Extract features from H0 only
        h0 = hydrophone_array.hydrophones[0]
        fs = 1 / h0.sampling_period
        all_feats = self._extract_features(h0.signal, fs)
        
        # Get 7 optimal features
        feat_vec = [all_feats[f] for f in self.optimal_features]
        feat_dict = {f: all_feats[f] for f in self.optimal_features}
        
        # Predict
        X_df = pd.DataFrame([feat_vec], columns=self.optimal_features)
        X_scaled = self.scaler.transform(X_df)
        is_nearby = bool(self.model.predict(X_scaled)[0])
        confidence = float(np.max(self.model.predict_proba(X_scaled)[0]))
        
        # Return for all selected hydrophones
        results = []
        for idx, is_sel in enumerate(selected):
            if is_sel:
                results.append({
                    'hydrophone_idx': idx,
                    'nearby': is_nearby,
                    'confidence': confidence,
                    'optimal_features_values': feat_dict,
                })
        
        return {'results': results, 'analyzer': self.get_name()}
