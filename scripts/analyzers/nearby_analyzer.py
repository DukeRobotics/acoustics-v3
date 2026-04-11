"""Nearby detection using trained Random Forest model on optimal 7 features."""
import numpy as np
import joblib
from pathlib import Path
from scipy.fft import fft, fftfreq
from scipy import signal as scipy_signal

from .base_analyzer import BaseAnalyzer


# Global model cache - loaded once on first instantiation
_ML_MODEL_CACHE = None
_ML_SCALER_CACHE = None
_ML_FEATURES_CACHE = None


def _load_ml_model():
    """Load the trained ML model and scaler (cached globally).
    
    Returns:
        Tuple of (model, scaler, features_list)
    """
    global _ML_MODEL_CACHE, _ML_SCALER_CACHE, _ML_FEATURES_CACHE
    
    if _ML_MODEL_CACHE is not None:
        return _ML_MODEL_CACHE, _ML_SCALER_CACHE, _ML_FEATURES_CACHE
    
    # Load from disk
    model_path = Path(__file__).parent.parent.parent / 'proximity_classifier_10ft_threshold.pkl'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model_pkg = joblib.load(str(model_path))
    _ML_MODEL_CACHE = model_pkg['model']
    _ML_SCALER_CACHE = model_pkg['scaler']
    _ML_FEATURES_CACHE = model_pkg['features']
    
    return _ML_MODEL_CACHE, _ML_SCALER_CACHE, _ML_FEATURES_CACHE


class NearbyAnalyzer(BaseAnalyzer):
    """Nearby presence detection using trained Random Forest model on 7 optimal features.
    
    This analyzer:
    1. Computes all 27+ features using FeatureAnalyzer
    2. Extracts the 7 optimal features identified during training
    3. Scales and runs through trained Random Forest model
    4. Returns nearby prediction + confidence score
    
    Model: Random Forest Classifier (10ft threshold)
    - Cross-validation accuracy: 94.62%
    - Full dataset accuracy: 100%
    - Decision boundary: nearby = <=10ft, far = >10ft
    """

    def __init__(self, **kwargs):
        """Initialize ML-based nearby analyzer.
        
        Loads the trained model and feature scaler on first instantiation (cached globally).
        
        Args:
            **kwargs: Additional arguments passed to BaseAnalyzer
        """
        super().__init__(**kwargs)
        
        # Load model and scaler (cached globally)
        try:
            self.model, self.scaler, self.optimal_features = _load_ml_model()
        except FileNotFoundError as e:
            raise RuntimeError(f"Failed to initialize ML model: {e}")

    def get_name(self):
        """Return analyzer name.
        
        Returns:
            String identifier for this analyzer
        """
        return "ML-based Nearby Analyzer (10ft threshold)"

    def print_results(self, analysis_results):
        """Print nearby detection results.
        
        Args:
            analysis_results: Dictionary returned from analyze_array
        """
        super().print_results(analysis_results)
        print(f"\nNearby Detection (ML model - 10ft threshold):")
        for result in analysis_results['results']:
            status = "NEARBY (≤10ft)" if result['nearby'] else "FAR (>10ft)"
            confidence = result.get('confidence', None)
            conf_str = f" [confidence: {confidence:.2%}]" if confidence is not None else ""
            print(f"  Hydrophone {result['hydrophone_idx']}: {status}{conf_str}")

    # ==================== FEATURE EXTRACTION METHODS ====================

    def _extract_temporal_features(self, envelope, sampling_freq, peak_idx):
        """Extract rise time, fall time, FWHM from envelope."""
        peak_val = envelope[peak_idx]
        
        # 10%, 50%, 90% thresholds for rise/fall
        thresh_10 = peak_val * 0.1
        thresh_50 = peak_val * 0.5
        
        # Rise time: search backwards from peak to find 10% crossing
        idx_10_rise = None
        for i in range(peak_idx, -1, -1):
            if envelope[i] < thresh_10:
                idx_10_rise = i
                break
        
        rise_time = ((peak_idx - idx_10_rise) / sampling_freq * 1000) if idx_10_rise is not None else None  # ms
        
        # Fall time: search forward from peak to find 10% crossing
        idx_10_fall = None
        for i in range(peak_idx, len(envelope)):
            if envelope[i] < thresh_10:
                idx_10_fall = i
                break
        
        fall_time = ((idx_10_fall - peak_idx) / sampling_freq * 1000) if idx_10_fall is not None else None  # ms
        
        # FWHM (Full Width Half Max at 50%): search backwards and forwards
        idx_50_before = None
        for i in range(peak_idx, -1, -1):
            if envelope[i] < thresh_50:
                idx_50_before = i
                break
        
        idx_50_after = None
        for i in range(peak_idx, len(envelope)):
            if envelope[i] < thresh_50 and i > peak_idx:
                idx_50_after = i
                break
        
        fwhm = ((idx_50_after - idx_50_before) / sampling_freq * 1000) if (idx_50_before is not None and idx_50_after is not None) else None  # ms
        
        # Pulse width (10% to 10%): from rise to fall
        pulse_width = ((idx_10_fall - idx_10_rise) / sampling_freq * 1000) if (idx_10_rise is not None and idx_10_fall is not None) else None  # ms
        
        return {
            'rise_time_ms': rise_time,
            'fall_time_ms': fall_time,
            'fwhm_ms': fwhm,
            'pulse_width_ms': pulse_width,
        }

    def _extract_spectral_features(self, signal, freqs):
        """Extract peak frequency, centroid, bandwidth, flatness."""
        spectrum = np.abs(fft(signal))
        freqs = np.abs(freqs)
        
        # Only positive frequencies
        pos_mask = freqs >= 0
        freqs_pos = freqs[pos_mask]
        spectrum_pos = spectrum[pos_mask]
        
        # Peak frequency
        peak_idx = np.argmax(spectrum_pos)
        peak_freq = freqs_pos[peak_idx]
        
        # Spectral centroid
        centroid = np.sum(freqs_pos * spectrum_pos) / (np.sum(spectrum_pos) + 1e-10)
        
        # Spectral flatness (Wiener entropy)
        geom_mean = np.exp(np.mean(np.log(spectrum_pos + 1e-10)))
        arith_mean = np.mean(spectrum_pos)
        flatness = geom_mean / (arith_mean + 1e-10)
        
        return {
            'peak_frequency_hz': peak_freq,
            'spectral_centroid_hz': centroid,
            'spectral_flatness': flatness,
        }

    def _extract_multipath_features(self, envelope, sampling_freq, peak_idx, pulse_width_ms):
        """Detect secondary peaks (reflections)."""
        if pulse_width_ms is None:
            pulse_width_s = 0.05
        else:
            pulse_width_s = pulse_width_ms / 1000
        
        # Search for peaks in late window for multipath
        late_start_idx = int(peak_idx + pulse_width_s * 1.5 * sampling_freq)
        late_end_idx = int(peak_idx + pulse_width_s * 4 * sampling_freq)
        late_end_idx = min(late_end_idx, len(envelope))
        
        if late_start_idx >= late_end_idx:
            return {
                'secondary_peak_count': 0,
                'secondary_peak_amplitude_ratio': 0,
                'time_to_secondary_peak_ms': 0,
            }
        
        late_region = envelope[late_start_idx:late_end_idx]
        peaks, properties = scipy_signal.find_peaks(late_region, height=np.max(late_region) * 0.1)
        
        secondary_count = len(peaks)
        
        if secondary_count > 0:
            # Get largest secondary peak
            largest_idx = np.argmax(properties['peak_heights'])
            secondary_peak_height = properties['peak_heights'][largest_idx]
            secondary_peak_amp_ratio = secondary_peak_height / (np.max(envelope) + 1e-10)
            time_to_secondary = (late_start_idx + peaks[largest_idx]) / sampling_freq - (peak_idx / sampling_freq)
            time_to_secondary_ms = time_to_secondary * 1000
        else:
            secondary_peak_amp_ratio = 0
            time_to_secondary_ms = 0
        
        return {
            'secondary_peak_count': secondary_count,
            'secondary_peak_amplitude_ratio': secondary_peak_amp_ratio,
            'time_to_secondary_peak_ms': time_to_secondary_ms,
        }

    def _analyze_single(self, hydrophone, sampling_freq):
        """Analyze single hydrophone using trained ML model.
        
        Args:
            hydrophone: Hydrophone object with signal data
            sampling_freq: Sampling frequency in Hz
            
        Returns:
            Dictionary containing:
                - nearby: Boolean indicating if source is nearby (≤10ft)
                - confidence: Model confidence score (probability)
                - optimal_features_values: Dictionary of extracted feature values
        """
        # Get signals
        raw_signal = hydrophone.signal
        filtered_signal = self.apply_bandpass(raw_signal, sampling_freq)
        
        # Compute envelopes using Hilbert transform
        raw_envelope = np.abs(scipy_signal.hilbert(raw_signal))
        filtered_envelope = np.abs(scipy_signal.hilbert(filtered_signal))
        
        # Find peaks
        peak_idx_filtered = np.argmax(filtered_envelope)
        peak_idx_raw = np.argmax(raw_envelope)
        
        # Extract features for FILTERED signal
        filt_temporal = self._extract_temporal_features(filtered_envelope, sampling_freq, peak_idx_filtered)
        filtered_freq = fftfreq(len(filtered_signal), 1/sampling_freq)
        filt_spectral = self._extract_spectral_features(filtered_signal, filtered_freq)
        filt_multipath = self._extract_multipath_features(filtered_envelope, sampling_freq, peak_idx_filtered, 
                                                          filt_temporal['pulse_width_ms'])
        
        # Extract features for RAW signal
        raw_temporal = self._extract_temporal_features(raw_envelope, sampling_freq, peak_idx_raw)
        raw_freq = fftfreq(len(raw_signal), 1/sampling_freq)
        raw_spectral = self._extract_spectral_features(raw_signal, raw_freq)
        
        # Build feature dictionary with H0_ prefix
        all_features = {}
        all_features.update({f"H0_FILTERED_{k}": v for k, v in filt_temporal.items()})
        all_features.update({f"H0_FILTERED_{k}": v for k, v in filt_spectral.items()})
        all_features.update({f"H0_FILTERED_{k}": v for k, v in filt_multipath.items()})
        all_features.update({f"H0_RAW_{k}": v for k, v in raw_temporal.items()})
        all_features.update({f"H0_RAW_{k}": v for k, v in raw_spectral.items()})
        
        # Step 2: Extract only the 7 optimal features
        feature_values = []
        feature_dict = {}
        
        for feat_name in self.optimal_features:
            if feat_name not in all_features:
                raise ValueError(f"Feature '{feat_name}' not found in extracted features. "
                               f"Available: {list(all_features.keys())}")
            value = all_features[feat_name]
            feature_values.append(value)
            feature_dict[feat_name] = value
        
        # Step 3: Scale features (must use same scaler as training)
        X_scaled = self.scaler.transform([feature_values])
        
        # Step 4: Get prediction and confidence
        nearby_pred = self.model.predict(X_scaled)[0]
        confidence = np.max(self.model.predict_proba(X_scaled)[0])
        
        # Convert prediction (0/1) to boolean
        nearby = bool(nearby_pred == 1)
        
        return {
            'nearby': nearby,
            'confidence': confidence,
            'optimal_features_values': feature_dict,
        }

    def _plot_single_signal(self, ax_time, ax_freq, hydrophone, result, idx):
        """Plot nearby detection results (text-based since this is ML-based).
        
        Args:
            ax_time: Matplotlib axis for time domain plot
            ax_freq: Matplotlib axis for frequency domain plot
            hydrophone: Hydrophone object with signal data
            result: Analysis result dictionary from _analyze_single
            idx: Hydrophone index
        """
        status = "NEARBY (≤10ft)" if result['nearby'] else "FAR (>10ft)"
        confidence = result.get('confidence', 0)
        
        ax_time.text(0.5, 0.7, f"Hydrophone {idx}: {status}", 
                     ha='center', va='center', transform=ax_time.transAxes,
                     fontsize=12, fontweight='bold')
        ax_time.text(0.5, 0.5, f"Confidence: {confidence:.2%}", 
                     ha='center', va='center', transform=ax_time.transAxes,
                     fontsize=10)
        ax_time.text(0.5, 0.3, "ML Model: Random Forest", 
                     ha='center', va='center', transform=ax_time.transAxes,
                     fontsize=9, style='italic')
        ax_time.axis('off')
        ax_freq.axis('off')
