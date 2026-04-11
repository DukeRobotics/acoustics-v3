"""Feature extraction analyzer for signal characterization."""
import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal as scipy_signal

from .base_analyzer import BaseAnalyzer


class FeatureAnalyzer(BaseAnalyzer):
    """Extract 27+ features characterizing ping signal properties.
    
    Features are computed for both RAW and FILTERED signals across all hydrophones.
    Includes temporal, spectral, energy, and multipath detection features.
    """

    def __init__(self, **kwargs):
        """Initialize feature analyzer."""
        super().__init__(**kwargs)

    def get_name(self):
        return "Feature Extraction Analyzer"

    def print_results(self, analysis_results):
        """Print feature extraction results."""
        super().print_results(analysis_results)
        print("\nFeature Extraction Complete")
        for result in analysis_results['results']:
            print(f"  Hydrophone {result['hydrophone_idx']}: Extracted {len([k for k in result.keys() if k.startswith('RAW_') or k.startswith('FILTERED_')])} features")

    def _find_pulse_end(self, envelope, peak_idx, sampling_freq, threshold_percent=0.2):
        """Find where pulse ends by detecting decay below threshold.
        
        Args:
            envelope: Magnitude envelope (post-peak)
            peak_idx: Index of peak
            sampling_freq: Sampling frequency
            threshold_percent: Consider pulse ended when envelope drops below this % of peak
            
        Returns:
            Time of pulse end in seconds, or None if not found
        """
        peak_val = envelope[peak_idx]
        threshold = peak_val * threshold_percent
        
        # Search forward from peak for decay
        for i in range(peak_idx, len(envelope)):
            if envelope[i] < threshold:
                return i / sampling_freq
        
        return None

    def _extract_temporal_features(self, envelope, sampling_freq, peak_idx):
        """Extract rise time, fall time, FWHM, etc."""
        peak_val = envelope[peak_idx]
        
        # 10%, 50%, 90% thresholds for rise/fall
        thresh_10 = peak_val * 0.1
        thresh_50 = peak_val * 0.5
        thresh_90 = peak_val * 0.9
        
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

    def _extract_energy_features(self, envelope, sampling_freq, peak_idx, pulse_width_ms):
        """Extract early/late energy, ratios."""
        if pulse_width_ms is None:
            pulse_width_s = 0.05  # fallback 50ms
        else:
            pulse_width_s = pulse_width_ms / 1000
        
        total_energy = np.sum(envelope ** 2)
        
        # Adaptive windows based on pulse width
        early_end_idx = int(peak_idx + pulse_width_s * 1.5 * sampling_freq)
        late_start_idx = early_end_idx
        late_end_idx = int(peak_idx + pulse_width_s * 4 * sampling_freq)
        
        # Clamp indices
        early_end_idx = min(early_end_idx, len(envelope))
        late_start_idx = min(late_start_idx, len(envelope))
        late_end_idx = min(late_end_idx, len(envelope))
        
        early_energy = np.sum(envelope[peak_idx:early_end_idx] ** 2)
        late_energy = np.sum(envelope[late_start_idx:late_end_idx] ** 2)
        
        early_late_ratio = early_energy / (late_energy + 1e-10)  # avoid division by zero
        energy_concentration = (early_energy / (total_energy + 1e-10)) * 100  # percentage
        
        return {
            'total_energy': total_energy,
            'early_window_energy': early_energy,
            'late_window_energy': late_energy,
            'early_late_ratio': early_late_ratio,
            'energy_concentration_pct': energy_concentration,
        }

    def _extract_amplitude_features(self, envelope, signal):
        """Extract peak amplitude, dynamic range, crest factor.
        
        Args:
            envelope: Magnitude envelope of signal
            signal: Corresponding signal (raw or filtered) for noise estimation
        """
        peak_amplitude = np.max(envelope)
        
        # Estimate noise from pre-signal region (first 10% of signal)
        noise_est = np.std(signal[:len(signal)//10])
        dynamic_range = peak_amplitude / (noise_est + 1e-10)
        
        rms_signal = np.sqrt(np.mean(signal ** 2))
        crest_factor = peak_amplitude / (rms_signal + 1e-10)
        
        return {
            'peak_amplitude': peak_amplitude,
            'dynamic_range': dynamic_range,
            'crest_factor': crest_factor,
        }

    def _extract_snr_features(self, envelope, signal, peak_idx, sampling_freq, pulse_width_ms):
        """Extract SNR, peak SNR, noise level.
        
        Args:
            envelope: Magnitude envelope of signal
            signal: Corresponding signal (raw or filtered) for SNR computation
            peak_idx: Index of signal peak
            sampling_freq: Sampling frequency in Hz
            pulse_width_ms: Pulse width in milliseconds
        """
        if pulse_width_ms is None:
            pulse_width_s = 0.05
        else:
            pulse_width_s = pulse_width_ms / 1000
        
        # Noise estimate from pre-signal region
        noise_start_idx = max(0, int(peak_idx - 0.5 * sampling_freq))  # 500ms before
        noise_region = signal[noise_start_idx:peak_idx]
        noise_power = np.mean(noise_region ** 2) if len(noise_region) > 0 else 1e-10
        noise_std = np.std(noise_region) if len(noise_region) > 0 else 1e-10
        
        # Signal power in early window
        end_idx = int(peak_idx + pulse_width_s * 1.5 * sampling_freq)
        end_idx = min(end_idx, len(signal))
        signal_region = signal[peak_idx:end_idx]
        signal_power = np.mean(signal_region ** 2)
        
        snr_db = 10 * np.log10((signal_power + 1e-10) / (noise_power + 1e-10))
        peak_snr = 20 * np.log10(np.max(np.abs(signal_region)) / (noise_std + 1e-10))
        
        return {
            'snr_db': snr_db,
            'peak_snr_db': peak_snr,
            'noise_std': noise_std,
        }

    def _extract_spectral_features(self, signal, freqs):
        """Extract peak frequency, centroid, bandwidth, flatness.
        
        Args:
            signal: Input signal (raw or filtered)
            freqs: Frequency bins from fftfreq
        """
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
        
        # Bandwidth (-3dB)
        peak_power = spectrum_pos[peak_idx]
        threshold = peak_power / np.sqrt(2)
        bw_mask = spectrum_pos > threshold
        if np.any(bw_mask):
            bw_freqs = freqs_pos[bw_mask]
            bandwidth = np.max(bw_freqs) - np.min(bw_freqs)
        else:
            bandwidth = 0
        
        # Spectral flatness (Wiener entropy)
        geom_mean = np.exp(np.mean(np.log(spectrum_pos + 1e-10)))
        arith_mean = np.mean(spectrum_pos)
        flatness = geom_mean / (arith_mean + 1e-10)
        
        # Peak-to-floor ratio
        floor = np.median(spectrum_pos)
        peak_floor_ratio = peak_power / (floor + 1e-10)
        
        return {
            'peak_frequency_hz': peak_freq,
            'spectral_centroid_hz': centroid,
            'spectral_bandwidth_hz': bandwidth,
            'spectral_flatness': flatness,
            'peak_floor_ratio': peak_floor_ratio,
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

    def _extract_envelope_statistics(self, envelope, peak_idx, sampling_freq):
        """Skewness, kurtosis, compactness."""
        from scipy.stats import skew, kurtosis
        
        skewness = skew(envelope)
        kurt = kurtosis(envelope)
        
        # Envelope compactness: weighted centroid of energy
        energy_weights = envelope ** 2
        time_indices = np.arange(len(envelope))
        weighted_center = np.sum(time_indices * energy_weights) / (np.sum(energy_weights) + 1e-10)
        spread = np.sqrt(np.sum((time_indices - weighted_center) ** 2 * energy_weights) / (np.sum(energy_weights) + 1e-10))
        compactness = 1 / (1 + spread / sampling_freq)  # normalize by sampling freq
        
        return {
            'envelope_skewness': skewness,
            'envelope_kurtosis': kurt,
            'envelope_compactness': compactness,
        }

    def _analyze_single(self, hydrophone, sampling_freq):
        """Extract all features for single hydrophone.
        
        Returns:
            Dictionary with RAW_* and FILTERED_* prefixed features
        """
        # Get signals
        raw_signal = hydrophone.signal
        filtered_signal = self.apply_bandpass(raw_signal, sampling_freq)
        
        # Compute envelopes using Hilbert transform on both signals
        raw_envelope = np.abs(scipy_signal.hilbert(raw_signal))
        filtered_envelope = np.abs(scipy_signal.hilbert(filtered_signal))
        
        # Find peaks in each envelope separately
        peak_idx_filtered = np.argmax(filtered_envelope)
        peak_idx_raw = np.argmax(raw_envelope)
        
        features = {}
        
        # Extract features for FILTERED signal
        temporal = self._extract_temporal_features(filtered_envelope, sampling_freq, peak_idx_filtered)
        filt_pulse_width = temporal['pulse_width_ms']
        
        features.update({f"FILTERED_{k}": v for k, v in temporal.items()})
        features.update({f"FILTERED_{k}": v for k, v in self._extract_energy_features(filtered_envelope, sampling_freq, peak_idx_filtered, filt_pulse_width).items()})
        features.update({f"FILTERED_{k}": v for k, v in self._extract_amplitude_features(filtered_envelope, filtered_signal).items()})
        features.update({f"FILTERED_{k}": v for k, v in self._extract_snr_features(filtered_envelope, filtered_signal, peak_idx_filtered, sampling_freq, filt_pulse_width).items()})
        
        filtered_freq = fftfreq(len(filtered_signal), 1/sampling_freq)
        features.update({f"FILTERED_{k}": v for k, v in self._extract_spectral_features(filtered_signal, filtered_freq).items()})
        features.update({f"FILTERED_{k}": v for k, v in self._extract_multipath_features(filtered_envelope, sampling_freq, peak_idx_filtered, filt_pulse_width).items()})
        features.update({f"FILTERED_{k}": v for k, v in self._extract_envelope_statistics(filtered_envelope, peak_idx_filtered, sampling_freq).items()})
        
        # Extract features for RAW signal
        raw_temporal = self._extract_temporal_features(raw_envelope, sampling_freq, peak_idx_raw)
        raw_pulse_width = raw_temporal['pulse_width_ms']
        
        features.update({f"RAW_{k}": v for k, v in raw_temporal.items()})
        features.update({f"RAW_{k}": v for k, v in self._extract_energy_features(raw_envelope, sampling_freq, peak_idx_raw, raw_pulse_width).items()})
        features.update({f"RAW_{k}": v for k, v in self._extract_amplitude_features(raw_envelope, raw_signal).items()})
        features.update({f"RAW_{k}": v for k, v in self._extract_snr_features(raw_envelope, raw_signal, peak_idx_raw, sampling_freq, raw_pulse_width).items()})
        
        raw_freq = fftfreq(len(raw_signal), 1/sampling_freq)
        features.update({f"RAW_{k}": v for k, v in self._extract_spectral_features(raw_signal, raw_freq).items()})
        features.update({f"RAW_{k}": v for k, v in self._extract_multipath_features(raw_envelope, sampling_freq, peak_idx_raw, raw_pulse_width).items()})
        features.update({f"RAW_{k}": v for k, v in self._extract_envelope_statistics(raw_envelope, peak_idx_raw, sampling_freq).items()})
        
        return features

    def _plot_single_signal(self, ax_time, ax_freq, hydrophone, result, idx):
        """Optional plotting - features are table data, not visual."""
        ax_time.text(0.5, 0.5, f"Feature Extraction Complete\nHydro {idx}: {len(result)} features", 
                     ha='center', va='center', transform=ax_time.transAxes)
        ax_freq.axis('off')
