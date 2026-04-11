"""Nearby detection using trained Random Forest model on optimal 7 features."""
import numpy as np
import joblib
from pathlib import Path

from .base_analyzer import BaseAnalyzer
from .feature_analyzer import FeatureAnalyzer


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
        
        # Initialize feature analyzer for computing all features
        self.feature_analyzer = FeatureAnalyzer(**kwargs)
        
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
                - all_features: All 27+ features for debugging
        """
        # Step 1: Extract all 27+ features using FeatureAnalyzer
        all_features = self.feature_analyzer._analyze_single(hydrophone, sampling_freq)
        
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
            'all_features': all_features,  # For debugging/analysis
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
