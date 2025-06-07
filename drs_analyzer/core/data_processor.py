"""
Advanced data processing for DRS spectroscopy
High-performance algorithms with parallel processing support
"""

import numpy as np
import pandas as pd
from scipy import signal, sparse, optimize
from scipy.ndimage import gaussian_filter1d
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA
import logging
from typing import Tuple, Optional, List, Dict, Any, Union
import warnings

# Optional high-performance imports
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Create dummy decorators
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(*args, **kwargs):
        return range(*args, **kwargs)

from ..config.settings import AppSettings, ProcessingSettings

class DataProcessingError(Exception):
    """Custom exception for data processing errors"""
    pass

class DataProcessor:
    """
    Advanced data processor for DRS spectroscopy
    Optimized for performance with parallel processing
    """
    
    def __init__(self, settings: AppSettings = None):
        self.settings = settings or AppSettings()
        self.logger = logging.getLogger(__name__)
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.wavelengths = None
        self.metadata = {}
        
        # Processing cache
        self._cache = {}
        self._processing_history = []
        
        # Preprocessing options
        self.scalers = {
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'robust': RobustScaler()
        }
    
    def load_data(self, spectra: np.ndarray, wavelengths: np.ndarray, 
                  metadata: Dict[str, Any] = None):
        """Load spectral data for processing"""
        try:
            if spectra is None or wavelengths is None:
                raise DataProcessingError("Spectra and wavelengths cannot be None")
            
            if len(wavelengths) != spectra.shape[1]:
                raise DataProcessingError("Wavelength array length must match spectra columns")
            
            self.raw_data = np.array(spectra, dtype=np.float64)
            self.processed_data = self.raw_data.copy()
            self.wavelengths = np.array(wavelengths, dtype=np.float64)
            self.metadata = metadata or {}
            
            # Clear cache
            self._cache.clear()
            self._processing_history.clear()
            
            self.logger.info(f"Loaded {spectra.shape[0]} spectra with {len(wavelengths)} wavelength points")
            
        except Exception as e:
            raise DataProcessingError(f"Failed to load data: {e}")
    
    def reset_processing(self):
        """Reset to original raw data"""
        if self.raw_data is not None:
            self.processed_data = self.raw_data.copy()
            self._cache.clear()
            self._processing_history.clear()
            self.logger.info("Reset to raw data")
    
    def apply_baseline_correction(self, method: str = "polynomial", 
                                **kwargs) -> np.ndarray:
        """Apply baseline correction to spectra"""
        try:
            if self.processed_data is None:
                raise DataProcessingError("No data loaded")
            
            cache_key = f"baseline_{method}_{hash(str(kwargs))}"
            if cache_key in self._cache:
                self.processed_data = self._cache[cache_key]
                return self.processed_data
            
            corrected_data = np.zeros_like(self.processed_data)
            
            if method == "polynomial":
                degree = kwargs.get('degree', 2)
                for i, spectrum in enumerate(self.processed_data):
                    baseline = self._polynomial_baseline(spectrum, degree)
                    corrected_data[i] = spectrum - baseline
            
            elif method == "asymmetric_least_squares":
                lam = kwargs.get('lambda', 1e4)
                p = kwargs.get('p', 0.01)
                for i, spectrum in enumerate(self.processed_data):
                    baseline = self._als_baseline(spectrum, lam, p)
                    corrected_data[i] = spectrum - baseline
            
            elif method == "rolling_ball":
                radius = kwargs.get('radius', 50)
                for i, spectrum in enumerate(self.processed_data):
                    baseline = self._rolling_ball_baseline(spectrum, radius)
                    corrected_data[i] = spectrum - baseline
            
            elif method == "snip":
                iterations = kwargs.get('iterations', 40)
                for i, spectrum in enumerate(self.processed_data):
                    baseline = self._snip_baseline(spectrum, iterations)
                    corrected_data[i] = spectrum - baseline
            
            else:
                raise DataProcessingError(f"Unknown baseline correction method: {method}")
            
            self.processed_data = corrected_data
            self._cache[cache_key] = corrected_data.copy()
            self._processing_history.append(f"baseline_correction_{method}")
            
            self.logger.info(f"Applied {method} baseline correction")
            return self.processed_data
            
        except Exception as e:
            raise DataProcessingError(f"Baseline correction failed: {e}")
    
    def _polynomial_baseline(self, spectrum: np.ndarray, degree: int) -> np.ndarray:
        """Polynomial baseline correction"""
        x = np.arange(len(spectrum))
        coeffs = np.polyfit(x, spectrum, degree)
        baseline = np.polyval(coeffs, x)
        return baseline
    
    def _als_baseline(self, spectrum: np.ndarray, lam: float, p: float) -> np.ndarray:
        """Asymmetric Least Squares baseline correction"""
        L = len(spectrum)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        
        for i in range(10):  # iterations
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = sparse.linalg.spsolve(Z, w*spectrum)
            w = p * (spectrum > z) + (1-p) * (spectrum < z)
        
        return z
    
    def _rolling_ball_baseline(self, spectrum: np.ndarray, radius: int) -> np.ndarray:
        """Rolling ball baseline correction"""
        # Simplified rolling ball algorithm
        from scipy.ndimage import minimum_filter1d, maximum_filter1d
        
        # Apply morphological operations
        baseline = minimum_filter1d(spectrum, size=radius*2)
        baseline = maximum_filter1d(baseline, size=radius)
        
        return baseline
    
    def _snip_baseline(self, spectrum: np.ndarray, iterations: int) -> np.ndarray:
        """SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) baseline"""
        baseline = spectrum.copy()
        
        for i in range(1, iterations + 1):
            for j in range(i, len(baseline) - i):
                a = baseline[j]
                b = (baseline[j - i] + baseline[j + i]) / 2
                if b < a:
                    baseline[j] = b
        
        return baseline
    
    def apply_smoothing(self, method: str = "savgol", **kwargs) -> np.ndarray:
        """Apply smoothing to spectra"""
        try:
            if self.processed_data is None:
                raise DataProcessingError("No data loaded")
            
            cache_key = f"smoothing_{method}_{hash(str(kwargs))}"
            if cache_key in self._cache:
                self.processed_data = self._cache[cache_key]
                return self.processed_data
            
            smoothed_data = np.zeros_like(self.processed_data)
            
            if method == "savgol":
                window_length = kwargs.get('window_length', 5)
                polyorder = kwargs.get('polyorder', 2)
                
                # Ensure odd window length
                if window_length % 2 == 0:
                    window_length += 1
                
                for i, spectrum in enumerate(self.processed_data):
                    smoothed_data[i] = signal.savgol_filter(spectrum, window_length, polyorder)
            
            elif method == "gaussian":
                sigma = kwargs.get('sigma', 1.0)
                for i, spectrum in enumerate(self.processed_data):
                    smoothed_data[i] = gaussian_filter1d(spectrum, sigma)
            
            elif method == "moving_average":
                window = kwargs.get('window', 5)
                for i, spectrum in enumerate(self.processed_data):
                    smoothed_data[i] = self._moving_average(spectrum, window)
            
            elif method == "median":
                kernel_size = kwargs.get('kernel_size', 3)
                for i, spectrum in enumerate(self.processed_data):
                    smoothed_data[i] = signal.medfilt(spectrum, kernel_size)
            
            else:
                raise DataProcessingError(f"Unknown smoothing method: {method}")
            
            self.processed_data = smoothed_data
            self._cache[cache_key] = smoothed_data.copy()
            self._processing_history.append(f"smoothing_{method}")
            
            self.logger.info(f"Applied {method} smoothing")
            return self.processed_data
            
        except Exception as e:
            raise DataProcessingError(f"Smoothing failed: {e}")
    
    def _moving_average(self, spectrum: np.ndarray, window: int) -> np.ndarray:
        """Simple moving average"""
        return np.convolve(spectrum, np.ones(window)/window, mode='same')
    
    def apply_normalization(self, method: str = "minmax", **kwargs) -> np.ndarray:
        """Apply normalization to spectra"""
        try:
            if self.processed_data is None:
                raise DataProcessingError("No data loaded")
            
            cache_key = f"normalization_{method}_{hash(str(kwargs))}"
            if cache_key in self._cache:
                self.processed_data = self._cache[cache_key]
                return self.processed_data
            
            normalized_data = self.processed_data.copy()
            
            if method == "minmax":
                feature_range = kwargs.get('feature_range', (0, 1))
                scaler = MinMaxScaler(feature_range=feature_range)
                normalized_data = scaler.fit_transform(normalized_data.T).T
            
            elif method == "standard":
                scaler = StandardScaler()
                normalized_data = scaler.fit_transform(normalized_data.T).T
            
            elif method == "robust":
                scaler = RobustScaler()
                normalized_data = scaler.fit_transform(normalized_data.T).T
            
            elif method == "vector":
                # L2 normalization
                norms = np.linalg.norm(normalized_data, axis=1, keepdims=True)
                normalized_data = normalized_data / (norms + 1e-8)
            
            elif method == "snv":
                # Standard Normal Variate
                normalized_data = self._snv_normalize(normalized_data)
            
            elif method == "msc":
                # Multiplicative Scatter Correction
                normalized_data = self._msc_normalize(normalized_data)
            
            else:
                raise DataProcessingError(f"Unknown normalization method: {method}")
            
            self.processed_data = normalized_data
            self._cache[cache_key] = normalized_data.copy()
            self._processing_history.append(f"normalization_{method}")
            
            self.logger.info(f"Applied {method} normalization")
            return self.processed_data
            
        except Exception as e:
            raise DataProcessingError(f"Normalization failed: {e}")
    
    def _snv_normalize(self, data: np.ndarray) -> np.ndarray:
        """Standard Normal Variate normalization"""
        normalized = np.zeros_like(data)
        for i, spectrum in enumerate(data):
            mean_spectrum = np.mean(spectrum)
            std_spectrum = np.std(spectrum)
            normalized[i] = (spectrum - mean_spectrum) / (std_spectrum + 1e-8)
        return normalized
    
    def _msc_normalize(self, data: np.ndarray) -> np.ndarray:
        """Multiplicative Scatter Correction"""
        # Calculate mean spectrum
        mean_spectrum = np.mean(data, axis=0)
        
        corrected = np.zeros_like(data)
        for i, spectrum in enumerate(data):
            # Linear regression
            coeffs = np.polyfit(mean_spectrum, spectrum, 1)
            slope, intercept = coeffs[0], coeffs[1]
            corrected[i] = (spectrum - intercept) / slope
        
        return corrected
    
    def calculate_derivative(self, order: int = 1, **kwargs) -> np.ndarray:
        """Calculate derivative of spectra"""
        try:
            if self.processed_data is None:
                raise DataProcessingError("No data loaded")
            
            cache_key = f"derivative_{order}_{hash(str(kwargs))}"
            if cache_key in self._cache:
                self.processed_data = self._cache[cache_key]
                return self.processed_data
            
            method = kwargs.get('method', 'savgol')
            
            if method == 'savgol':
                window_length = kwargs.get('window_length', 7)
                polyorder = kwargs.get('polyorder', 3)
                
                if window_length % 2 == 0:
                    window_length += 1
                
                derivative_data = np.zeros_like(self.processed_data)
                for i, spectrum in enumerate(self.processed_data):
                    derivative_data[i] = signal.savgol_filter(
                        spectrum, window_length, polyorder, deriv=order
                    )
            
            elif method == 'gradient':
                derivative_data = np.gradient(self.processed_data, axis=1)
                if order > 1:
                    for _ in range(order - 1):
                        derivative_data = np.gradient(derivative_data, axis=1)
            
            else:
                raise DataProcessingError(f"Unknown derivative method: {method}")
            
            self.processed_data = derivative_data
            self._cache[cache_key] = derivative_data.copy()
            self._processing_history.append(f"derivative_{order}_{method}")
            
            self.logger.info(f"Calculated {order} order derivative using {method}")
            return self.processed_data
            
        except Exception as e:
            raise DataProcessingError(f"Derivative calculation failed: {e}")
    
    def process_spectra(self, processing_settings: ProcessingSettings = None) -> np.ndarray:
        """Apply complete processing pipeline"""
        try:
            if processing_settings is None:
                processing_settings = self.settings.get_processing_settings()
            
            # Reset to raw data
            self.reset_processing()
            
            # Apply baseline correction
            if processing_settings.baseline_method:
                self.apply_baseline_correction(
                    method=processing_settings.baseline_method,
                    degree=processing_settings.baseline_degree
                )
            
            # Apply smoothing
            if processing_settings.smoothing_method:
                self.apply_smoothing(
                    method=processing_settings.smoothing_method,
                    window_length=processing_settings.smoothing_window
                )
            
            # Apply normalization
            if processing_settings.normalization_method:
                self.apply_normalization(method=processing_settings.normalization_method)
            
            # Calculate derivative if requested
            if processing_settings.derivative_order > 0:
                self.calculate_derivative(order=processing_settings.derivative_order)
            
            self.logger.info("Completed full processing pipeline")
            return self.processed_data
            
        except Exception as e:
            raise DataProcessingError(f"Processing pipeline failed: {e}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of applied processing steps"""
        return {
            'processing_history': self._processing_history.copy(),
            'data_shape': self.processed_data.shape if self.processed_data is not None else None,
            'wavelength_range': (self.wavelengths.min(), self.wavelengths.max()) if self.wavelengths is not None else None,
            'cache_size': len(self._cache),
            'metadata': self.metadata.copy()
        }
    
    def clear_cache(self):
        """Clear processing cache"""
        self._cache.clear()
        self.logger.info("Processing cache cleared")
    
    def export_processed_data(self) -> Dict[str, Any]:
        """Export processed data and metadata"""
        return {
            'processed_data': self.processed_data.copy() if self.processed_data is not None else None,
            'raw_data': self.raw_data.copy() if self.raw_data is not None else None,
            'wavelengths': self.wavelengths.copy() if self.wavelengths is not None else None,
            'processing_history': self._processing_history.copy(),
            'metadata': self.metadata.copy()
        }