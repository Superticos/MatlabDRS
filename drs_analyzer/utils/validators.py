"""
Data validation utilities for DRS spectroscopy
Comprehensive validation for spectral data and parameters
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
import warnings

class ValidationError(Exception):
    """Custom exception for validation errors"""
    pass

class DataValidator:
    """
    Comprehensive data validator for DRS spectroscopy
    Validates spectral data, wavelengths, and processing parameters
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.warnings = []
        self.errors = []
    
    def validate_spectral_data(self, spectra: np.ndarray, wavelengths: np.ndarray) -> Dict[str, Any]:
        """Comprehensive validation of spectral data"""
        self.warnings.clear()
        self.errors.clear()
        
        try:
            # Basic type validation
            if not isinstance(spectra, np.ndarray):
                self.errors.append("Spectra must be a numpy array")
            
            if not isinstance(wavelengths, np.ndarray):
                self.errors.append("Wavelengths must be a numpy array")
            
            if self.errors:
                return self._create_validation_result(False)
            
            # Shape validation
            if spectra.ndim != 2:
                self.errors.append(f"Spectra must be 2D array, got {spectra.ndim}D")
            
            if wavelengths.ndim != 1:
                self.errors.append(f"Wavelengths must be 1D array, got {wavelengths.ndim}D")
            
            if self.errors:
                return self._create_validation_result(False)
            
            # Dimension compatibility
            if spectra.shape[1] != len(wavelengths):
                self.errors.append(
                    f"Spectra columns ({spectra.shape[1]}) must match wavelength length ({len(wavelengths)})"
                )
            
            # Data quality checks
            self._validate_data_quality(spectra, wavelengths)
            
            # Range validation
            self._validate_wavelength_range(wavelengths)
            self._validate_intensity_range(spectra)
            
            # Statistical validation
            self._validate_statistical_properties(spectra)
            
            is_valid = len(self.errors) == 0
            
            result = self._create_validation_result(is_valid)
            
            if is_valid:
                self.logger.info(f"Data validation passed for {spectra.shape[0]} spectra")
            else:
                self.logger.error(f"Data validation failed with {len(self.errors)} errors")
            
            return result
            
        except Exception as e:
            self.errors.append(f"Validation failed with exception: {e}")
            return self._create_validation_result(False)
    
    def _validate_data_quality(self, spectra: np.ndarray, wavelengths: np.ndarray):
        """Validate data quality issues"""
        
        # Check for NaN values
        nan_spectra = np.isnan(spectra).sum()
        if nan_spectra > 0:
            self.errors.append(f"Found {nan_spectra} NaN values in spectra")
        
        nan_wavelengths = np.isnan(wavelengths).sum()
        if nan_wavelengths > 0:
            self.errors.append(f"Found {nan_wavelengths} NaN values in wavelengths")
        
        # Check for infinite values
        inf_spectra = np.isinf(spectra).sum()
        if inf_spectra > 0:
            self.errors.append(f"Found {inf_spectra} infinite values in spectra")
        
        inf_wavelengths = np.isinf(wavelengths).sum()
        if inf_wavelengths > 0:
            self.errors.append(f"Found {inf_wavelengths} infinite values in wavelengths")
        
        # Check for negative wavelengths
        negative_wavelengths = (wavelengths < 0).sum()
        if negative_wavelengths > 0:
            self.errors.append(f"Found {negative_wavelengths} negative wavelengths")
        
        # Check wavelength ordering
        if not np.all(np.diff(wavelengths) > 0):
            self.warnings.append("Wavelengths are not in ascending order")
        
        # Check for duplicate wavelengths
        unique_wavelengths = len(np.unique(wavelengths))
        if unique_wavelengths != len(wavelengths):
            duplicates = len(wavelengths) - unique_wavelengths
            self.warnings.append(f"Found {duplicates} duplicate wavelengths")
        
        # Check for zero or negative intensities
        zero_intensities = (spectra <= 0).sum()
        if zero_intensities > 0:
            self.warnings.append(f"Found {zero_intensities} zero or negative intensities")
    
    def _validate_wavelength_range(self, wavelengths: np.ndarray):
        """Validate wavelength range"""
        
        min_wl = wavelengths.min()
        max_wl = wavelengths.max()
        
        # Typical spectroscopy ranges
        if min_wl < 100:
            self.warnings.append(f"Minimum wavelength ({min_wl:.1f}) is unusually low")
        
        if max_wl > 10000:
            self.warnings.append(f"Maximum wavelength ({max_wl:.1f}) is unusually high")
        
        # Check for reasonable spacing
        spacing = np.diff(wavelengths)
        mean_spacing = np.mean(spacing)
        std_spacing = np.std(spacing)
        
        if std_spacing > 0.1 * mean_spacing:
            self.warnings.append("Wavelength spacing is highly irregular")
        
        # Check for too fine or too coarse resolution
        if mean_spacing < 0.01:
            self.warnings.append(f"Wavelength resolution ({mean_spacing:.3f}) is very fine")
        elif mean_spacing > 50:
            self.warnings.append(f"Wavelength resolution ({mean_spacing:.1f}) is very coarse")
    
    def _validate_intensity_range(self, spectra: np.ndarray):
        """Validate intensity range"""
        
        min_intensity = spectra.min()
        max_intensity = spectra.max()
        
        # Check for unreasonable intensity values
        if max_intensity > 1e6:
            self.warnings.append(f"Maximum intensity ({max_intensity:.1e}) is very high")
        
        # Check dynamic range
        if min_intensity > 0:
            dynamic_range = max_intensity / min_intensity
            if dynamic_range > 1e6:
                self.warnings.append(f"Dynamic range ({dynamic_range:.1e}) is very large")
        
        # Check for flat spectra
        spectrum_ranges = np.ptp(spectra, axis=1)
        flat_spectra = (spectrum_ranges < 0.01 * np.median(spectrum_ranges)).sum()
        if flat_spectra > 0:
            self.warnings.append(f"Found {flat_spectra} potentially flat spectra")
    
    def _validate_statistical_properties(self, spectra: np.ndarray):
        """Validate statistical properties"""
        
        # Check for outlier spectra
        spectrum_means = np.mean(spectra, axis=1)
        mean_threshold = 3 * np.std(spectrum_means)
        outliers = np.abs(spectrum_means - np.mean(spectrum_means)) > mean_threshold
        n_outliers = outliers.sum()
        
        if n_outliers > 0:
            self.warnings.append(f"Found {n_outliers} potential outlier spectra")
        
        # Check signal-to-noise ratio
        signal = np.mean(spectra, axis=0)
        noise = np.std(spectra, axis=0)
        snr = np.mean(signal / (noise + 1e-10))
        
        if snr < 10:
            self.warnings.append(f"Low signal-to-noise ratio: {snr:.1f}")
        elif snr > 1000:
            self.warnings.append(f"Unusually high signal-to-noise ratio: {snr:.1f}")
    
    def validate_processing_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate processing parameters"""
        self.warnings.clear()
        self.errors.clear()
        
        try:
            # Baseline correction parameters
            if 'baseline_method' in parameters:
                method = parameters['baseline_method']
                if method not in ['polynomial', 'asymmetric_least_squares', 'rolling_ball', 'snip']:
                    self.errors.append(f"Unknown baseline method: {method}")
                
                if method == 'polynomial':
                    degree = parameters.get('baseline_degree', 2)
                    if not isinstance(degree, int) or degree < 0 or degree > 10:
                        self.errors.append(f"Baseline degree must be integer 0-10, got {degree}")
            
            # Smoothing parameters
            if 'smoothing_method' in parameters:
                method = parameters['smoothing_method']
                if method not in ['savgol', 'gaussian', 'moving_average', 'median']:
                    self.errors.append(f"Unknown smoothing method: {method}")
                
                window = parameters.get('smoothing_window', 5)
                if not isinstance(window, int) or window < 3 or window > 101:
                    self.errors.append(f"Smoothing window must be integer 3-101, got {window}")
                
                if window % 2 == 0:
                    self.warnings.append("Smoothing window should be odd for best results")
            
            # Normalization parameters
            if 'normalization_method' in parameters:
                method = parameters['normalization_method']
                if method not in ['minmax', 'standard', 'robust', 'vector', 'snv', 'msc']:
                    self.errors.append(f"Unknown normalization method: {method}")
            
            # Peak detection parameters
            if 'height_threshold' in parameters:
                height = parameters['height_threshold']
                if not isinstance(height, (int, float)) or height < 0:
                    self.errors.append("Height threshold must be non-negative number")
            
            if 'prominence_threshold' in parameters:
                prominence = parameters['prominence_threshold']
                if not isinstance(prominence, (int, float)) or prominence < 0:
                    self.errors.append("Prominence threshold must be non-negative number")
            
            return self._create_validation_result(len(self.errors) == 0)
            
        except Exception as e:
            self.errors.append(f"Parameter validation failed: {e}")
            return self._create_validation_result(False)
    
    def validate_file_structure(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Validate file structure and format"""
        self.warnings.clear()
        self.errors.clear()
        
        try:
            file_path = Path(file_path)
            
            # Basic file checks
            if not file_path.exists():
                self.errors.append(f"File does not exist: {file_path}")
                return self._create_validation_result(False)
            
            if file_path.stat().st_size == 0:
                self.errors.append("File is empty")
                return self._create_validation_result(False)
            
            # File extension validation
            suffix = file_path.suffix.lower()
            supported_formats = ['.csv', '.txt', '.xlsx', '.xls', '.h5', '.hdf5', '.json']
            
            if suffix not in supported_formats:
                self.errors.append(f"Unsupported file format: {suffix}")
            
            # File size validation
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 1000:  # 1GB limit
                self.warnings.append(f"Large file size: {file_size_mb:.1f} MB")
            
            # Format-specific validation
            if suffix in ['.csv', '.txt']:
                self._validate_csv_structure(file_path)
            elif suffix in ['.xlsx', '.xls']:
                self._validate_excel_structure(file_path)
            elif suffix in ['.h5', '.hdf5']:
                self._validate_hdf5_structure(file_path)
            elif suffix == '.json':
                self._validate_json_structure(file_path)
            
            return self._create_validation_result(len(self.errors) == 0)
            
        except Exception as e:
            self.errors.append(f"File validation failed: {e}")
            return self._create_validation_result(False)
    
    def _validate_csv_structure(self, file_path: Path):
        """Validate CSV file structure"""
        try:
            # Read first few lines to check structure
            with open(file_path, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
            
            if not any(first_lines):
                self.errors.append("CSV file appears to be empty")
                return
            
            # Check for common separators
            separators = [',', '\t', ';', ' ']
            separator_counts = {}
            
            for sep in separators:
                separator_counts[sep] = first_lines[0].count(sep)
            
            best_sep = max(separator_counts, key=separator_counts.get)
            if separator_counts[best_sep] == 0:
                self.warnings.append("Could not detect column separator")
            
            # Try to read with pandas
            try:
                df = pd.read_csv(file_path, sep=best_sep, nrows=10)
                if df.empty:
                    self.errors.append("CSV file has no readable data")
                elif df.shape[1] < 2:
                    self.warnings.append("CSV file has only one column")
            except Exception as e:
                self.errors.append(f"Could not parse CSV file: {e}")
                
        except Exception as e:
            self.errors.append(f"CSV validation failed: {e}")
    
    def _validate_excel_structure(self, file_path: Path):
        """Validate Excel file structure"""
        try:
            df = pd.read_excel(file_path, nrows=10)
            if df.empty:
                self.errors.append("Excel file has no readable data")
            elif df.shape[1] < 2:
                self.warnings.append("Excel file has only one column")
        except Exception as e:
            self.errors.append(f"Could not read Excel file: {e}")
    
    def _validate_hdf5_structure(self, file_path: Path):
        """Validate HDF5 file structure"""
        try:
            import h5py
            with h5py.File(file_path, 'r') as f:
                if len(f.keys()) == 0:
                    self.errors.append("HDF5 file has no datasets")
                else:
                    # Look for typical dataset names
                    expected_names = ['spectra', 'data', 'wavelengths', 'wavelength']
                    found_names = list(f.keys())
                    
                    if not any(name in found_names for name in expected_names):
                        self.warnings.append(f"No standard dataset names found. Available: {found_names}")
        except Exception as e:
            self.errors.append(f"Could not read HDF5 file: {e}")
    
    def _validate_json_structure(self, file_path: Path):
        """Validate JSON file structure"""
        try:
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                self.errors.append("JSON file must contain a dictionary")
            else:
                required_keys = ['spectra', 'wavelengths']
                missing_keys = [key for key in required_keys if key not in data]
                if missing_keys:
                    self.errors.append(f"JSON file missing required keys: {missing_keys}")
        except Exception as e:
            self.errors.append(f"Could not parse JSON file: {e}")
    
    def _create_validation_result(self, is_valid: bool) -> Dict[str, Any]:
        """Create standardized validation result"""
        return {
            'is_valid': is_valid,
            'errors': self.errors.copy(),
            'warnings': self.warnings.copy(),
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
    
    def validate_export_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Validate export parameters"""
        self.warnings.clear()
        self.errors.clear()
        
        try:
            # File format validation
            if 'format' in parameters:
                fmt = parameters['format'].lower()
                if fmt not in ['png', 'jpg', 'jpeg', 'pdf', 'svg', 'tiff', 'eps']:
                    self.errors.append(f"Unsupported export format: {fmt}")
            
            # DPI validation
            if 'dpi' in parameters:
                dpi = parameters['dpi']
                if not isinstance(dpi, int) or dpi < 50 or dpi > 1200:
                    self.errors.append("DPI must be integer between 50-1200")
            
            # Size validation
            if 'figsize' in parameters:
                figsize = parameters['figsize']
                if not isinstance(figsize, (list, tuple)) or len(figsize) != 2:
                    self.errors.append("Figure size must be (width, height) tuple")
                elif any(dim <= 0 or dim > 50 for dim in figsize):
                    self.errors.append("Figure dimensions must be between 0-50 inches")
            
            return self._create_validation_result(len(self.errors) == 0)
            
        except Exception as e:
            self.errors.append(f"Export parameter validation failed: {e}")
            return self._create_validation_result(False)
    
    def validate_memory_requirements(self, data_shape: Tuple[int, int], 
                                   operation: str = "processing") -> Dict[str, Any]:
        """Validate memory requirements for operations"""
        self.warnings.clear()
        self.errors.clear()
        
        try:
            n_spectra, n_wavelengths = data_shape
            
            # Estimate memory usage (rough approximation)
            base_memory_mb = (n_spectra * n_wavelengths * 8) / (1024 * 1024)  # float64
            
            if operation == "processing":
                estimated_memory_mb = base_memory_mb * 3  # Original + processed + temporary
            elif operation == "pca":
                estimated_memory_mb = base_memory_mb * 5  # Additional for covariance matrices
            elif operation == "clustering":
                estimated_memory_mb = base_memory_mb * 2  # Distance matrices
            else:
                estimated_memory_mb = base_memory_mb
            
            # Memory warnings/errors
            if estimated_memory_mb > 8192:  # 8GB
                self.errors.append(f"Operation may require {estimated_memory_mb:.0f} MB memory")
            elif estimated_memory_mb > 4096:  # 4GB
                self.warnings.append(f"Operation may require {estimated_memory_mb:.0f} MB memory")
            
            # Data size warnings
            if n_spectra > 10000:
                self.warnings.append(f"Large number of spectra: {n_spectra}")
            
            if n_wavelengths > 10000:
                self.warnings.append(f"Large number of wavelength points: {n_wavelengths}")
            
            return self._create_validation_result(len(self.errors) == 0)
            
        except Exception as e:
            self.errors.append(f"Memory validation failed: {e}")
            return self._create_validation_result(False)
    
    def get_validation_summary(self) -> str:
        """Get human-readable validation summary"""
        summary = []
        
        if self.errors:
            summary.append(f"❌ {len(self.errors)} errors found:")
            for error in self.errors:
                summary.append(f"  • {error}")
        
        if self.warnings:
            summary.append(f"⚠️  {len(self.warnings)} warnings:")
            for warning in self.warnings:
                summary.append(f"  • {warning}")
        
        if not self.errors and not self.warnings:
            summary.append("✅ Validation passed with no issues")
        
        return "\n".join(summary)