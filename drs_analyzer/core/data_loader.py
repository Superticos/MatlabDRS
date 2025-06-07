"""
Data loading utilities for DRS spectroscopy files
Supports multiple file formats: .txt, .csv, .mat, .xlsx
"""

import os
import numpy as np
import pandas as pd  # Add this missing import
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import warnings

# Import optional dependencies with fallbacks
try:
    import scipy.io as sio
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

class DataLoadError(Exception):
    """Custom exception for data loading errors"""
    pass

class DataLoader:
    """
    Data loader for DRS spectroscopy files
    Supports .mat, .csv, .xlsx, .h5, and .txt formats
    """
    
    SUPPORTED_FORMATS = {
        '.mat': 'MATLAB file',
        '.csv': 'Comma-separated values',
        '.xlsx': 'Excel file',
        '.xls': 'Excel file (legacy)',
        '.h5': 'HDF5 file',
        '.hdf5': 'HDF5 file',
        '.txt': 'Text file',
        '.dat': 'Data file'
    }
    
    def __init__(self, settings=None):
        """Initialize DataLoader with optional settings"""
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.last_loaded_file = None
        self.cache = {}
        
    def load_file(self, file_path: Union[str, Path]) -> Tuple[bool, Dict[str, Any]]:
        """
        Load spectral data from file with comprehensive error handling
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            Tuple of (success: bool, data: Dict[str, Any])
        """
        try:
            file_path = Path(file_path)
            
            # Validate file exists and is readable
            if not file_path.exists():
                return False, {'error': f'File not found: {file_path}'}
            
            if not file_path.is_file():
                return False, {'error': f'Path is not a file: {file_path}'}
            
            # Check file permissions
            if not os.access(file_path, os.R_OK):
                return False, {'error': f'File not readable: {file_path}'}
            
            # Check file size (avoid loading huge files accidentally)
            file_size_mb = file_path.stat().st_size / (1024 * 1024)
            if file_size_mb > 1000:  # 1GB limit
                return False, {'error': f'File too large: {file_size_mb:.1f} MB (limit: 1000 MB)'}
            
            # Check format support
            suffix = file_path.suffix.lower()
            if suffix not in self.SUPPORTED_FORMATS:
                return False, {
                    'error': f'Unsupported format: {suffix}. Supported: {list(self.SUPPORTED_FORMATS.keys())}'
                }
            
            self.logger.info(f"Loading file: {file_path} ({self.SUPPORTED_FORMATS[suffix]})")
            
            # Load based on file type
            if suffix == '.mat':
                success, data = self._load_matlab_file(file_path)
            elif suffix in ['.csv', '.txt', '.dat']:
                success, data = self._load_csv_file(file_path)
            elif suffix in ['.xlsx', '.xls']:
                success, data = self._load_excel_file(file_path)
            elif suffix in ['.h5', '.hdf5']:
                success, data = self._load_hdf5_file(file_path)
            else:
                return False, {'error': f'Handler not implemented for {suffix}'}
            
            if success:
                # Validate the loaded data
                is_valid, validation_message = self._validate_spectral_data(data)
                if not is_valid:
                    return False, {'error': f'Data validation failed: {validation_message}'}
                
                # Add metadata
                data['metadata'].update({
                    'file_path': str(file_path),
                    'file_size_mb': file_size_mb,
                    'file_format': self.SUPPORTED_FORMATS[suffix],
                    'load_timestamp': pd.Timestamp.now().isoformat()
                })
                
                self.last_loaded_file = file_path
                self.logger.info(f"Successfully loaded {data['spectra'].shape[0]} spectra")
                
            return success, data
            
        except PermissionError:
            return False, {'error': f'Permission denied accessing file: {file_path}'}
        except FileNotFoundError:
            return False, {'error': f'File not found: {file_path}'}
        except MemoryError:
            return False, {'error': 'Insufficient memory to load file'}
        except Exception as e:
            self.logger.exception(f"Unexpected error loading {file_path}")
            return False, {'error': f'Unexpected error: {str(e)}'}
    
    def _load_matlab_file(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Load MATLAB .mat file"""
        if not HAS_SCIPY:
            return False, {'error': 'scipy not available for MATLAB file loading'}
        
        try:
            mat_data = sio.loadmat(file_path, squeeze_me=True)
            
            # Remove MATLAB metadata
            mat_data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
            
            # Try to identify spectra and wavelength data
            spectra, wavelengths = self._extract_spectral_data_matlab(mat_data)
            
            if spectra is None:
                return False, {'error': 'Could not identify spectral data in MATLAB file'}
            
            metadata = {
                'original_keys': list(mat_data.keys()),
                'matlab_version': mat_data.get('__version__', 'unknown')
            }
            
            return True, {
                'spectra': spectra,
                'wavelengths': wavelengths,
                'metadata': metadata,
                'raw_data': mat_data
            }
            
        except Exception as e:
            return False, {'error': f'Failed to load MATLAB file: {str(e)}'}
    
    def _load_csv_file(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Load CSV/TXT file"""
        try:
            # Try different delimiters
            delimiters = [',', '\t', ';', ' ']
            
            for delimiter in delimiters:
                try:
                    df = pd.read_csv(file_path, delimiter=delimiter)
                    if df.shape[1] > 1:  # Found proper delimiter
                        break
                except:
                    continue
            else:
                return False, {'error': 'Could not parse CSV file with any delimiter'}
            
            # Extract spectral data
            spectra, wavelengths = self._extract_spectral_data_csv(df)
            
            if spectra is None:
                return False, {'error': 'Could not identify spectral data in CSV file'}
            
            metadata = {
                'original_columns': list(df.columns),
                'delimiter': delimiter,
                'shape': df.shape
            }
            
            return True, {
                'spectra': spectra,
                'wavelengths': wavelengths,
                'metadata': metadata,
                'raw_data': df
            }
            
        except Exception as e:
            return False, {'error': f'Failed to load CSV file: {str(e)}'}
    
    def _load_excel_file(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Load Excel file"""
        try:
            # Try to read the first sheet
            df = pd.read_excel(file_path, sheet_name=0)
            
            # Extract spectral data
            spectra, wavelengths = self._extract_spectral_data_csv(df)
            
            if spectra is None:
                return False, {'error': 'Could not identify spectral data in Excel file'}
            
            metadata = {
                'original_columns': list(df.columns),
                'shape': df.shape,
                'sheet_name': 0
            }
            
            return True, {
                'spectra': spectra,
                'wavelengths': wavelengths,
                'metadata': metadata,
                'raw_data': df
            }
            
        except Exception as e:
            return False, {'error': f'Failed to load Excel file: {str(e)}'}
    
    def _load_hdf5_file(self, file_path: Path) -> Tuple[bool, Dict[str, Any]]:
        """Load HDF5 file"""
        if not HAS_H5PY:
            return False, {'error': 'h5py not available for HDF5 file loading'}
        
        try:
            with h5py.File(file_path, 'r') as f:
                # Try to find spectral data
                spectra, wavelengths = self._extract_spectral_data_hdf5(f)
                
                if spectra is None:
                    return False, {'error': 'Could not identify spectral data in HDF5 file'}
                
                metadata = {
                    'hdf5_keys': list(f.keys()),
                    'file_format': 'HDF5'
                }
                
                return True, {
                    'spectra': spectra,
                    'wavelengths': wavelengths,
                    'metadata': metadata
                }
                
        except Exception as e:
            return False, {'error': f'Failed to load HDF5 file: {str(e)}'}
    
    def _extract_spectral_data_matlab(self, mat_data: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract spectral data from MATLAB data structure"""
        spectra = None
        wavelengths = None
        
        # Common field names for spectra
        spectra_keys = ['spectra', 'data', 'reflectance', 'absorbance', 'intensity', 'y', 'Y']
        wavelength_keys = ['wavelengths', 'wavelength', 'wl', 'lambda', 'x', 'X', 'wavenumber']
        
        # Find spectra
        for key in spectra_keys:
            if key in mat_data:
                candidate = np.array(mat_data[key])
                if candidate.ndim >= 1:
                    spectra = candidate
                    break
        
        # Find wavelengths
        for key in wavelength_keys:
            if key in mat_data:
                candidate = np.array(mat_data[key])
                if candidate.ndim == 1:
                    wavelengths = candidate
                    break
        
        # Ensure proper dimensions
        if spectra is not None:
            if spectra.ndim == 1:
                spectra = spectra.reshape(1, -1)
            elif spectra.ndim > 2:
                # Flatten extra dimensions
                spectra = spectra.reshape(-1, spectra.shape[-1])
        
        # Generate wavelengths if not found
        if spectra is not None and wavelengths is None:
            wavelengths = np.arange(spectra.shape[1])
            
        return spectra, wavelengths
    
    def _extract_spectral_data_csv(self, df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract spectral data from CSV DataFrame"""
        # Assume first column might be wavelengths, rest are spectra
        # Or first row might be wavelengths
        
        try:
            # Check if first column looks like wavelengths
            first_col = df.iloc[:, 0]
            if self._looks_like_wavelengths(first_col.values):
                wavelengths = first_col.values
                spectra = df.iloc[:, 1:].values.T  # Transpose so each row is a spectrum
            else:
                # Assume each column is a spectrum
                spectra = df.values.T
                wavelengths = np.arange(spectra.shape[1])
            
            # Ensure at least 2D
            if spectra.ndim == 1:
                spectra = spectra.reshape(1, -1)
                
            return spectra, wavelengths
            
        except Exception:
            return None, None
    
    def _extract_spectral_data_hdf5(self, h5_file) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Extract spectral data from HDF5 file"""
        spectra = None
        wavelengths = None
        
        # Common dataset names
        spectra_keys = ['spectra', 'data', 'reflectance', 'absorbance', 'intensity']
        wavelength_keys = ['wavelengths', 'wavelength', 'wl', 'lambda']
        
        # Find spectra
        for key in spectra_keys:
            if key in h5_file:
                spectra = np.array(h5_file[key])
                break
        
        # Find wavelengths
        for key in wavelength_keys:
            if key in h5_file:
                wavelengths = np.array(h5_file[key])
                break
        
        # Ensure proper dimensions
        if spectra is not None and spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
            
        if spectra is not None and wavelengths is None:
            wavelengths = np.arange(spectra.shape[1])
            
        return spectra, wavelengths
    
    def _looks_like_wavelengths(self, data: np.ndarray) -> bool:
        """Check if data looks like wavelength values"""
        try:
            # Check if monotonically increasing
            if not np.all(np.diff(data) > 0):
                return False
            
            # Check reasonable range for spectroscopy (200-3000 nm)
            if np.min(data) < 100 or np.max(data) > 5000:
                return False
            
            # Check reasonable spacing
            spacing = np.diff(data)
            if np.std(spacing) / np.mean(spacing) > 0.5:  # Too irregular
                return False
                
            return True
            
        except Exception:
            return False
    
    def _validate_spectral_data(self, data: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate loaded spectral data"""
        try:
            spectra = data.get('spectra')
            wavelengths = data.get('wavelengths')
            
            if spectra is None:
                return False, "No spectral data found"
            
            if wavelengths is None:
                return False, "No wavelength data found"
            
            # Convert to numpy arrays
            spectra = np.array(spectra)
            wavelengths = np.array(wavelengths)
            
            # Check dimensions
            if spectra.ndim != 2:
                return False, f"Spectra must be 2D array, got {spectra.ndim}D"
            
            if wavelengths.ndim != 1:
                return False, f"Wavelengths must be 1D array, got {wavelengths.ndim}D"
            
            # Check shape compatibility
            if spectra.shape[1] != len(wavelengths):
                return False, f"Dimension mismatch: {spectra.shape[1]} spectral points vs {len(wavelengths)} wavelengths"
            
            # Check for invalid values
            if np.any(np.isnan(spectra)) or np.any(np.isinf(spectra)):
                return False, "Spectra contain NaN or infinite values"
            
            if np.any(np.isnan(wavelengths)) or np.any(np.isinf(wavelengths)):
                return False, "Wavelengths contain NaN or infinite values"
            
            # Update data with validated arrays
            data['spectra'] = spectra
            data['wavelengths'] = wavelengths
            
            return True, "Data validation passed"
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def get_supported_formats(self) -> Dict[str, str]:
        """Get dictionary of supported file formats"""
        return self.SUPPORTED_FORMATS.copy()
    
    def load_multiple_files(self, file_paths: List[Union[str, Path]]) -> Dict[str, Tuple[bool, Dict[str, Any]]]:
        """Load multiple files and return results"""
        results = {}
        
        for file_path in file_paths:
            results[str(file_path)] = self.load_file(file_path)
            
        return results
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get information about a file without loading it"""
        file_path = Path(file_path)
        
        try:
            stat = file_path.stat()
            return {
                'exists': file_path.exists(),
                'size_mb': stat.st_size / (1024 * 1024),
                'modified': pd.Timestamp.fromtimestamp(stat.st_mtime),
                'format': self.SUPPORTED_FORMATS.get(file_path.suffix.lower(), 'Unknown'),
                'supported': file_path.suffix.lower() in self.SUPPORTED_FORMATS
            }
        except Exception as e:
            return {
                'exists': False,
                'error': str(e)
            }