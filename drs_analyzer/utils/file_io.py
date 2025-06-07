"""
File I/O utilities for DRS Analyzer
Handles reading and writing various file formats
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple  # Add Tuple to the import
import logging

# Optional imports with fallbacks
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

try:
    from scipy.io import savemat, loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

from ..config.settings import AppSettings

class FileIOError(Exception):
    """Custom exception for file I/O errors"""
    pass

class FileManager:
    """File management utilities"""
    
    SUPPORTED_FORMATS = {
        '.csv': 'CSV file',
        '.txt': 'Text file',
        '.xlsx': 'Excel file',
        '.xls': 'Excel file (legacy)',
        '.h5': 'HDF5 file',
        '.hdf5': 'HDF5 file',
        '.json': 'JSON file'
    }
    
    def __init__(self, settings: AppSettings = None):
        self.settings = settings or AppSettings()
        self.logger = logging.getLogger(__name__)
        if not HAS_HDF5:
            self.logger.warning("h5py not available - HDF5 support disabled")
        if not HAS_SCIPY:
            self.logger.warning("scipy not available - MATLAB file support disabled")
    
    def load_data(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load spectral data from file"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileIOError(f"File not found: {file_path}")
            
            suffix = file_path.suffix.lower()
            
            if suffix not in self.SUPPORTED_FORMATS:
                raise FileIOError(f"Unsupported file format: {suffix}")
            
            self.logger.info(f"Loading data from {file_path}")
            
            if suffix == '.csv' or suffix == '.txt':
                return self._load_csv(file_path)
            elif suffix in ['.xlsx', '.xls']:
                return self._load_excel(file_path)
            elif suffix in ['.h5', '.hdf5']:
                return self._load_hdf5(file_path)
            elif suffix == '.json':
                return self._load_json(file_path)
            
        except Exception as e:
            raise FileIOError(f"Failed to load data from {file_path}: {e}")
    
    def _load_csv(self, file_path: Path) -> Dict[str, Any]:
        """Load data from CSV/text file"""
        try:
            # Try to detect separator and structure
            with open(file_path, 'r') as f:
                first_lines = [f.readline().strip() for _ in range(5)]
            
            # Detect separator
            separators = [',', '\t', ';', ' ']
            best_sep = ','
            max_cols = 0
            
            for sep in separators:
                cols = len(first_lines[0].split(sep))
                if cols > max_cols:
                    max_cols = cols
                    best_sep = sep
            
            # Load data
            df = pd.read_csv(file_path, sep=best_sep)
            
            # Determine data structure
            if 'wavelength' in df.columns.str.lower():
                # Wavelength column present
                wavelength_col = df.columns[df.columns.str.lower().str.contains('wavelength')][0]
                wavelengths = df[wavelength_col].values
                
                # Remove wavelength column for spectra
                spectra_df = df.drop(columns=[wavelength_col])
                spectra = spectra_df.values.T  # Transpose so each row is a spectrum
                
            else:
                # Assume first row/column contains wavelengths
                if df.iloc[0].dtype in [np.float64, np.int64]:
                    # First row is wavelengths
                    wavelengths = df.iloc[0].values
                    spectra = df.iloc[1:].values
                else:
                    # First column is wavelengths
                    wavelengths = df.iloc[:, 0].values
                    spectra = df.iloc[:, 1:].values.T
            
            metadata = {
                'filename': file_path.name,
                'file_format': 'CSV',
                'separator': best_sep,
                'original_shape': df.shape
            }
            
            return {
                'spectra': spectra,
                'wavelengths': wavelengths,
                'metadata': metadata
            }
            
        except Exception as e:
            raise FileIOError(f"CSV loading failed: {e}")
    
    def _load_excel(self, file_path: Path) -> Dict[str, Any]:
        """Load data from Excel file"""
        try:
            # Try to load the first sheet
            df = pd.read_excel(file_path, sheet_name=0)
            
            # Similar logic to CSV loading
            if 'wavelength' in df.columns.str.lower():
                wavelength_col = df.columns[df.columns.str.lower().str.contains('wavelength')][0]
                wavelengths = df[wavelength_col].values
                spectra_df = df.drop(columns=[wavelength_col])
                spectra = spectra_df.values.T
            else:
                if df.iloc[0].dtype in [np.float64, np.int64]:
                    wavelengths = df.iloc[0].values
                    spectra = df.iloc[1:].values
                else:
                    wavelengths = df.iloc[:, 0].values
                    spectra = df.iloc[:, 1:].values.T
            
            metadata = {
                'filename': file_path.name,
                'file_format': 'Excel',
                'sheet_name': 0,
                'original_shape': df.shape
            }
            
            return {
                'spectra': spectra,
                'wavelengths': wavelengths,
                'metadata': metadata
            }
            
        except Exception as e:
            raise FileIOError(f"Excel loading failed: {e}")
    
    def _load_hdf5(self, file_path: Path) -> Dict[str, Any]:
        """Load data from HDF5 file"""
        try:
            with h5py.File(file_path, 'r') as f:
                # Try common dataset names
                spectra_names = ['spectra', 'data', 'intensities', 'y']
                wavelength_names = ['wavelengths', 'wavelength', 'x', 'wavenumbers']
                
                spectra = None
                wavelengths = None
                
                # Find spectra data
                for name in spectra_names:
                    if name in f:
                        spectra = f[name][:]
                        break
                
                # Find wavelength data
                for name in wavelength_names:
                    if name in f:
                        wavelengths = f[name][:]
                        break
                
                if spectra is None:
                    # Use first 2D dataset found
                    for key in f.keys():
                        if len(f[key].shape) == 2:
                            spectra = f[key][:]
                            break
                
                if wavelengths is None:
                    # Use first 1D dataset found
                    for key in f.keys():
                        if len(f[key].shape) == 1 and f[key].shape[0] == spectra.shape[-1]:
                            wavelengths = f[key][:]
                            break
                
                if spectra is None or wavelengths is None:
                    raise FileIOError("Could not find spectra and wavelength data in HDF5 file")
                
                # Load metadata if available
                metadata = {
                    'filename': file_path.name,
                    'file_format': 'HDF5',
                    'datasets': list(f.keys())
                }
                
                # Try to load attributes as metadata
                for key in f.attrs.keys():
                    metadata[key] = f.attrs[key]
            
            return {
                'spectra': spectra,
                'wavelengths': wavelengths,
                'metadata': metadata
            }
            
        except Exception as e:
            raise FileIOError(f"HDF5 loading failed: {e}")
    
    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        """Load data from JSON file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert lists to numpy arrays
            spectra = np.array(data['spectra'])
            wavelengths = np.array(data['wavelengths'])
            metadata = data.get('metadata', {})
            metadata['filename'] = file_path.name
            metadata['file_format'] = 'JSON'
            
            return {
                'spectra': spectra,
                'wavelengths': wavelengths,
                'metadata': metadata
            }
            
        except Exception as e:
            raise FileIOError(f"JSON loading failed: {e}")
    
    def save_data(self, data: Dict[str, Any], file_path: Union[str, Path], 
                  format_type: Optional[str] = None) -> bool:
        """Save spectral data to file"""
        try:
            file_path = Path(file_path)
            
            if format_type is None:
                format_type = file_path.suffix.lower()
            
            self.logger.info(f"Saving data to {file_path}")
            
            if format_type == '.csv':
                return self._save_csv(data, file_path)
            elif format_type in ['.xlsx', '.xls']:
                return self._save_excel(data, file_path)
            elif format_type in ['.h5', '.hdf5']:
                return self._save_hdf5(data, file_path)
            elif format_type == '.json':
                return self._save_json(data, file_path)
            else:
                raise FileIOError(f"Unsupported save format: {format_type}")
                
        except Exception as e:
            raise FileIOError(f"Failed to save data: {e}")
    
    def _save_csv(self, data: Dict[str, Any], file_path: Path) -> bool:
        """Save data to CSV file"""
        try:
            spectra = data['spectra']
            wavelengths = data['wavelengths']
            
            # Create DataFrame
            df_data = {'wavelength': wavelengths}
            
            # Add spectra columns
            for i, spectrum in enumerate(spectra):
                df_data[f'spectrum_{i+1}'] = spectrum
            
            df = pd.DataFrame(df_data)
            df.to_csv(file_path, index=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"CSV save failed: {e}")
            return False
    
    def _save_excel(self, data: Dict[str, Any], file_path: Path) -> bool:
        """Save data to Excel file"""
        try:
            spectra = data['spectra']
            wavelengths = data['wavelengths']
            
            df_data = {'wavelength': wavelengths}
            for i, spectrum in enumerate(spectra):
                df_data[f'spectrum_{i+1}'] = spectrum
            
            df = pd.DataFrame(df_data)
            
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Spectra', index=False)
                
                # Add metadata sheet if available
                if 'metadata' in data:
                    metadata_df = pd.DataFrame(list(data['metadata'].items()), 
                                             columns=['Parameter', 'Value'])
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Excel save failed: {e}")
            return False
    
    def _save_hdf5(self, data: Dict[str, Any], file_path: Path) -> bool:
        """Save data to HDF5 file"""
        try:
            with h5py.File(file_path, 'w') as f:
                f.create_dataset('spectra', data=data['spectra'])
                f.create_dataset('wavelengths', data=data['wavelengths'])
                
                # Save metadata as attributes
                if 'metadata' in data:
                    for key, value in data['metadata'].items():
                        try:
                            f.attrs[key] = value
                        except:
                            # Skip non-serializable metadata
                            pass
            
            return True
            
        except Exception as e:
            self.logger.error(f"HDF5 save failed: {e}")
            return False
    
    def _save_json(self, data: Dict[str, Any], file_path: Path) -> bool:
        """Save data to JSON file"""
        try:
            json_data = {
                'spectra': data['spectra'].tolist(),
                'wavelengths': data['wavelengths'].tolist(),
                'metadata': data.get('metadata', {})
            }
            
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            return True
            
        except Exception as e:
            self.logger.error(f"JSON save failed: {e}")
            return False
    
    def create_backup(self, data: Dict[str, Any]) -> Path:
        """Create backup file"""
        try:
            backup_dir = Path.home() / '.drs_analyzer' / 'backups'
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f'backup_{timestamp}.h5'
            
            self._save_hdf5(data, backup_path)
            return backup_path
            
        except Exception as e:
            raise FileIOError(f"Backup creation failed: {e}")
    
    def get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Get file information without loading data"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                raise FileIOError(f"File not found: {file_path}")
            
            info = {
                'path': str(file_path),
                'name': file_path.name,
                'size': file_path.stat().st_size,
                'format': file_path.suffix.lower(),
                'supported': file_path.suffix.lower() in self.SUPPORTED_FORMATS
            }
            
            return info
            
        except Exception as e:
            raise FileIOError(f"Failed to get file info: {e}")
    
    def validate_file(self, file_path: Union[str, Path]) -> Tuple[bool, str]:
        """Validate file format and basic structure"""
        try:
            file_path = Path(file_path)
            
            if not file_path.exists():
                return False, "File does not exist"
            
            if file_path.suffix.lower() not in self.SUPPORTED_FORMATS:
                return False, f"Unsupported format: {file_path.suffix}"
            
            # Try to load a small portion to validate structure
            try:
                data = self.load_data(file_path)
                if 'spectra' not in data or 'wavelengths' not in data:
                    return False, "Missing required data fields"
                
                spectra = data['spectra']
                wavelengths = data['wavelengths']
                
                if len(wavelengths) != spectra.shape[1]:
                    return False, "Wavelength array length doesn't match spectra"
                
                return True, "File is valid"
                
            except Exception as e:
                return False, f"Data validation failed: {e}"
                
        except Exception as e:
            return False, f"Validation error: {e}"