"""
Export manager for DRS Analyzer
Handles exporting data in various formats
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

# Optional imports
try:
    import h5py
    HAS_HDF5 = True
except ImportError:
    HAS_HDF5 = False

try:
    from scipy.io import savemat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

class ExportManager:
    """Manages data export in various formats"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def export_to_csv(self, data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """Export data to CSV format"""
        try:
            file_path = Path(file_path)
            if 'spectra' in data and 'wavelengths' in data:
                df = pd.DataFrame(data['spectra'])
                if 'wavelengths' in data:
                    df.index = data['wavelengths']
                df.to_csv(file_path)
                return True
        except Exception as e:
            self.logger.error(f"Failed to export to CSV: {e}")
            return False
    
    def export_to_json(self, data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """Export data to JSON format"""
        try:
            file_path = Path(file_path)
            # Convert numpy arrays to lists for JSON serialization
            json_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    json_data[key] = value.tolist()
                else:
                    json_data[key] = value
            
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            return True
        except Exception as e:
            self.logger.error(f"Failed to export to JSON: {e}")
            return False
    
    def export_to_hdf5(self, data: Dict[str, Any], file_path: Union[str, Path]) -> bool:
        """Export data to HDF5 format"""
        if not HAS_HDF5:
            self.logger.error("HDF5 export not available - h5py not installed")
            return False
        
        try:
            file_path = Path(file_path)
            with h5py.File(file_path, 'w') as f:
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        f.create_dataset(key, data=value)
                    else:
                        f.attrs[key] = value
            return True
        except Exception as e:
            self.logger.error(f"Failed to export to HDF5: {e}")
            return False