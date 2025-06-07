"""
File utilities for DRS Analyzer
"""

import os
import shutil
import logging
from pathlib import Path
from typing import List, Union, Optional
import hashlib

class FileUtils:  # Make sure this is the actual class name
    """File utility functions"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def find_files(self, 
                   directory: Union[str, Path], 
                   extensions: List[str], 
                   recursive: bool = True) -> List[Path]:
        """Find files with given extensions"""
        try:
            directory = Path(directory)
            if not directory.exists():
                self.logger.warning(f"Directory does not exist: {directory}")
                return []
                
            found_files = []
            
            for ext in extensions:
                if not ext.startswith('.'):
                    ext = f'.{ext}'
                
                if recursive:
                    found_files.extend(directory.rglob(f"*{ext}"))
                else:
                    found_files.extend(directory.glob(f"*{ext}"))
            
            return found_files
        except Exception as e:
            self.logger.error(f"Error finding files: {e}")
            return []
    
    def get_file_hash(self, file_path: Union[str, Path]) -> Optional[str]:
        """Get MD5 hash of a file"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                return None
                
            hash_md5 = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            self.logger.error(f"Error calculating file hash: {e}")
            return None
    
    def copy_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Copy a file"""
        try:
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            self.logger.error(f"Error copying file: {e}")
            return False
    
    def move_file(self, src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Move a file"""
        try:
            shutil.move(src, dst)
            return True
        except Exception as e:
            self.logger.error(f"Error moving file: {e}")
            return False
    
    def delete_file(self, file_path: Union[str, Path]) -> bool:
        """Delete a file"""
        try:
            Path(file_path).unlink()
            return True
        except Exception as e:
            self.logger.error(f"Error deleting file: {e}")
            return False