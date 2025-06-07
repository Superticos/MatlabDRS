"""
Utility modules for DRS Analyzer
"""

# Make imports optional to handle missing dependencies gracefully
try:
    from .file_io import FileManager
except ImportError as e:
    print(f"Warning: FileManager not available: {e}")
    FileManager = None

try:
    from .validators import DataValidator
except ImportError as e:
    print(f"Warning: DataValidator not available: {e}")
    DataValidator = None

try:
    from .export_manager import ExportManager
except ImportError as e:
    print(f"Warning: ExportManager not available: {e}")
    ExportManager = None

try:
    from .file_utils import FileUtils
except ImportError as e:
    print(f"Warning: FileUtils not available: {e}")
    FileUtils = None

try:
    from .logger import setup_logging
except ImportError as e:
    print(f"Warning: Logger setup not available: {e}")
    setup_logging = None

# Only export what's actually available
__all__ = []
if FileManager is not None:
    __all__.append('FileManager')
if DataValidator is not None:
    __all__.append('DataValidator')
if ExportManager is not None:
    __all__.append('ExportManager')
if FileUtils is not None:
    __all__.append('FileUtils')
if setup_logging is not None:
    __all__.append('setup_logging')