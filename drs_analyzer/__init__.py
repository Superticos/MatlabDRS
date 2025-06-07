"""
DRS Analyzer - Advanced spectroscopy analysis package
"""

__version__ = "1.0.0"
__author__ = "Spectroscopy Lab"
__email__ = "lab@example.com"

# Core imports (these should always work)
from .core.data_loader import DataLoader
from .core.data_processor import DataProcessor
from .config.settings import AppSettings

# Initialize the basic __all__ list
__all__ = [
    'DataLoader',
    'DataProcessor',
    'AppSettings'
]

# Optional analysis imports with error handling
try:
    from .analysis.pca_analyser import *  # Import whatever classes exist
    __all__.extend([name for name in dir() if not name.startswith('_') and name not in __all__])
except ImportError as e:
    print(f"Warning: Could not import PCA analyzer: {e}")

try:
    from .core.peak_detector import PeakDetector
    __all__.append('PeakDetector')
except ImportError as e:
    print(f"Warning: Could not import PeakDetector: {e}")

try:
    from .analysis.statistical_analyser import *  # Import whatever classes exist
    # Add any new classes to __all__ that aren't already there
    current_vars = [name for name in dir() if not name.startswith('_')]
    for name in current_vars:
        if name not in __all__:
            __all__.append(name)
except ImportError as e:
    print(f"Warning: Could not import Statistical analyzer: {e}")

# Optional plotting imports
try:
    from .plotting.drs_plotter import DRSPlotter
    __all__.append('DRSPlotter')
except ImportError as e:
    print(f"Warning: Could not import DRSPlotter: {e}")

# Optional utility imports
try:
    from .utils.export_manager import ExportManager
    __all__.append('ExportManager')
except ImportError as e:
    print(f"Warning: Could not import ExportManager: {e}")

# GUI imports (optional, for when PyQt5 is available)
GUI_AVAILABLE = False
try:
    from .gui.main_window import DRSMainWindow
    __all__.append('DRSMainWindow')
    GUI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: GUI not available: {e}")
    GUI_AVAILABLE = False