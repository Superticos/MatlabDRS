"""
Main application entry point for DRS Analyzer
"""
import sys
import logging
import os
from pathlib import Path

# Check if we're in a headless environment
HEADLESS = os.environ.get('DISPLAY') is None

# Only import PyQt5 if we have a display
if not HEADLESS:
    try:
        from PyQt5.QtWidgets import QApplication, QMessageBox
        from PyQt5.QtCore import Qt
        QT_AVAILABLE = True
    except ImportError:
        QT_AVAILABLE = False
        HEADLESS = True
else:
    QT_AVAILABLE = False

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import GUI components with error handling
GUI_AVAILABLE = False
if not HEADLESS and QT_AVAILABLE:
    try:
        from drs_analyzer.gui.main_window import DRSMainWindow
        GUI_AVAILABLE = True
    except ImportError as e:
        print(f"GUI components not available: {e}")
        DRSMainWindow = None

from drs_analyzer.config.settings import AppSettings

# Try to import logger with fallback
try:
    from drs_analyzer.utils.logger import setup_logging
except ImportError:
    def setup_logging(level=logging.INFO):
        logging.basicConfig(level=level)
        return logging.getLogger(__name__)

def run_headless_demo():
    """Run a simple demo in headless mode"""
    print("=" * 50)
    print("DRS Analyzer - Headless Demo")
    print("=" * 50)
    
    try:
        # Setup logging
        logger = setup_logging(level=logging.INFO)
        logger.info("Starting DRS Analyzer in headless mode")
        
        # Test core components
        from drs_analyzer.core.data_loader import DataLoader
        from drs_analyzer.core.data_processor import DataProcessor
        
        print("✓ Core components loaded successfully")
        
        # Create instances
        loader = DataLoader()
        processor = DataProcessor()
        settings = AppSettings()
        
        print("✓ Components instantiated successfully")
        print(f"✓ Settings loaded: {len(settings._settings)} configuration items")
        
        # Show available functionality
        print("\nAvailable core functionality:")
        print("- Data loading from multiple formats (.txt, .csv, .mat, .xlsx)")
        print("- Advanced spectral processing and analysis")
        print("- Statistical analysis and PCA")
        print("- Data export capabilities")
        
        print("\nTo use the full GUI interface:")
        print("1. Install display server (X11, VNC, etc.)")
        print("2. Set DISPLAY environment variable")
        print("3. Ensure PyQt5 dependencies are available")
        
        return 0
        
    except Exception as e:
        print(f"✗ Error in headless demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

def main():
    """Main application entry point"""
    
    # If we're in a headless environment, run demo
    if HEADLESS or not QT_AVAILABLE:
        print("Running in headless environment...")
        return run_headless_demo()
    
    if not GUI_AVAILABLE:
        print("Error: GUI components are not available due to missing dependencies.")
        print("Please install missing dependencies:")
        print("pip install h5py opencv-python pyqtgraph")
        print("\nOr run in headless mode for core functionality testing.")
        return 1
    
    try:
        # Setup logging first
        logger = setup_logging(level=logging.INFO)
        logger.info("Starting DRS Analyzer application")
        
        # Create QApplication
        app = QApplication(sys.argv)
        app.setApplicationName("DRS Analyzer")
        app.setApplicationVersion("1.0.0")
        app.setOrganizationName("Spectroscopy Lab")
        
        # Set application icon if available
        icon_path = Path(__file__).parent / "icons" / "app_icon.png"
        if icon_path.exists():
            from PyQt5.QtGui import QIcon
            app.setWindowIcon(QIcon(str(icon_path)))
        
        # Load settings
        try:
            settings = AppSettings()
            logger.info("Settings loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            settings = AppSettings()  # Use defaults
        
        # Create main window
        try:
            main_window = DRSMainWindow()
            main_window.show()
            logger.info("Main window created and shown")
        except Exception as e:
            logger.error(f"Failed to create main window: {e}")
            QMessageBox.critical(
                None, 
                "Startup Error",
                f"Failed to start DRS Analyzer:\n{str(e)}\n\nCheck the log for details."
            )
            return 1
        
        logger.info("Application started successfully")
        
        # Start event loop
        result = app.exec_()
        logger.info("Application closed")
        return result
        
    except Exception as e:
        # Last resort error handling
        print(f"Critical error during startup: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        
        if not HEADLESS:
            try:
                app = QApplication.instance()
                if app is None:
                    app = QApplication(sys.argv)
                QMessageBox.critical(
                    None,
                    "Critical Error", 
                    f"Critical error during startup:\n{str(e)}"
                )
            except:
                pass
            
        return 1

if __name__ == "__main__":
    sys.exit(main())