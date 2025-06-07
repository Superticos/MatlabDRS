"""
Main window for DRS Analyzer Pro
Modern PyQt5-based interface with dockable panels
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
import traceback

from PyQt5.QtWidgets import (
    QMainWindow, QApplication, QVBoxLayout, QHBoxLayout, QWidget,
    QMenuBar, QStatusBar, QToolBar, QAction, QFileDialog, QMessageBox,
    QDockWidget, QTabWidget, QSplitter, QProgressBar, QLabel,
    QTextEdit, QCheckBox, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QIcon, QPixmap, QFont, QKeySequence

from ..config.settings import AppSettings
from ..core.data_processor import DataProcessor
from ..core.peak_detector import PeakDetector  
from ..core.statistics import DRSStatistics
from ..utils.file_io import FileManager
from ..utils.validators import DataValidator

from .plot_widgets import SpectraPlotWidget, StatisticsPlotWidget
from .control_panels import ProcessingControlPanel, PeakDetectionPanel, StatisticsPanel

class DataLoadWorker(QThread):
    """Worker thread for loading data"""
    data_loaded = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int)
    
    def __init__(self, file_path: str, file_manager: FileManager):
        super().__init__()
        self.file_path = file_path
        self.file_manager = file_manager
    
    def run(self):
        try:
            self.progress_updated.emit(25)
            data = self.file_manager.load_data(self.file_path)
            self.progress_updated.emit(75)
            
            # Validate data
            validator = DataValidator()
            validation = validator.validate_spectral_data(
                data['spectra'], data['wavelengths']
            )
            
            data['validation'] = validation
            self.progress_updated.emit(100)
            self.data_loaded.emit(data)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class ProcessingWorker(QThread):
    """Worker thread for data processing"""
    processing_completed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    progress_updated = pyqtSignal(int, str)
    
    def __init__(self, processor: DataProcessor, settings: Dict[str, Any]):
        super().__init__()
        self.processor = processor
        self.settings = settings
    
    def run(self):
        try:
            self.progress_updated.emit(20, "Applying baseline correction...")
            if self.settings.get('baseline_method'):
                self.processor.apply_baseline_correction(
                    method=self.settings['baseline_method'],
                    **self.settings.get('baseline_params', {})
                )
            
            self.progress_updated.emit(40, "Applying smoothing...")
            if self.settings.get('smoothing_method'):
                self.processor.apply_smoothing(
                    method=self.settings['smoothing_method'],
                    **self.settings.get('smoothing_params', {})
                )
            
            self.progress_updated.emit(60, "Applying normalization...")
            if self.settings.get('normalization_method'):
                self.processor.apply_normalization(
                    method=self.settings['normalization_method']
                )
            
            self.progress_updated.emit(80, "Calculating derivatives...")
            if self.settings.get('derivative_order', 0) > 0:
                self.processor.calculate_derivative(
                    order=self.settings['derivative_order']
                )
            
            self.progress_updated.emit(100, "Processing complete!")
            
            result = {
                'processed_data': self.processor.processed_data,
                'summary': self.processor.get_processing_summary()
            }
            
            self.processing_completed.emit(result)
            
        except Exception as e:
            self.error_occurred.emit(str(e))

class DRSMainWindow(QMainWindow):
    """
    Main application window for DRS Analyzer Pro
    Features modern interface with dockable panels and comprehensive tools
    """
    
    def __init__(self):
        super().__init__()
        
        # Initialize settings and components
        self.settings = AppSettings()
        self.file_manager = FileManager(self.settings)
        self.data_processor = DataProcessor(self.settings)
        self.peak_detector = PeakDetector(self.settings)
        self.statistics = DRSStatistics(self.settings)
        self.validator = DataValidator()
        
        # Data storage
        self.current_data = None
        self.processed_data = None
        
        # Worker threads
        self.load_worker = None
        self.processing_worker = None
        
        # UI components
        self.central_widget = None
        self.plot_tabs = None
        self.spectra_plot = None
        self.statistics_plot = None
        
        # Control panels
        self.processing_panel = None
        self.peak_panel = None
        self.stats_panel = None
        
        # Progress tracking
        self.progress_bar = None
        self.status_label = None
        
        # Initialize UI
        self.init_ui()
        self.create_menus()
        self.create_toolbars()
        self.create_status_bar()
        self.create_dock_widgets()
        self.setup_connections()
        self.load_window_settings()
        
        # Setup logging
        self.setup_logging()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("DRS Analyzer Pro initialized successfully")
    
    def init_ui(self):
        """Initialize the main user interface"""
        self.setWindowTitle("DRS Analyzer Pro v2.0")
        self.setMinimumSize(1200, 800)
        self.resize(1600, 1000)
        
        # Central widget with plot tabs
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Create main layout
        layout = QVBoxLayout(self.central_widget)
        
        # Create tab widget for different plot types
        self.plot_tabs = QTabWidget()
        layout.addWidget(self.plot_tabs)
        
        # Create plot widgets
        self.spectra_plot = SpectraPlotWidget()
        self.statistics_plot = StatisticsPlotWidget()
        
        # Add tabs
        self.plot_tabs.addTab(self.spectra_plot, "📊 Spectra")
        self.plot_tabs.addTab(self.statistics_plot, "📈 Statistics")
        
        # Style tabs
        self.plot_tabs.setTabPosition(QTabWidget.North)
        self.plot_tabs.setMovable(True)
        self.plot_tabs.setTabsClosable(False)
    
    def create_menus(self):
        """Create application menus"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('📁 &File')
        
        # Open file action
        open_action = QAction('&Open...', self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setStatusTip('Open spectral data file')
        open_action.triggered.connect(self.open_file)
        file_menu.addAction(open_action)
        
        # Recent files submenu
        self.recent_menu = file_menu.addMenu('Recent Files')
        self.update_recent_files_menu()
        
        file_menu.addSeparator()
        
        # Save actions
        save_action = QAction('&Save Results...', self)
        save_action.setShortcut(QKeySequence.Save)
        save_action.triggered.connect(self.save_results)
        file_menu.addAction(save_action)
        
        export_action = QAction('&Export Plot...', self)
        export_action.setShortcut('Ctrl+E')
        export_action.triggered.connect(self.export_plot)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction('E&xit', self)
        exit_action.setShortcut(QKeySequence.Quit)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Process menu
        process_menu = menubar.addMenu('⚙️ &Process')
        
        baseline_action = QAction('Baseline Correction', self)
        baseline_action.triggered.connect(self.apply_baseline_correction)
        process_menu.addAction(baseline_action)
        
        smooth_action = QAction('Smoothing', self)
        smooth_action.triggered.connect(self.apply_smoothing)
        process_menu.addAction(smooth_action)
        
        normalize_action = QAction('Normalization', self)
        normalize_action.triggered.connect(self.apply_normalization)
        process_menu.addAction(normalize_action)
        
        process_menu.addSeparator()
        
        reset_action = QAction('Reset Processing', self)
        reset_action.triggered.connect(self.reset_processing)
        process_menu.addAction(reset_action)
        
        # Analysis menu
        analysis_menu = menubar.addMenu('🔬 &Analysis')
        
        peaks_action = QAction('Detect Peaks', self)
        peaks_action.triggered.connect(self.detect_peaks)
        analysis_menu.addAction(peaks_action)
        
        pca_action = QAction('PCA Analysis', self)
        pca_action.triggered.connect(self.perform_pca)
        analysis_menu.addAction(pca_action)
        
        stats_action = QAction('Statistical Report', self)
        stats_action.triggered.connect(self.generate_statistics_report)
        analysis_menu.addAction(stats_action)
        
        # View menu
        view_menu = menubar.addMenu('👁️ &View')
        
        # Theme submenu
        theme_menu = view_menu.addMenu('Theme')
        for theme in self.settings.AVAILABLE_THEMES:
            theme_action = QAction(theme.replace('_', ' ').title(), self)
            theme_action.triggered.connect(lambda checked, t=theme: self.change_theme(t))
            theme_menu.addAction(theme_action)
        
        view_menu.addSeparator()
        
        # Panel toggles (will be populated when dock widgets are created)
        self.panels_menu = view_menu.addMenu('Panels')
        
        # Help menu
        help_menu = menubar.addMenu('❓ &Help')
        
        about_action = QAction('&About', self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbars(self):
        """Create application toolbars"""
        # Main toolbar
        main_toolbar = self.addToolBar('Main')
        main_toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        
        # File operations
        open_action = QAction('📂\nOpen', self)
        open_action.triggered.connect(self.open_file)
        main_toolbar.addAction(open_action)
        
        save_action = QAction('💾\nSave', self)
        save_action.triggered.connect(self.save_results)
        main_toolbar.addAction(save_action)
        
        main_toolbar.addSeparator()
        
        # Processing operations
        process_action = QAction('⚙️\nProcess', self)
        process_action.triggered.connect(self.quick_process)
        main_toolbar.addAction(process_action)
        
        reset_action = QAction('🔄\nReset', self)
        reset_action.triggered.connect(self.reset_processing)
        main_toolbar.addAction(reset_action)
        
        main_toolbar.addSeparator()
        
        # Analysis operations
        peaks_action = QAction('🔍\nPeaks', self)
        peaks_action.triggered.connect(self.detect_peaks)
        main_toolbar.addAction(peaks_action)
        
        stats_action = QAction('📊\nStats', self)
        stats_action.triggered.connect(self.perform_pca)
        main_toolbar.addAction(stats_action)
    
    def create_status_bar(self):
        """Create status bar with progress indicator"""
        status_bar = self.statusBar()
        
        # Status label
        self.status_label = QLabel("Ready")
        status_bar.addWidget(self.status_label)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setMaximumWidth(200)
        status_bar.addPermanentWidget(self.progress_bar)
        
        # Data info label
        self.data_info_label = QLabel("")
        status_bar.addPermanentWidget(self.data_info_label)
    
    def create_dock_widgets(self):
        """Create dockable control panels"""
        # Processing control panel
        self.processing_dock = QDockWidget("⚙️ Processing Controls", self)
        self.processing_panel = ProcessingControlPanel(self.settings)
        self.processing_dock.setWidget(self.processing_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.processing_dock)
        
        # Peak detection panel
        self.peak_dock = QDockWidget("🔍 Peak Detection", self)
        self.peak_panel = PeakDetectionPanel(self.settings)
        self.peak_dock.setWidget(self.peak_panel)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.peak_dock)
        
        # Statistics panel
        self.stats_dock = QDockWidget("📊 Statistics", self)
        self.stats_panel = StatisticsPanel(self.settings)
        self.stats_dock.setWidget(self.stats_panel)
        self.addDockWidget(Qt.RightDockWidgetArea, self.stats_dock)
        
        # Log panel
        self.log_dock = QDockWidget("📋 Log", self)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setReadOnly(True)
        self.log_dock.setWidget(self.log_text)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.log_dock)
        
        # Add panel toggle actions to view menu
        for dock in [self.processing_dock, self.peak_dock, self.stats_dock, self.log_dock]:
            self.panels_menu.addAction(dock.toggleViewAction())
    
    def setup_connections(self):
        """Setup signal-slot connections"""
        # Processing panel connections
        self.processing_panel.processing_requested.connect(self.apply_processing)
        self.processing_panel.reset_requested.connect(self.reset_processing)
        
        # Peak detection panel connections
        self.peak_panel.detection_requested.connect(self.detect_peaks_with_params)
        
        # Statistics panel connections
        self.stats_panel.analysis_requested.connect(self.perform_analysis)
        
        # Plot tab changes
        self.plot_tabs.currentChanged.connect(self.on_tab_changed)
    
    def setup_logging(self):
        """Setup logging to display in log panel"""
        # Create custom handler for log panel
        class LogPanelHandler(logging.Handler):
            def __init__(self, log_widget):
                super().__init__()
                self.log_widget = log_widget
            
            def emit(self, record):
                msg = self.format(record)
                self.log_widget.append(msg)
                # Auto-scroll to bottom
                scrollbar = self.log_widget.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
        
        # Setup logging
        logger = logging.getLogger()
        handler = LogPanelHandler(self.log_text)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    def open_file(self):
        """Open file dialog and load spectral data"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Spectral Data",
            self.settings.get('last_directory', str(Path.home())),
            "All Supported (*.csv *.txt *.xlsx *.xls *.h5 *.hdf5 *.json);;"
            "CSV Files (*.csv *.txt);;"
            "Excel Files (*.xlsx *.xls);;"
            "HDF5 Files (*.h5 *.hdf5);;"
            "JSON Files (*.json);;"
            "All Files (*)"
        )
        
        if file_path:
            self.load_data_file(file_path)
    
    def load_data_file(self, file_path: str):
        """Load data from file using worker thread"""
        try:
            # Update last directory
            self.settings.set('last_directory', str(Path(file_path).parent))
            
            # Show progress
            self.show_progress("Loading data...")
            
            # Create and start worker thread
            self.load_worker = DataLoadWorker(file_path, self.file_manager)
            self.load_worker.data_loaded.connect(self.on_data_loaded)
            self.load_worker.error_occurred.connect(self.on_load_error)
            self.load_worker.progress_updated.connect(self.update_progress)
            self.load_worker.start()
            
        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Error", f"Failed to start data loading: {e}")
    
    def on_data_loaded(self, data: Dict[str, Any]):
        """Handle successful data loading"""
        try:
            self.current_data = data
            
            # Load data into processors
            self.data_processor.load_data(
                data['spectra'], 
                data['wavelengths'], 
                data.get('metadata', {})
            )
            
            self.statistics.load_data(
                data['spectra'],
                data['wavelengths'],
                data.get('metadata', {})
            )
            
            # Update plots
            self.spectra_plot.update_data(
                data['spectra'],
                data['wavelengths']
            )
            
            # Update status
            n_spectra = data['spectra'].shape[0]
            n_wavelengths = len(data['wavelengths'])
            wl_range = f"{data['wavelengths'].min():.1f}-{data['wavelengths'].max():.1f}"
            
            self.data_info_label.setText(
                f"📊 {n_spectra} spectra, {n_wavelengths} points, {wl_range} nm"
            )
            
            # Add to recent files
            file_path = data.get('metadata', {}).get('filename', 'Unknown')
            if 'filename' in data.get('metadata', {}):
                self.settings.add_recent_file(file_path)
                self.update_recent_files_menu()
            
            # Show validation results
            if 'validation' in data:
                validation = data['validation']
                if not validation['is_valid']:
                    QMessageBox.warning(
                        self, 
                        "Data Validation Warning",
                        f"Data validation issues found:\n\n" + 
                        "\n".join(validation['errors'][:5])  # Show first 5 errors
                    )
            
            self.hide_progress()
            self.status_label.setText("Data loaded successfully")
            
            logging.getLogger(__name__).info(
                f"Loaded {n_spectra} spectra from {file_path}"
            )
            
        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Error", f"Failed to process loaded data: {e}")
    
    def on_load_error(self, error: str):
        """Handle data loading error"""
        self.hide_progress()
        QMessageBox.critical(self, "Loading Error", f"Failed to load data:\n\n{error}")
    
    def apply_processing(self, settings: Dict[str, Any]):
        """Apply processing with given settings"""
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "No data loaded")
            return
        
        try:
            self.show_progress("Processing data...")
            
            # Create and start processing worker
            self.processing_worker = ProcessingWorker(self.data_processor, settings)
            self.processing_worker.processing_completed.connect(self.on_processing_completed)
            self.processing_worker.error_occurred.connect(self.on_processing_error)
            self.processing_worker.progress_updated.connect(self.update_progress_with_text)
            self.processing_worker.start()
            
        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Error", f"Failed to start processing: {e}")
    
    def on_processing_completed(self, result: Dict[str, Any]):
        """Handle completed processing"""
        try:
            self.processed_data = result['processed_data']
            
            # Update spectra plot with processed data
            self.spectra_plot.update_processed_data(
                self.processed_data,
                self.data_processor.wavelengths
            )
            
            self.hide_progress()
            self.status_label.setText("Processing completed")
            
            logging.getLogger(__name__).info("Data processing completed successfully")
            
        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Error", f"Failed to handle processing results: {e}")
    
    def on_processing_error(self, error: str):
        """Handle processing error"""
        self.hide_progress()
        QMessageBox.critical(self, "Processing Error", f"Processing failed:\n\n{error}")
    
    def detect_peaks_with_params(self, params: Dict[str, Any]):
        """Detect peaks with given parameters"""
        if self.processed_data is None:
            QMessageBox.warning(self, "Warning", "No processed data available")
            return
        
        try:
            self.show_progress("Detecting peaks...")
            
            # Use processed data if available, otherwise raw data
            data_to_use = self.processed_data if self.processed_data is not None else self.current_data['spectra']
            
            peaks = self.peak_detector.detect_peaks(
                data_to_use,
                self.data_processor.wavelengths,
                **params
            )
            
            # Update plots with peaks
            self.spectra_plot.update_peaks(peaks)
            
            # Update statistics
            peak_stats = self.peak_detector.get_peak_statistics()
            self.stats_panel.update_peak_statistics(peak_stats)
            
            self.hide_progress()
            self.status_label.setText(f"Detected {sum(len(p) for p in peaks)} peaks")
            
            logging.getLogger(__name__).info(f"Peak detection completed: {sum(len(p) for p in peaks)} peaks found")
            
        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Error", f"Peak detection failed:\n\n{e}")
    
    def perform_analysis(self, analysis_type: str, params: Dict[str, Any]):
        """Perform statistical analysis"""
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "No data loaded")
            return
        
        try:
            self.show_progress(f"Performing {analysis_type}...")
            
            if analysis_type == "pca":
                result = self.statistics.perform_pca(**params)
                self.statistics_plot.update_pca_results(result)
                
            elif analysis_type == "clustering":
                result = self.statistics.perform_clustering(**params)
                self.statistics_plot.update_clustering_results(result)
                
            elif analysis_type == "correlation":
                result = self.statistics.calculate_correlation_matrix(**params)
                self.statistics_plot.update_correlation_results(result)
                
            self.hide_progress()
            self.status_label.setText(f"{analysis_type.upper()} analysis completed")
            
            logging.getLogger(__name__).info(f"{analysis_type} analysis completed")
            
        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Error", f"{analysis_type} analysis failed:\n\n{e}")
    
    def show_progress(self, message: str):
        """Show progress bar with message"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.status_label.setText(message)
    
    def update_progress(self, value: int):
        """Update progress bar value"""
        self.progress_bar.setValue(value)
    
    def update_progress_with_text(self, value: int, text: str):
        """Update progress bar value and text"""
        self.progress_bar.setValue(value)
        self.status_label.setText(text)
    
    def hide_progress(self):
        """Hide progress bar"""
        self.progress_bar.setVisible(False)
        QTimer.singleShot(2000, lambda: self.status_label.setText("Ready"))
    
    # Quick action methods
    def quick_process(self):
        """Apply quick processing with default settings"""
        settings = {
            'baseline_method': 'polynomial',
            'baseline_params': {'degree': 2},
            'smoothing_method': 'savgol',
            'smoothing_params': {'window_length': 5},
            'normalization_method': 'minmax'
        }
        self.apply_processing(settings)
    
    def reset_processing(self):
        """Reset to original data"""
        if self.data_processor.raw_data is not None:
            self.data_processor.reset_processing()
            self.spectra_plot.update_processed_data(
                self.data_processor.processed_data,
                self.data_processor.wavelengths
            )
            self.status_label.setText("Reset to original data")
    
    def detect_peaks(self):
        """Detect peaks with default settings"""
        params = {
            'method': 'scipy_peaks',
            'height': 0.1,
            'prominence': 0.05
        }
        self.detect_peaks_with_params(params)
    
    def perform_pca(self):
        """Perform PCA analysis"""
        params = {'n_components': 5}
        self.perform_analysis('pca', params)
    
    def generate_statistics_report(self):
        """Generate comprehensive statistics report"""
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "No data loaded")
            return
        
        try:
            self.show_progress("Generating statistics report...")
            report = self.statistics.generate_statistical_report()
            
            # Display report in statistics plot
            self.statistics_plot.update_report(report)
            
            # Switch to statistics tab
            self.plot_tabs.setCurrentWidget(self.statistics_plot)
            
            self.hide_progress()
            self.status_label.setText("Statistics report generated")
            
        except Exception as e:
            self.hide_progress()
            QMessageBox.critical(self, "Error", f"Failed to generate report:\n\n{e}")
    
    # Menu action methods
    def save_results(self):
        """Save analysis results"""
        if self.current_data is None:
            QMessageBox.warning(self, "Warning", "No data to save")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Results",
            "",
            "HDF5 Files (*.h5);; JSON Files (*.json);; Excel Files (*.xlsx)"
        )
        
        if file_path:
            try:
                # Prepare data for saving
                save_data = {
                    'spectra': self.current_data['spectra'],
                    'wavelengths': self.current_data['wavelengths'],
                    'metadata': self.current_data.get('metadata', {})
                }
                
                if self.processed_data is not None:
                    save_data['processed_spectra'] = self.processed_data
                    save_data['processing_history'] = self.data_processor.get_processing_summary()
                
                # Save using file manager
                success = self.file_manager.save_data(save_data, file_path)
                
                if success:
                    self.status_label.setText("Results saved successfully")
                else:
                    QMessageBox.warning(self, "Warning", "Failed to save results")
                    
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save results:\n\n{e}")
    
    def export_plot(self):
        """Export current plot"""
        current_plot = self.plot_tabs.currentWidget()
        if hasattr(current_plot, 'export_plot'):
            current_plot.export_plot()
        else:
            QMessageBox.information(self, "Info", "Current plot does not support export")
    
    def change_theme(self, theme_name: str):
        """Change application theme"""
        try:
            import qt_material
            qt_material.apply_stylesheet(QApplication.instance(), theme=theme_name)
            self.settings.set('theme', theme_name)
            self.status_label.setText(f"Theme changed to {theme_name}")
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to change theme: {e}")
    
    def update_recent_files_menu(self):
        """Update recent files menu"""
        self.recent_menu.clear()
        recent_files = self.settings.get_recent_files()
        
        for file_path in recent_files[:10]:  # Show only last 10
            action = QAction(Path(file_path).name, self)
            action.setStatusTip(file_path)
            action.triggered.connect(lambda checked, path=file_path: self.load_data_file(path))
            self.recent_menu.addAction(action)
        
        if recent_files:
            self.recent_menu.addSeparator()
            clear_action = QAction("Clear Recent Files", self)
            clear_action.triggered.connect(self.clear_recent_files)
            self.recent_menu.addAction(clear_action)
    
    def clear_recent_files(self):
        """Clear recent files list"""
        self.settings.clear_recent_files()
        self.update_recent_files_menu()
    
    def on_tab_changed(self, index: int):
        """Handle tab changes"""
        current_widget = self.plot_tabs.widget(index)
        tab_name = self.plot_tabs.tabText(index)
        self.status_label.setText(f"Switched to {tab_name}")
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About DRS Analyzer Pro",
            """
            <h3>DRS Analyzer Pro v2.0</h3>
            <p>Professional Diffuse Reflectance Spectroscopy Analysis Software</p>
            
            <p><b>Features:</b></p>
            <ul>
            <li>Advanced spectral processing</li>
            <li>Intelligent peak detection</li>
            <li>Statistical analysis (PCA, clustering)</li>
            <li>Modern PyQt5 interface</li>
            <li>Multiple file format support</li>
            </ul>
            
            <p><b>Developed by:</b> DRS Analysis Team</p>
            <p><b>Contact:</b> contact@drs-analyzer.com</p>
            """
        )
    
    def load_window_settings(self):
        """Load window geometry and state"""
        qt_settings = QSettings('DRSAnalyzer', 'MainWindow')
        
        # Restore geometry
        geometry = qt_settings.value('geometry')
        if geometry:
            self.restoreGeometry(geometry)
        
        # Restore window state (dock positions, etc.)
        state = qt_settings.value('windowState')
        if state:
            self.restoreState(state)
    
    def save_window_settings(self):
        """Save window geometry and state"""
        qt_settings = QSettings('DRSAnalyzer', 'MainWindow')
        qt_settings.setValue('geometry', self.saveGeometry())
        qt_settings.setValue('windowState', self.saveState())
    
    def closeEvent(self, event):
        """Handle application close"""
        try:
            # Save window settings
            self.save_window_settings()
            
            # Stop any running workers
            if self.load_worker and self.load_worker.isRunning():
                self.load_worker.terminate()
                self.load_worker.wait()
            
            if self.processing_worker and self.processing_worker.isRunning():
                self.processing_worker.terminate()
                self.processing_worker.wait()
            
            # Accept the close event
            event.accept()
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Error during shutdown: {e}")
            event.accept()  # Force close anyway

# Additional convenience methods for processing
    def apply_baseline_correction(self):
        """Apply baseline correction with current settings"""
        settings = self.processing_panel.get_baseline_settings()
        self.apply_processing({'baseline_method': settings['method'], 'baseline_params': settings})
    
    def apply_smoothing(self):
        """Apply smoothing with current settings"""
        settings = self.processing_panel.get_smoothing_settings()
        self.apply_processing({'smoothing_method': settings['method'], 'smoothing_params': settings})
    
    def apply_normalization(self):
        """Apply normalization with current settings"""
        settings = self.processing_panel.get_normalization_settings()
        self.apply_processing({'normalization_method': settings['method']})

def main():
    """Main entry point for standalone execution"""
    app = QApplication(sys.argv)
    
    # Apply theme
    try:
        import qt_material
        qt_material.apply_stylesheet(app, theme='dark_teal.xml')
    except ImportError:
        pass  # Use default theme if qt_material not available
    
    window = DRSMainWindow()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()