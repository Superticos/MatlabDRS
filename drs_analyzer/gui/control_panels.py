"""
Control panels for DRS spectroscopy analysis
Interactive parameter controls with real-time validation
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QGroupBox, QLabel, QPushButton, QCheckBox, QComboBox, 
    QSpinBox, QDoubleSpinBox, QSlider, QLineEdit, QTextEdit,
    QTabWidget, QScrollArea, QSplitter, QFrame, QButtonGroup,
    QProgressBar, QListWidget, QTreeWidget, QTreeWidgetItem
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QPixmap, QIcon, QPalette
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

from ..config.settings import AppSettings, ProcessingSettings, PeakDetectionSettings

class BaseControlPanel(QWidget):
    """Base class for control panels with common functionality"""
    
    # Common signals
    parameters_changed = pyqtSignal(dict)
    apply_requested = pyqtSignal(dict)
    reset_requested = pyqtSignal()
    
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Parameter storage
        self.current_parameters = {}
        self.default_parameters = {}
        
        # UI components
        self.main_layout = QVBoxLayout(self)
        self.scroll_area = QScrollArea()
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        
        # Setup UI
        self.init_ui()
        self.setup_connections()
        self.load_defaults()
    
    def init_ui(self):
        """Initialize the user interface"""
        # Make scrollable
        self.scroll_area.setWidget(self.content_widget)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarNever)
        
        self.main_layout.addWidget(self.scroll_area)
        
        # Add apply/reset buttons at bottom
        self.setup_action_buttons()
    
    def setup_action_buttons(self):
        """Setup apply and reset buttons"""
        button_layout = QHBoxLayout()
        
        self.apply_button = QPushButton("🔧 Apply")
        self.apply_button.setStyleSheet("""
            QPushButton {
                background-color: #2E86C1;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #3498DB;
            }
        """)
        
        self.reset_button = QPushButton("🔄 Reset")
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        
        self.main_layout.addLayout(button_layout)
    
    def setup_connections(self):
        """Setup signal connections"""
        self.apply_button.clicked.connect(self.apply_parameters)
        self.reset_button.clicked.connect(self.reset_parameters)
    
    def load_defaults(self):
        """Load default parameters - to be implemented by subclasses"""
        pass
    
    def apply_parameters(self):
        """Apply current parameters"""
        self.update_current_parameters()
        self.apply_requested.emit(self.current_parameters.copy())
    
    def reset_parameters(self):
        """Reset to default parameters"""
        self.current_parameters = self.default_parameters.copy()
        self.update_ui_from_parameters()
        self.reset_requested.emit()
    
    def update_current_parameters(self):
        """Update current parameters from UI - to be implemented by subclasses"""
        pass
    
    def update_ui_from_parameters(self):
        """Update UI from current parameters - to be implemented by subclasses"""
        pass
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get current parameters"""
        self.update_current_parameters()
        return self.current_parameters.copy()

class ProcessingControlPanel(BaseControlPanel):
    """Control panel for spectral processing parameters"""
    
    # Specific signals
    processing_requested = pyqtSignal(dict)
    
    def __init__(self, settings: AppSettings, parent=None):
        self.baseline_widgets = {}
        self.smoothing_widgets = {}
        self.normalization_widgets = {}
        self.derivative_widgets = {}
        
        super().__init__(settings, parent)
    
    def init_ui(self):
        """Initialize processing control UI"""
        super().init_ui()
        
        # Create processing groups
        self.create_baseline_group()
        self.create_smoothing_group()
        self.create_normalization_group()
        self.create_derivative_group()
        self.create_preview_group()
    
    def create_baseline_group(self):
        """Create baseline correction controls"""
        group = QGroupBox("📈 Baseline Correction")
        layout = QFormLayout(group)
        
        # Enable baseline correction
        self.baseline_widgets['enabled'] = QCheckBox("Enable Baseline Correction")
        self.baseline_widgets['enabled'].setChecked(True)
        layout.addRow(self.baseline_widgets['enabled'])
        
        # Method selection
        self.baseline_widgets['method'] = QComboBox()
        self.baseline_widgets['method'].addItems([
            'polynomial', 'asymmetric_least_squares', 'rolling_ball', 'snip'
        ])
        layout.addRow("Method:", self.baseline_widgets['method'])
        
        # Polynomial degree (for polynomial method)
        self.baseline_widgets['degree'] = QSpinBox()
        self.baseline_widgets['degree'].setRange(1, 10)
        self.baseline_widgets['degree'].setValue(2)
        layout.addRow("Polynomial Degree:", self.baseline_widgets['degree'])
        
        # Lambda parameter (for ALS method)
        self.baseline_widgets['lambda'] = QDoubleSpinBox()
        self.baseline_widgets['lambda'].setRange(1, 1e6)
        self.baseline_widgets['lambda'].setValue(1e4)
        self.baseline_widgets['lambda'].setDecimals(0)
        layout.addRow("Lambda (ALS):", self.baseline_widgets['lambda'])
        
        # P parameter (for ALS method)
        self.baseline_widgets['p'] = QDoubleSpinBox()
        self.baseline_widgets['p'].setRange(0.001, 0.1)
        self.baseline_widgets['p'].setValue(0.01)
        self.baseline_widgets['p'].setDecimals(3)
        layout.addRow("P parameter (ALS):", self.baseline_widgets['p'])
        
        # Rolling ball radius
        self.baseline_widgets['radius'] = QSpinBox()
        self.baseline_widgets['radius'].setRange(10, 200)
        self.baseline_widgets['radius'].setValue(50)
        layout.addRow("Rolling Ball Radius:", self.baseline_widgets['radius'])
        
        # SNIP iterations
        self.baseline_widgets['iterations'] = QSpinBox()
        self.baseline_widgets['iterations'].setRange(10, 100)
        self.baseline_widgets['iterations'].setValue(40)
        layout.addRow("SNIP Iterations:", self.baseline_widgets['iterations'])
        
        # Connect method change to update visibility
        self.baseline_widgets['method'].currentTextChanged.connect(self.update_baseline_visibility)
        
        self.content_layout.addWidget(group)
        self.update_baseline_visibility()
    
    def update_baseline_visibility(self):
        """Update visibility of baseline parameters based on method"""
        method = self.baseline_widgets['method'].currentText()
        
        # Hide all parameter-specific widgets
        self.baseline_widgets['degree'].setVisible(method == 'polynomial')
        self.baseline_widgets['lambda'].setVisible(method == 'asymmetric_least_squares')
        self.baseline_widgets['p'].setVisible(method == 'asymmetric_least_squares')
        self.baseline_widgets['radius'].setVisible(method == 'rolling_ball')
        self.baseline_widgets['iterations'].setVisible(method == 'snip')
        
        # Update labels accordingly
        for i in range(self.baseline_widgets['degree'].parent().layout().count()):
            item = self.baseline_widgets['degree'].parent().layout().itemAt(i)
            if item and item.widget():
                widget = item.widget()
                if isinstance(widget, QLabel):
                    text = widget.text()
                    if "Polynomial Degree" in text:
                        widget.setVisible(method == 'polynomial')
                    elif "Lambda" in text:
                        widget.setVisible(method == 'asymmetric_least_squares')
                    elif "P parameter" in text:
                        widget.setVisible(method == 'asymmetric_least_squares')
                    elif "Rolling Ball" in text:
                        widget.setVisible(method == 'rolling_ball')
                    elif "SNIP" in text:
                        widget.setVisible(method == 'snip')
    
    def create_smoothing_group(self):
        """Create smoothing controls"""
        group = QGroupBox("🌊 Smoothing")
        layout = QFormLayout(group)
        
        # Enable smoothing
        self.smoothing_widgets['enabled'] = QCheckBox("Enable Smoothing")
        self.smoothing_widgets['enabled'].setChecked(True)
        layout.addRow(self.smoothing_widgets['enabled'])
        
        # Method selection
        self.smoothing_widgets['method'] = QComboBox()
        self.smoothing_widgets['method'].addItems([
            'savgol', 'gaussian', 'moving_average', 'median'
        ])
        layout.addRow("Method:", self.smoothing_widgets['method'])
        
        # Window length (for Savitzky-Golay and moving average)
        self.smoothing_widgets['window_length'] = QSpinBox()
        self.smoothing_widgets['window_length'].setRange(3, 51)
        self.smoothing_widgets['window_length'].setValue(5)
        self.smoothing_widgets['window_length'].setSingleStep(2)  # Keep odd
        layout.addRow("Window Length:", self.smoothing_widgets['window_length'])
        
        # Polynomial order (for Savitzky-Golay)
        self.smoothing_widgets['polyorder'] = QSpinBox()
        self.smoothing_widgets['polyorder'].setRange(1, 5)
        self.smoothing_widgets['polyorder'].setValue(2)
        layout.addRow("Polynomial Order:", self.smoothing_widgets['polyorder'])
        
        # Sigma (for Gaussian)
        self.smoothing_widgets['sigma'] = QDoubleSpinBox()
        self.smoothing_widgets['sigma'].setRange(0.1, 10.0)
        self.smoothing_widgets['sigma'].setValue(1.0)
        self.smoothing_widgets['sigma'].setDecimals(1)
        layout.addRow("Sigma (Gaussian):", self.smoothing_widgets['sigma'])
        
        # Kernel size (for median)
        self.smoothing_widgets['kernel_size'] = QSpinBox()
        self.smoothing_widgets['kernel_size'].setRange(3, 21)
        self.smoothing_widgets['kernel_size'].setValue(3)
        self.smoothing_widgets['kernel_size'].setSingleStep(2)  # Keep odd
        layout.addRow("Kernel Size (Median):", self.smoothing_widgets['kernel_size'])
        
        # Connect method change
        self.smoothing_widgets['method'].currentTextChanged.connect(self.update_smoothing_visibility)
        
        self.content_layout.addWidget(group)
        self.update_smoothing_visibility()
    
    def update_smoothing_visibility(self):
        """Update visibility of smoothing parameters based on method"""
        method = self.smoothing_widgets['method'].currentText()
        
        self.smoothing_widgets['window_length'].setVisible(method in ['savgol', 'moving_average'])
        self.smoothing_widgets['polyorder'].setVisible(method == 'savgol')
        self.smoothing_widgets['sigma'].setVisible(method == 'gaussian')
        self.smoothing_widgets['kernel_size'].setVisible(method == 'median')
    
    def create_normalization_group(self):
        """Create normalization controls"""
        group = QGroupBox("📏 Normalization")
        layout = QFormLayout(group)
        
        # Enable normalization
        self.normalization_widgets['enabled'] = QCheckBox("Enable Normalization")
        self.normalization_widgets['enabled'].setChecked(True)
        layout.addRow(self.normalization_widgets['enabled'])
        
        # Method selection
        self.normalization_widgets['method'] = QComboBox()
        self.normalization_widgets['method'].addItems([
            'minmax', 'standard', 'robust', 'vector', 'snv', 'msc'
        ])
        layout.addRow("Method:", self.normalization_widgets['method'])
        
        # Feature range for MinMax
        minmax_layout = QHBoxLayout()
        self.normalization_widgets['min_range'] = QDoubleSpinBox()
        self.normalization_widgets['min_range'].setRange(-10, 10)
        self.normalization_widgets['min_range'].setValue(0)
        self.normalization_widgets['max_range'] = QDoubleSpinBox()
        self.normalization_widgets['max_range'].setRange(-10, 10)
        self.normalization_widgets['max_range'].setValue(1)
        
        minmax_layout.addWidget(QLabel("Min:"))
        minmax_layout.addWidget(self.normalization_widgets['min_range'])
        minmax_layout.addWidget(QLabel("Max:"))
        minmax_layout.addWidget(self.normalization_widgets['max_range'])
        
        range_widget = QWidget()
        range_widget.setLayout(minmax_layout)
        layout.addRow("Feature Range:", range_widget)
        
        # Connect method change
        self.normalization_widgets['method'].currentTextChanged.connect(self.update_normalization_visibility)
        
        self.content_layout.addWidget(group)
        self.update_normalization_visibility()
    
    def update_normalization_visibility(self):
        """Update visibility of normalization parameters"""
        method = self.normalization_widgets['method'].currentText()
        
        # Only show range for minmax
        range_visible = method == 'minmax'
        self.normalization_widgets['min_range'].setVisible(range_visible)
        self.normalization_widgets['max_range'].setVisible(range_visible)
    
    def create_derivative_group(self):
        """Create derivative controls"""
        group = QGroupBox("📐 Derivative")
        layout = QFormLayout(group)
        
        # Enable derivative
        self.derivative_widgets['enabled'] = QCheckBox("Enable Derivative")
        self.derivative_widgets['enabled'].setChecked(False)
        layout.addRow(self.derivative_widgets['enabled'])
        
        # Derivative order
        self.derivative_widgets['order'] = QSpinBox()
        self.derivative_widgets['order'].setRange(1, 3)
        self.derivative_widgets['order'].setValue(1)
        layout.addRow("Derivative Order:", self.derivative_widgets['order'])
        
        # Method for derivative
        self.derivative_widgets['method'] = QComboBox()
        self.derivative_widgets['method'].addItems(['savgol', 'gradient'])
        layout.addRow("Method:", self.derivative_widgets['method'])
        
        # Window length for Savitzky-Golay derivative
        self.derivative_widgets['window_length'] = QSpinBox()
        self.derivative_widgets['window_length'].setRange(5, 25)
        self.derivative_widgets['window_length'].setValue(7)
        self.derivative_widgets['window_length'].setSingleStep(2)
        layout.addRow("Window Length:", self.derivative_widgets['window_length'])
        
        # Polynomial order for Savitzky-Golay derivative
        self.derivative_widgets['polyorder'] = QSpinBox()
        self.derivative_widgets['polyorder'].setRange(2, 5)
        self.derivative_widgets['polyorder'].setValue(3)
        layout.addRow("Polynomial Order:", self.derivative_widgets['polyorder'])
        
        # Connect method change
        self.derivative_widgets['method'].currentTextChanged.connect(self.update_derivative_visibility)
        
        self.content_layout.addWidget(group)
        self.update_derivative_visibility()
    
    def update_derivative_visibility(self):
        """Update visibility of derivative parameters"""
        method = self.derivative_widgets['method'].currentText()
        
        savgol_visible = method == 'savgol'
        self.derivative_widgets['window_length'].setVisible(savgol_visible)
        self.derivative_widgets['polyorder'].setVisible(savgol_visible)
    
    def create_preview_group(self):
        """Create preview and batch processing controls"""
        group = QGroupBox("👁️ Preview & Batch")
        layout = QVBoxLayout(group)
        
        # Preview options
        preview_layout = QHBoxLayout()
        
        self.preview_button = QPushButton("👁️ Preview Changes")
        self.preview_button.setStyleSheet("""
            QPushButton {
                background-color: #F39C12;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #E67E22;
            }
        """)
        
        self.batch_process_button = QPushButton("⚡ Batch Process")
        self.batch_process_button.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        
        preview_layout.addWidget(self.preview_button)
        preview_layout.addWidget(self.batch_process_button)
        preview_layout.addStretch()
        
        layout.addLayout(preview_layout)
        
        # Processing order
        order_label = QLabel("Processing Order:")
        order_label.setFont(QFont("Arial", 9, QFont.Bold))
        layout.addWidget(order_label)
        
        order_text = QLabel("1. Baseline Correction → 2. Smoothing → 3. Normalization → 4. Derivative")
        order_text.setStyleSheet("color: #666; font-style: italic;")
        order_text.setWordWrap(True)
        layout.addWidget(order_text)
        
        self.content_layout.addWidget(group)
    
    def setup_connections(self):
        """Setup additional signal connections"""
        super().setup_connections()
        
        # Connect preview and batch buttons
        self.preview_button.clicked.connect(self.preview_processing)
        self.batch_process_button.clicked.connect(self.batch_process)
        
        # Connect parameter changes
        for widget_dict in [self.baseline_widgets, self.smoothing_widgets, 
                           self.normalization_widgets, self.derivative_widgets]:
            for widget in widget_dict.values():
                if hasattr(widget, 'valueChanged'):
                    widget.valueChanged.connect(self.on_parameter_changed)
                elif hasattr(widget, 'currentTextChanged'):
                    widget.currentTextChanged.connect(self.on_parameter_changed)
                elif hasattr(widget, 'toggled'):
                    widget.toggled.connect(self.on_parameter_changed)
    
    def on_parameter_changed(self):
        """Handle parameter changes"""
        self.update_current_parameters()
        self.parameters_changed.emit(self.current_parameters.copy())
    
    def preview_processing(self):
        """Preview processing with current parameters"""
        self.update_current_parameters()
        # Add preview logic here - could emit a special signal
        self.logger.info("Preview processing requested")
    
    def batch_process(self):
        """Apply batch processing with current parameters"""
        self.apply_parameters()
    
    def load_defaults(self):
        """Load default processing parameters"""
        self.default_parameters = {
            'baseline_enabled': True,
            'baseline_method': 'polynomial',
            'baseline_degree': 2,
            'baseline_lambda': 1e4,
            'baseline_p': 0.01,
            'baseline_radius': 50,
            'baseline_iterations': 40,
            
            'smoothing_enabled': True,
            'smoothing_method': 'savgol',
            'smoothing_window_length': 5,
            'smoothing_polyorder': 2,
            'smoothing_sigma': 1.0,
            'smoothing_kernel_size': 3,
            
            'normalization_enabled': True,
            'normalization_method': 'minmax',
            'normalization_min_range': 0,
            'normalization_max_range': 1,
            
            'derivative_enabled': False,
            'derivative_order': 1,
            'derivative_method': 'savgol',
            'derivative_window_length': 7,
            'derivative_polyorder': 3
        }
        
        self.current_parameters = self.default_parameters.copy()
    
    def update_current_parameters(self):
        """Update current parameters from UI"""
        # Baseline parameters
        self.current_parameters['baseline_enabled'] = self.baseline_widgets['enabled'].isChecked()
        self.current_parameters['baseline_method'] = self.baseline_widgets['method'].currentText()
        self.current_parameters['baseline_degree'] = self.baseline_widgets['degree'].value()
        self.current_parameters['baseline_lambda'] = self.baseline_widgets['lambda'].value()
        self.current_parameters['baseline_p'] = self.baseline_widgets['p'].value()
        self.current_parameters['baseline_radius'] = self.baseline_widgets['radius'].value()
        self.current_parameters['baseline_iterations'] = self.baseline_widgets['iterations'].value()
        
        # Smoothing parameters
        self.current_parameters['smoothing_enabled'] = self.smoothing_widgets['enabled'].isChecked()
        self.current_parameters['smoothing_method'] = self.smoothing_widgets['method'].currentText()
        self.current_parameters['smoothing_window_length'] = self.smoothing_widgets['window_length'].value()
        self.current_parameters['smoothing_polyorder'] = self.smoothing_widgets['polyorder'].value()
        self.current_parameters['smoothing_sigma'] = self.smoothing_widgets['sigma'].value()
        self.current_parameters['smoothing_kernel_size'] = self.smoothing_widgets['kernel_size'].value()
        
        # Normalization parameters
        self.current_parameters['normalization_enabled'] = self.normalization_widgets['enabled'].isChecked()
        self.current_parameters['normalization_method'] = self.normalization_widgets['method'].currentText()
        self.current_parameters['normalization_min_range'] = self.normalization_widgets['min_range'].value()
        self.current_parameters['normalization_max_range'] = self.normalization_widgets['max_range'].value()
        
        # Derivative parameters
        self.current_parameters['derivative_enabled'] = self.derivative_widgets['enabled'].isChecked()
        self.current_parameters['derivative_order'] = self.derivative_widgets['order'].value()
        self.current_parameters['derivative_method'] = self.derivative_widgets['method'].currentText()
        self.current_parameters['derivative_window_length'] = self.derivative_widgets['window_length'].value()
        self.current_parameters['derivative_polyorder'] = self.derivative_widgets['polyorder'].value()
    
    def update_ui_from_parameters(self):
        """Update UI from current parameters"""
        # Baseline
        self.baseline_widgets['enabled'].setChecked(self.current_parameters['baseline_enabled'])
        self.baseline_widgets['method'].setCurrentText(self.current_parameters['baseline_method'])
        self.baseline_widgets['degree'].setValue(self.current_parameters['baseline_degree'])
        self.baseline_widgets['lambda'].setValue(self.current_parameters['baseline_lambda'])
        self.baseline_widgets['p'].setValue(self.current_parameters['baseline_p'])
        self.baseline_widgets['radius'].setValue(self.current_parameters['baseline_radius'])
        self.baseline_widgets['iterations'].setValue(self.current_parameters['baseline_iterations'])
        
        # Smoothing
        self.smoothing_widgets['enabled'].setChecked(self.current_parameters['smoothing_enabled'])
        self.smoothing_widgets['method'].setCurrentText(self.current_parameters['smoothing_method'])
        self.smoothing_widgets['window_length'].setValue(self.current_parameters['smoothing_window_length'])
        self.smoothing_widgets['polyorder'].setValue(self.current_parameters['smoothing_polyorder'])
        self.smoothing_widgets['sigma'].setValue(self.current_parameters['smoothing_sigma'])
        self.smoothing_widgets['kernel_size'].setValue(self.current_parameters['smoothing_kernel_size'])
        
        # Normalization
        self.normalization_widgets['enabled'].setChecked(self.current_parameters['normalization_enabled'])
        self.normalization_widgets['method'].setCurrentText(self.current_parameters['normalization_method'])
        self.normalization_widgets['min_range'].setValue(self.current_parameters['normalization_min_range'])
        self.normalization_widgets['max_range'].setValue(self.current_parameters['normalization_max_range'])
        
        # Derivative
        self.derivative_widgets['enabled'].setChecked(self.current_parameters['derivative_enabled'])
        self.derivative_widgets['order'].setValue(self.current_parameters['derivative_order'])
        self.derivative_widgets['method'].setCurrentText(self.current_parameters['derivative_method'])
        self.derivative_widgets['window_length'].setValue(self.current_parameters['derivative_window_length'])
        self.derivative_widgets['polyorder'].setValue(self.current_parameters['derivative_polyorder'])
        
        # Update visibility
        self.update_baseline_visibility()
        self.update_smoothing_visibility()
        self.update_normalization_visibility()
        self.update_derivative_visibility()
    
    def get_baseline_settings(self) -> Dict[str, Any]:
        """Get baseline correction settings"""
        if not self.baseline_widgets['enabled'].isChecked():
            return {'method': None}
        
        method = self.baseline_widgets['method'].currentText()
        settings = {'method': method}
        
        if method == 'polynomial':
            settings['degree'] = self.baseline_widgets['degree'].value()
        elif method == 'asymmetric_least_squares':
            settings['lambda'] = self.baseline_widgets['lambda'].value()
            settings['p'] = self.baseline_widgets['p'].value()
        elif method == 'rolling_ball':
            settings['radius'] = self.baseline_widgets['radius'].value()
        elif method == 'snip':
            settings['iterations'] = self.baseline_widgets['iterations'].value()
        
        return settings
    
    def get_smoothing_settings(self) -> Dict[str, Any]:
        """Get smoothing settings"""
        if not self.smoothing_widgets['enabled'].isChecked():
            return {'method': None}
        
        method = self.smoothing_widgets['method'].currentText()
        settings = {'method': method}
        
        if method == 'savgol':
            settings['window_length'] = self.smoothing_widgets['window_length'].value()
            settings['polyorder'] = self.smoothing_widgets['polyorder'].value()
        elif method == 'gaussian':
            settings['sigma'] = self.smoothing_widgets['sigma'].value()
        elif method == 'moving_average':
            settings['window'] = self.smoothing_widgets['window_length'].value()
        elif method == 'median':
            settings['kernel_size'] = self.smoothing_widgets['kernel_size'].value()
        
        return settings
    
    def get_normalization_settings(self) -> Dict[str, Any]:
        """Get normalization settings"""
        if not self.normalization_widgets['enabled'].isChecked():
            return {'method': None}
        
        method = self.normalization_widgets['method'].currentText()
        settings = {'method': method}
        
        if method == 'minmax':
            settings['feature_range'] = (
                self.normalization_widgets['min_range'].value(),
                self.normalization_widgets['max_range'].value()
            )
        
        return settings

class PeakDetectionPanel(BaseControlPanel):
    """Control panel for peak detection parameters"""
    
    # Specific signals
    detection_requested = pyqtSignal(dict)
    peak_selected = pyqtSignal(int, int)  # spectrum_idx, peak_idx
    
    def __init__(self, settings: AppSettings, parent=None):
        self.detection_widgets = {}
        self.filter_widgets = {}
        self.clustering_widgets = {}
        
        # Peak results
        self.current_peaks = []
        self.peak_statistics = {}
        
        super().__init__(settings, parent)
    
    def init_ui(self):
        """Initialize peak detection UI"""
        super().init_ui()
        
        self.create_detection_group()
        self.create_filter_group()
        self.create_clustering_group()
        self.create_results_group()
    
    def create_detection_group(self):
        """Create peak detection method controls"""
        group = QGroupBox("🔍 Peak Detection")
        layout = QFormLayout(group)
        
        # Detection method
        self.detection_widgets['method'] = QComboBox()
        self.detection_widgets['method'].addItems([
            'scipy_peaks', 'cwt_peaks', 'derivative_peaks'
        ])
        layout.addRow("Method:", self.detection_widgets['method'])
        
        # Height threshold
        self.detection_widgets['height'] = QDoubleSpinBox()
        self.detection_widgets['height'].setRange(0.0, 1.0)
        self.detection_widgets['height'].setValue(0.1)
        self.detection_widgets['height'].setDecimals(3)
        layout.addRow("Height Threshold:", self.detection_widgets['height'])
        
        # Prominence threshold
        self.detection_widgets['prominence'] = QDoubleSpinBox()
        self.detection_widgets['prominence'].setRange(0.0, 1.0)
        self.detection_widgets['prominence'].setValue(0.05)
        self.detection_widgets['prominence'].setDecimals(3)
        layout.addRow("Prominence:", self.detection_widgets['prominence'])
        
        # Distance between peaks
        self.detection_widgets['distance'] = QSpinBox()
        self.detection_widgets['distance'].setRange(1, 100)
        self.detection_widgets['distance'].setValue(10)
        layout.addRow("Min Distance:", self.detection_widgets['distance'])
        
        # Width constraints
        width_layout = QHBoxLayout()
        self.detection_widgets['width_min'] = QSpinBox()
        self.detection_widgets['width_min'].setRange(1, 50)
        self.detection_widgets['width_min'].setValue(2)
        self.detection_widgets['width_max'] = QSpinBox()
        self.detection_widgets['width_max'].setRange(1, 100)
        self.detection_widgets['width_max'].setValue(20)
        
        width_layout.addWidget(QLabel("Min:"))
        width_layout.addWidget(self.detection_widgets['width_min'])
        width_layout.addWidget(QLabel("Max:"))
        width_layout.addWidget(self.detection_widgets['width_max'])
        
        width_widget = QWidget()
        width_widget.setLayout(width_layout)
        layout.addRow("Width Range:", width_widget)
        
        # CWT specific parameters
        self.detection_widgets['cwt_widths_min'] = QSpinBox()
        self.detection_widgets['cwt_widths_min'].setRange(1, 50)
        self.detection_widgets['cwt_widths_min'].setValue(1)
        
        self.detection_widgets['cwt_widths_max'] = QSpinBox()
        self.detection_widgets['cwt_widths_max'].setRange(1, 100)
        self.detection_widgets['cwt_widths_max'].setValue(20)
        
        cwt_layout = QHBoxLayout()
        cwt_layout.addWidget(QLabel("Min:"))
        cwt_layout.addWidget(self.detection_widgets['cwt_widths_min'])
        cwt_layout.addWidget(QLabel("Max:"))
        cwt_layout.addWidget(self.detection_widgets['cwt_widths_max'])
        
        cwt_widget = QWidget()
        cwt_widget.setLayout(cwt_layout)
        layout.addRow("CWT Width Range:", cwt_widget)
        
        # Signal-to-noise ratio for CWT
        self.detection_widgets['min_snr'] = QDoubleSpinBox()
        self.detection_widgets['min_snr'].setRange(0.1, 10.0)
        self.detection_widgets['min_snr'].setValue(1.0)
        layout.addRow("Min SNR (CWT):", self.detection_widgets['min_snr'])
        
        # Connect method change
        self.detection_widgets['method'].currentTextChanged.connect(self.update_detection_visibility)
        
        self.content_layout.addWidget(group)
        self.update_detection_visibility()
    
    def update_detection_visibility(self):
        """Update visibility based on detection method"""
        method = self.detection_widgets['method'].currentText()
        
        # CWT-specific parameters
        cwt_visible = method == 'cwt_peaks'
        self.detection_widgets['cwt_widths_min'].setVisible(cwt_visible)
        self.detection_widgets['cwt_widths_max'].setVisible(cwt_visible)
        self.detection_widgets['min_snr'].setVisible(cwt_visible)
    
    def create_filter_group(self):
        """Create peak filtering controls"""
        group = QGroupBox("🔧 Peak Filtering")
        layout = QFormLayout(group)
        
        # Enable filtering
        self.filter_widgets['enabled'] = QCheckBox("Enable Peak Filtering")
        self.filter_widgets['enabled'].setChecked(True)
        layout.addRow(self.filter_widgets['enabled'])
        
        # Intensity filter
        self.filter_widgets['min_intensity'] = QDoubleSpinBox()
        self.filter_widgets['min_intensity'].setRange(0.0, 1.0)
        self.filter_widgets['min_intensity'].setValue(0.05)
        self.filter_widgets['min_intensity'].setDecimals(3)
        layout.addRow("Min Intensity:", self.filter_widgets['min_intensity'])
        
        # Relative height filter
        self.filter_widgets['relative_height'] = QDoubleSpinBox()
        self.filter_widgets['relative_height'].setRange(0.0, 1.0)
        self.filter_widgets['relative_height'].setValue(0.1)
        self.filter_widgets['relative_height'].setDecimals(2)
        layout.addRow("Relative Height:", self.filter_widgets['relative_height'])
        
        # Wavelength range filter
        wl_layout = QHBoxLayout()
        self.filter_widgets['wl_min'] = QDoubleSpinBox()
        self.filter_widgets['wl_min'].setRange(0, 5000)
        self.filter_widgets['wl_min'].setValue(200)
        self.filter_widgets['wl_max'] = QDoubleSpinBox()
        self.filter_widgets['wl_max'].setRange(0, 5000)
        self.filter_widgets['wl_max'].setValue(2500)
        
        wl_layout.addWidget(QLabel("Min:"))
        wl_layout.addWidget(self.filter_widgets['wl_min'])
        wl_layout.addWidget(QLabel("Max:"))
        wl_layout.addWidget(self.filter_widgets['wl_max'])
        
        wl_widget = QWidget()
        wl_widget.setLayout(wl_layout)
        layout.addRow("Wavelength Range:", wl_widget)
        
        self.content_layout.addWidget(group)
    
    def create_clustering_group(self):
        """Create peak clustering controls"""
        group = QGroupBox("🎯 Peak Clustering")
        layout = QFormLayout(group)
        
        # Enable clustering
        self.clustering_widgets['enabled'] = QCheckBox("Enable Peak Clustering")
        self.clustering_widgets['enabled'].setChecked(False)
        layout.addRow(self.clustering_widgets['enabled'])
        
        # Clustering method
        self.clustering_widgets['method'] = QComboBox()
        self.clustering_widgets['method'].addItems(['dbscan', 'kmeans', 'hierarchical'])
        layout.addRow("Method:", self.clustering_widgets['method'])
        
        # Distance threshold
        self.clustering_widgets['distance_threshold'] = QDoubleSpinBox()
        self.clustering_widgets['distance_threshold'].setRange(1.0, 100.0)
        self.clustering_widgets['distance_threshold'].setValue(5.0)
        layout.addRow("Distance Threshold:", self.clustering_widgets['distance_threshold'])
        
        # Minimum cluster size
        self.clustering_widgets['min_cluster_size'] = QSpinBox()
        self.clustering_widgets['min_cluster_size'].setRange(2, 20)
        self.clustering_widgets['min_cluster_size'].setValue(3)
        layout.addRow("Min Cluster Size:", self.clustering_widgets['min_cluster_size'])
        
        self.content_layout.addWidget(group)
    
    def create_results_group(self):
        """Create peak results display"""
        group = QGroupBox("📊 Peak Results")
        layout = QVBoxLayout(group)
        
        # Summary statistics
        self.stats_label = QLabel("No peaks detected")
        self.stats_label.setStyleSheet("""
            QLabel {
                background-color: #F8F9FA;
                border: 1px solid #DEE2E6;
                border-radius: 4px;
                padding: 8px;
                font-family: monospace;
            }
        """)
        layout.addWidget(self.stats_label)
        
        # Peak list
        self.peak_list = QTreeWidget()
        self.peak_list.setHeaderLabels(['Spectrum', 'Peak #', 'Wavelength', 'Intensity', 'Width'])
        self.peak_list.setMaximumHeight(200)
        self.peak_list.itemClicked.connect(self.on_peak_selected)
        layout.addWidget(self.peak_list)
        
        # Export buttons
        export_layout = QHBoxLayout()
        
        self.export_peaks_button = QPushButton("📋 Export Peaks")
        self.export_peaks_button.clicked.connect(self.export_peaks)
        
        self.export_stats_button = QPushButton("📊 Export Statistics")
        self.export_stats_button.clicked.connect(self.export_statistics)
        
        export_layout.addWidget(self.export_peaks_button)
        export_layout.addWidget(self.export_stats_button)
        export_layout.addStretch()
        
        layout.addLayout(export_layout)
        
        self.content_layout.addWidget(group)
    
    def setup_connections(self):
        """Setup signal connections"""
        super().setup_connections()
        
        # Override apply button to emit detection_requested
        self.apply_button.clicked.disconnect()
        self.apply_button.clicked.connect(self.detect_peaks)
    
    def detect_peaks(self):
        """Emit peak detection request"""
        self.update_current_parameters()
        self.detection_requested.emit(self.current_parameters.copy())
    
    def on_peak_selected(self, item, column):
        """Handle peak selection in tree"""
        spectrum_idx = int(item.text(0)) - 1
        peak_idx = int(item.text(1)) - 1
        self.peak_selected.emit(spectrum_idx, peak_idx)
    
    def update_peak_results(self, peaks: List[List], statistics: Dict[str, Any]):
        """Update peak results display"""
        self.current_peaks = peaks
        self.peak_statistics = statistics
        
        # Update statistics label
        total_peaks = sum(len(spectrum_peaks) for spectrum_peaks in peaks)
        avg_peaks = total_peaks / len(peaks) if peaks else 0
        
        stats_text = f"""
        Total Peaks: {total_peaks}
        Average per Spectrum: {avg_peaks:.1f}
        Spectra with Peaks: {sum(1 for p in peaks if p)}
        """
        self.stats_label.setText(stats_text.strip())
        
        # Update peak list
        self.peak_list.clear()
        for spectrum_idx, spectrum_peaks in enumerate(peaks):
            for peak_idx, peak in enumerate(spectrum_peaks):
                item = QTreeWidgetItem([
                    str(spectrum_idx + 1),
                    str(peak_idx + 1),
                    f"{peak.wavelength:.2f}",
                    f"{peak.intensity:.4f}",
                    f"{peak.width:.2f}"
                ])
                self.peak_list.addTopLevelItem(item)
    
    def export_peaks(self):
        """Export peak data to file"""
        # Implementation for exporting peak data
        self.logger.info("Peak export requested")
    
    def export_statistics(self):
        """Export peak statistics to file"""
        # Implementation for exporting statistics
        self.logger.info("Statistics export requested")
    
    def load_defaults(self):
        """Load default peak detection parameters"""
        self.default_parameters = {
            'method': 'scipy_peaks',
            'height': 0.1,
            'prominence': 0.05,
            'distance': 10,
            'width_min': 2,
            'width_max': 20,
            'cwt_widths_min': 1,
            'cwt_widths_max': 20,
            'min_snr': 1.0,
            'filter_enabled': True,
            'min_intensity': 0.05,
            'relative_height': 0.1,
            'wl_min': 200,
            'wl_max': 2500,
            'clustering_enabled': False,
            'clustering_method': 'dbscan',
            'distance_threshold': 5.0,
            'min_cluster_size': 3
        }
        
        self.current_parameters = self.default_parameters.copy()
    
    def update_current_parameters(self):
        """Update current parameters from UI"""
        # Detection parameters
        self.current_parameters['method'] = self.detection_widgets['method'].currentText()
        self.current_parameters['height'] = self.detection_widgets['height'].value()
        self.current_parameters['prominence'] = self.detection_widgets['prominence'].value()
        self.current_parameters['distance'] = self.detection_widgets['distance'].value()
        self.current_parameters['width_min'] = self.detection_widgets['width_min'].value()
        self.current_parameters['width_max'] = self.detection_widgets['width_max'].value()
        self.current_parameters['cwt_widths_min'] = self.detection_widgets['cwt_widths_min'].value()
        self.current_parameters['cwt_widths_max'] = self.detection_widgets['cwt_widths_max'].value()
        self.current_parameters['min_snr'] = self.detection_widgets['min_snr'].value()
        
        # Filter parameters
        self.current_parameters['filter_enabled'] = self.filter_widgets['enabled'].isChecked()
        self.current_parameters['min_intensity'] = self.filter_widgets['min_intensity'].value()
        self.current_parameters['relative_height'] = self.filter_widgets['relative_height'].value()
        self.current_parameters['wl_min'] = self.filter_widgets['wl_min'].value()
        self.current_parameters['wl_max'] = self.filter_widgets['wl_max'].value()
        
        # Clustering parameters
        self.current_parameters['clustering_enabled'] = self.clustering_widgets['enabled'].isChecked()
        self.current_parameters['clustering_method'] = self.clustering_widgets['method'].currentText()
        self.current_parameters['distance_threshold'] = self.clustering_widgets['distance_threshold'].value()
        self.current_parameters['min_cluster_size'] = self.clustering_widgets['min_cluster_size'].value()

class StatisticsPanel(BaseControlPanel):
    """Control panel for statistical analysis"""
    
    # Specific signals
    analysis_requested = pyqtSignal(str, dict)  # analysis_type, parameters
    
    def __init__(self, settings: AppSettings, parent=None):
        self.pca_widgets = {}
        self.clustering_widgets = {}
        self.correlation_widgets = {}
        
        super().__init__(settings, parent)
    
    def init_ui(self):
        """Initialize statistics UI"""
        super().init_ui()
        
        # Create analysis tabs
        self.analysis_tabs = QTabWidget()
        
        self.create_pca_tab()
        self.create_clustering_tab()
        self.create_correlation_tab()
        self.create_general_tab()
        
        self.content_layout.addWidget(self.analysis_tabs)
    
    def create_pca_tab(self):
        """Create PCA analysis controls"""
        pca_widget = QWidget()
        layout = QFormLayout(pca_widget)
        
        # Number of components
        self.pca_widgets['n_components'] = QSpinBox()
        self.pca_widgets['n_components'].setRange(2, 20)
        self.pca_widgets['n_components'].setValue(5)
        layout.addRow("Number of Components:", self.pca_widgets['n_components'])
        
        # Include loadings
        self.pca_widgets['return_loadings'] = QCheckBox("Include Loadings")
        self.pca_widgets['return_loadings'].setChecked(True)
        layout.addRow(self.pca_widgets['return_loadings'])
        
        # Scaling method
        self.pca_widgets['scaling'] = QComboBox()
        self.pca_widgets['scaling'].addItems(['standard', 'minmax', 'robust', 'none'])
        layout.addRow("Data Scaling:", self.pca_widgets['scaling'])
        
        # Center data
        self.pca_widgets['center'] = QCheckBox("Center Data")
        self.pca_widgets['center'].setChecked(True)
        layout.addRow(self.pca_widgets['center'])
        
        # PCA run button
        self.pca_button = QPushButton("🔬 Run PCA Analysis")
        self.pca_button.setStyleSheet("""
            QPushButton {
                background-color: #3498DB;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #2980B9;
            }
        """)
        self.pca_button.clicked.connect(lambda: self.run_analysis('pca'))
        layout.addRow(self.pca_button)
        
        self.analysis_tabs.addTab(pca_widget, "📊 PCA")
    
    def create_clustering_tab(self):
        """Create clustering analysis controls"""
        clustering_widget = QWidget()
        layout = QFormLayout(clustering_widget)
        
        # Clustering method
        self.clustering_widgets['method'] = QComboBox()
        self.clustering_widgets['method'].addItems(['kmeans', 'dbscan', 'hierarchical'])
        self.clustering_widgets['method'].currentTextChanged.connect(self.update_clustering_visibility)
        layout.addRow("Method:", self.clustering_widgets['method'])
        
        # Number of clusters (for K-means, hierarchical)
        self.clustering_widgets['n_clusters'] = QSpinBox()
        self.clustering_widgets['n_clusters'].setRange(2, 20)
        self.clustering_widgets['n_clusters'].setValue(3)
        layout.addRow("Number of Clusters:", self.clustering_widgets['n_clusters'])
        
        # DBSCAN parameters
        self.clustering_widgets['eps'] = QDoubleSpinBox()
        self.clustering_widgets['eps'].setRange(0.1, 10.0)
        self.clustering_widgets['eps'].setValue(0.5)
        layout.addRow("Eps (DBSCAN):", self.clustering_widgets['eps'])
        
        self.clustering_widgets['min_samples'] = QSpinBox()
        self.clustering_widgets['min_samples'].setRange(2, 20)
        self.clustering_widgets['min_samples'].setValue(5)
        layout.addRow("Min Samples (DBSCAN):", self.clustering_widgets['min_samples'])
        
        # Hierarchical linkage
        self.clustering_widgets['linkage'] = QComboBox()
        self.clustering_widgets['linkage'].addItems(['ward', 'complete', 'average', 'single'])
        layout.addRow("Linkage (Hierarchical):", self.clustering_widgets['linkage'])
        
        # Preprocessing
        self.clustering_widgets['preprocessing'] = QComboBox()
        self.clustering_widgets['preprocessing'].addItems(['standard', 'minmax', 'robust', 'none'])
        layout.addRow("Data Scaling:", self.clustering_widgets['preprocessing'])
        
        # Clustering run button
        self.clustering_button = QPushButton("🎯 Run Clustering")
        self.clustering_button.setStyleSheet("""
            QPushButton {
                background-color: #E74C3C;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #C0392B;
            }
        """)
        self.clustering_button.clicked.connect(lambda: self.run_analysis('clustering'))
        layout.addRow(self.clustering_button)
        
        self.analysis_tabs.addTab(clustering_widget, "🎯 Clustering")
        self.update_clustering_visibility()
    
    def update_clustering_visibility(self):
        """Update clustering parameter visibility"""
        method = self.clustering_widgets['method'].currentText()
        
        # K-means and hierarchical need n_clusters
        self.clustering_widgets['n_clusters'].setVisible(method in ['kmeans', 'hierarchical'])
        
        # DBSCAN specific parameters
        self.clustering_widgets['eps'].setVisible(method == 'dbscan')
        self.clustering_widgets['min_samples'].setVisible(method == 'dbscan')
        
        # Hierarchical specific
        self.clustering_widgets['linkage'].setVisible(method == 'hierarchical')
    
    def create_correlation_tab(self):
        """Create correlation analysis controls"""
        correlation_widget = QWidget()
        layout = QFormLayout(correlation_widget)
        
        # Correlation method
        self.correlation_widgets['method'] = QComboBox()
        self.correlation_widgets['method'].addItems(['pearson', 'spearman', 'kendall'])
        layout.addRow("Method:", self.correlation_widgets['method'])
        
        # Correlation threshold for highlighting
        self.correlation_widgets['threshold'] = QDoubleSpinBox()
        self.correlation_widgets['threshold'].setRange(0.5, 1.0)
        self.correlation_widgets['threshold'].setValue(0.9)
        self.correlation_widgets['threshold'].setDecimals(2)
        layout.addRow("High Correlation Threshold:", self.correlation_widgets['threshold'])
        
        # Include p-values
        self.correlation_widgets['include_pvalues'] = QCheckBox("Include P-values")
        self.correlation_widgets['include_pvalues'].setChecked(False)
        layout.addRow(self.correlation_widgets['include_pvalues'])
        
        # Correlation run button
        self.correlation_button = QPushButton("🔗 Run Correlation Analysis")
        self.correlation_button.setStyleSheet("""
            QPushButton {
                background-color: #9B59B6;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #8E44AD;
            }
        """)
        self.correlation_button.clicked.connect(lambda: self.run_analysis('correlation'))
        layout.addRow(self.correlation_button)
        
        self.analysis_tabs.addTab(correlation_widget, "🔗 Correlation")
    
    def create_general_tab(self):
        """Create general statistics controls"""
        general_widget = QWidget()
        layout = QVBoxLayout(general_widget)
        
        # Quick analysis buttons
        quick_group = QGroupBox("Quick Analysis")
        quick_layout = QVBoxLayout(quick_group)
        
        self.basic_stats_button = QPushButton("📋 Basic Statistics")
        self.basic_stats_button.clicked.connect(self.run_basic_statistics)
        quick_layout.addWidget(self.basic_stats_button)
        
        self.outlier_detection_button = QPushButton("🎯 Outlier Detection")
        self.outlier_detection_button.clicked.connect(self.run_outlier_detection)
        quick_layout.addWidget(self.outlier_detection_button)
        
        self.tsne_button = QPushButton("📈 t-SNE Visualization")
        self.tsne_button.clicked.connect(self.run_tsne)
        quick_layout.addWidget(self.tsne_button)
        
        self.full_report_button = QPushButton("📊 Full Statistical Report")
        self.full_report_button.setStyleSheet("""
            QPushButton {
                background-color: #27AE60;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #229954;
            }
        """)
        self.full_report_button.clicked.connect(self.run_full_report)
        quick_layout.addWidget(self.full_report_button)
        
        layout.addWidget(quick_group)
        
        # Options
        options_group = QGroupBox("Analysis Options")
        options_layout = QFormLayout(options_group)
        
        self.save_results_check = QCheckBox("Auto-save Results")
        self.save_results_check.setChecked(False)
        options_layout.addRow(self.save_results_check)
        
        self.detailed_output_check = QCheckBox("Detailed Output")
        self.detailed_output_check.setChecked(True)
        options_layout.addRow(self.detailed_output_check)
        
        layout.addWidget(options_group)
        layout.addStretch()
        
        self.analysis_tabs.addTab(general_widget, "⚙️ General")
    
    def run_analysis(self, analysis_type: str):
        """Run specific analysis with current parameters"""
        if analysis_type == 'pca':
            params = {
                'n_components': self.pca_widgets['n_components'].value(),
                'return_loadings': self.pca_widgets['return_loadings'].isChecked()
            }
        elif analysis_type == 'clustering':
            params = {
                'method': self.clustering_widgets['method'].currentText(),
                'n_clusters': self.clustering_widgets['n_clusters'].value(),
                'eps': self.clustering_widgets['eps'].value(),
                'min_samples': self.clustering_widgets['min_samples'].value(),
                'linkage': self.clustering_widgets['linkage'].currentText()
            }
        elif analysis_type == 'correlation':
            params = {
                'method': self.correlation_widgets['method'].currentText()
            }
        else:
            params = {}
        
        self.analysis_requested.emit(analysis_type, params)
    
    def run_basic_statistics(self):
        """Run basic statistical analysis"""
        self.analysis_requested.emit('basic_statistics', {})
    
    def run_outlier_detection(self):
        """Run outlier detection"""
        self.analysis_requested.emit('outlier_detection', {'method': 'isolation_forest'})
    
    def run_tsne(self):
        """Run t-SNE analysis"""
        self.analysis_requested.emit('tsne', {'n_components': 2, 'perplexity': 30})
    
    def run_full_report(self):
        """Generate full statistical report"""
        self.analysis_requested.emit('full_report', {'include_all': True})
    
    def update_peak_statistics(self, statistics: Dict[str, Any]):
        """Update display with peak statistics"""
        # This could update a display showing peak-related statistics
        pass
    
    def load_defaults(self):
        """Load default statistics parameters"""
        self.default_parameters = {
            'pca_n_components': 5,
            'pca_return_loadings': True,
            'pca_scaling': 'standard',
            'clustering_method': 'kmeans',
            'clustering_n_clusters': 3,
            'clustering_eps': 0.5,
            'clustering_min_samples': 5,
            'clustering_linkage': 'ward',
            'clustering_preprocessing': 'standard',
            'correlation_method': 'pearson',
            'correlation_threshold': 0.9
        }
        
        self.current_parameters = self.default_parameters.copy()
    
    def update_current_parameters(self):
        """Update current parameters from UI"""
        # PCA parameters
        self.current_parameters['pca_n_components'] = self.pca_widgets['n_components'].value()
        self.current_parameters['pca_return_loadings'] = self.pca_widgets['return_loadings'].isChecked()
        self.current_parameters['pca_scaling'] = self.pca_widgets['scaling'].currentText()
        
        # Clustering parameters
        self.current_parameters['clustering_method'] = self.clustering_widgets['method'].currentText()
        self.current_parameters['clustering_n_clusters'] = self.clustering_widgets['n_clusters'].value()
        self.current_parameters['clustering_eps'] = self.clustering_widgets['eps'].value()
        self.current_parameters['clustering_min_samples'] = self.clustering_widgets['min_samples'].value()
        self.current_parameters['clustering_linkage'] = self.clustering_widgets['linkage'].currentText()
        self.current_parameters['clustering_preprocessing'] = self.clustering_widgets['preprocessing'].currentText()
        
        # Correlation parameters
        self.current_parameters['correlation_method'] = self.correlation_widgets['method'].currentText()
        self.current_parameters['correlation_threshold'] = self.correlation_widgets['threshold'].value()