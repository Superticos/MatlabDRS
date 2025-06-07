"""
Settings dialog for DRS Analyzer
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QTabWidget,
                           QWidget, QLabel, QLineEdit, QSpinBox, QDoubleSpinBox,
                           QComboBox, QCheckBox, QPushButton, QGroupBox,
                           QFormLayout, QMessageBox)
from PyQt5.QtCore import Qt
from drs_analyzer.config.settings import AppSettings

class SettingsDialog(QDialog):
    """Settings configuration dialog"""
    
    def __init__(self, settings: AppSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.temp_settings = settings.get_all_settings().copy()
        
        self.setWindowTitle("DRS Analyzer - Settings")
        self.setModal(True)
        self.resize(500, 400)
        
        self.init_ui()
        self.load_current_settings()
    
    def init_ui(self):
        """Initialize the user interface"""
        layout = QVBoxLayout(self)
        
        # Create tab widget
        tab_widget = QTabWidget()
        
        # Processing tab
        tab_widget.addTab(self.create_processing_tab(), "Processing")
        
        # Plotting tab
        tab_widget.addTab(self.create_plotting_tab(), "Plotting")
        
        # GUI tab
        tab_widget.addTab(self.create_gui_tab(), "Interface")
        
        layout.addWidget(tab_widget)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.ok_button = QPushButton("OK")
        self.cancel_button = QPushButton("Cancel")
        self.apply_button = QPushButton("Apply")
        self.reset_button = QPushButton("Reset to Defaults")
        
        self.ok_button.clicked.connect(self.accept_settings)
        self.cancel_button.clicked.connect(self.reject)
        self.apply_button.clicked.connect(self.apply_settings)
        self.reset_button.clicked.connect(self.reset_to_defaults)
        
        button_layout.addWidget(self.reset_button)
        button_layout.addStretch()
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.cancel_button)
        button_layout.addWidget(self.ok_button)
        
        layout.addLayout(button_layout)
    
    def create_processing_tab(self) -> QWidget:
        """Create processing settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Baseline correction group
        baseline_group = QGroupBox("Baseline Correction")
        baseline_layout = QFormLayout(baseline_group)
        
        self.baseline_method = QComboBox()
        self.baseline_method.addItems(['polynomial', 'als', 'rolling_ball', 'snip'])
        baseline_layout.addRow("Method:", self.baseline_method)
        
        self.baseline_degree = QSpinBox()
        self.baseline_degree.setRange(1, 10)
        baseline_layout.addRow("Polynomial Degree:", self.baseline_degree)
        
        layout.addWidget(baseline_group)
        
        # Smoothing group
        smoothing_group = QGroupBox("Smoothing")
        smoothing_layout = QFormLayout(smoothing_group)
        
        self.smoothing_method = QComboBox()
        self.smoothing_method.addItems(['savgol', 'gaussian', 'moving_average'])
        smoothing_layout.addRow("Method:", self.smoothing_method)
        
        self.smoothing_window = QSpinBox()
        self.smoothing_window.setRange(3, 51)
        self.smoothing_window.setSingleStep(2)
        smoothing_layout.addRow("Window Length:", self.smoothing_window)
        
        layout.addWidget(smoothing_group)
        
        layout.addStretch()
        return widget
    
    def create_plotting_tab(self) -> QWidget:
        """Create plotting settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Plot settings group
        plot_group = QGroupBox("Plot Settings")
        plot_layout = QFormLayout(plot_group)
        
        self.plot_theme = QComboBox()
        self.plot_theme.addItems(['default', 'dark', 'seaborn', 'classic'])
        plot_layout.addRow("Theme:", self.plot_theme)
        
        self.plot_dpi = QSpinBox()
        self.plot_dpi.setRange(72, 600)
        plot_layout.addRow("DPI:", self.plot_dpi)
        
        self.figure_width = QDoubleSpinBox()
        self.figure_width.setRange(4.0, 20.0)
        self.figure_width.setSingleStep(0.5)
        plot_layout.addRow("Figure Width:", self.figure_width)
        
        self.figure_height = QDoubleSpinBox()
        self.figure_height.setRange(3.0, 15.0)
        self.figure_height.setSingleStep(0.5)
        plot_layout.addRow("Figure Height:", self.figure_height)
        
        layout.addWidget(plot_group)
        layout.addStretch()
        return widget
    
    def create_gui_tab(self) -> QWidget:
        """Create GUI settings tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Interface group
        interface_group = QGroupBox("Interface")
        interface_layout = QFormLayout(interface_group)
        
        self.gui_theme = QComboBox()
        self.gui_theme.addItems(['default', 'dark', 'light'])
        interface_layout.addRow("Theme:", self.gui_theme)
        
        self.remember_window_size = QCheckBox()
        interface_layout.addRow("Remember Window Size:", self.remember_window_size)
        
        self.auto_save_settings = QCheckBox()
        interface_layout.addRow("Auto-save Settings:", self.auto_save_settings)
        
        layout.addWidget(interface_group)
        layout.addStretch()
        return widget
    
    def load_current_settings(self):
        """Load current settings into controls"""
        # Processing settings
        processing = self.temp_settings.get('processing', {})
        self.baseline_method.setCurrentText(processing.get('default_baseline', 'polynomial'))
        self.baseline_degree.setValue(processing.get('baseline_degree', 2))
        self.smoothing_method.setCurrentText(processing.get('default_smoothing', 'savgol'))
        self.smoothing_window.setValue(processing.get('smoothing_window', 5))
        
        # Plotting settings
        plotting = self.temp_settings.get('plotting', {})
        self.plot_theme.setCurrentText(plotting.get('theme', 'default'))
        self.plot_dpi.setValue(plotting.get('dpi', 100))
        
        figure_size = plotting.get('figure_size', [10, 6])
        self.figure_width.setValue(figure_size[0])
        self.figure_height.setValue(figure_size[1])
        
        # GUI settings
        gui = self.temp_settings.get('gui', {})
        self.gui_theme.setCurrentText(gui.get('theme', 'default'))
        self.remember_window_size.setChecked(gui.get('remember_size', True))
        self.auto_save_settings.setChecked(gui.get('auto_save', True))
    
    def save_settings_from_controls(self):
        """Save settings from controls to temp settings"""
        # Processing settings
        if 'processing' not in self.temp_settings:
            self.temp_settings['processing'] = {}
        
        self.temp_settings['processing'].update({
            'default_baseline': self.baseline_method.currentText(),
            'baseline_degree': self.baseline_degree.value(),
            'default_smoothing': self.smoothing_method.currentText(),
            'smoothing_window': self.smoothing_window.value()
        })
        
        # Plotting settings
        if 'plotting' not in self.temp_settings:
            self.temp_settings['plotting'] = {}
        
        self.temp_settings['plotting'].update({
            'theme': self.plot_theme.currentText(),
            'dpi': self.plot_dpi.value(),
            'figure_size': [self.figure_width.value(), self.figure_height.value()]
        })
        
        # GUI settings
        if 'gui' not in self.temp_settings:
            self.temp_settings['gui'] = {}
        
        self.temp_settings['gui'].update({
            'theme': self.gui_theme.currentText(),
            'remember_size': self.remember_window_size.isChecked(),
            'auto_save': self.auto_save_settings.isChecked()
        })
    
    def apply_settings(self):
        """Apply current settings"""
        self.save_settings_from_controls()
        
        # Update actual settings
        for key, value in self.temp_settings.items():
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    self.settings.set(f'{key}.{subkey}', subvalue)
            else:
                self.settings.set(key, value)
        
        self.settings.save_settings()
        QMessageBox.information(self, "Settings", "Settings applied successfully")
    
    def accept_settings(self):
        """Accept and apply settings"""
        self.apply_settings()
        self.accept()
    
    def reset_to_defaults(self):
        """Reset settings to defaults"""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.Yes | QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.settings.reset_to_defaults()
            self.temp_settings = self.settings.get_all_settings().copy()
            self.load_current_settings()
            QMessageBox.information(self, "Settings", "Settings reset to defaults")