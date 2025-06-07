"""
Application settings and configuration management
Centralized configuration with persistence and validation
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import os
from dataclasses import dataclass, asdict
from enum import Enum

class PlotTheme(Enum):
    """Available plot themes"""
    LIGHT = "light"
    DARK = "dark"
    SEABORN = "seaborn"
    CLASSIC = "classic"

class ProcessingMethod(Enum):
    """Available processing methods"""
    BASELINE_CORRECTION = "baseline"
    SMOOTHING = "smoothing"
    NORMALIZATION = "normalization"
    DERIVATIVE = "derivative"

@dataclass
class PlotSettings:
    """Plot-specific settings"""
    figure_size: tuple = (12, 8)
    dpi: int = 100
    line_width: float = 1.5
    marker_size: float = 4.0
    alpha: float = 0.8
    grid: bool = True
    legend: bool = True
    theme: str = "dark"
    colormap: str = "viridis"
    auto_scale: bool = True

@dataclass
class ProcessingSettings:
    """Data processing settings"""
    baseline_method: str = "polynomial"
    baseline_degree: int = 2
    smoothing_window: int = 5
    smoothing_method: str = "savgol"
    normalization_method: str = "minmax"
    derivative_order: int = 1
    auto_process: bool = False

@dataclass
class PeakDetectionSettings:
    """Peak detection settings"""
    height_threshold: float = 0.1
    prominence_threshold: float = 0.05
    distance_threshold: float = 5.0
    width_range: tuple = (1.0, 50.0)
    clustering_tolerance: float = 2.0
    min_cluster_size: int = 2

@dataclass
class ExportSettings:
    """Export settings"""
    default_format: str = "png"
    image_dpi: int = 300
    include_metadata: bool = True
    compression: bool = True
    animation_fps: int = 10
    animation_duration: float = 5.0

class AppSettings:
    """
    Centralized application settings manager
    Handles configuration persistence and validation
    """
    
    # Default theme options
    AVAILABLE_THEMES = [
        'dark_teal.xml',
        'dark_purple.xml',
        'dark_pink.xml',
        'dark_blue.xml',
        'light_teal.xml',
        'light_purple.xml',
        'light_pink.xml',
        'light_blue.xml'
    ]
    
    DEFAULT_THEME = 'dark_teal.xml'
    
    def __init__(self, config_file: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        
        # Set up config file path
        if config_file is None:
            config_dir = Path.home() / '.drs_analyzer'
            config_dir.mkdir(exist_ok=True)
            self.config_file = config_dir / 'settings.json'
        else:
            self.config_file = Path(config_file)
        
        # Initialize default settings
        self.plot_settings = PlotSettings()
        self.processing_settings = ProcessingSettings()
        self.peak_detection_settings = PeakDetectionSettings()
        self.export_settings = ExportSettings()
        
        # General settings
        self._settings = {
            'theme': self.DEFAULT_THEME,
            'auto_save': True,
            'auto_update': False,
            'max_recent_files': 10,
            'recent_files': [],
            'window_geometry': None,
            'dock_states': {},
            'last_directory': str(Path.home()),
            'log_level': 'INFO',
            'performance_mode': False,
            'parallel_processing': True,
            'memory_limit_mb': 8192,
            'cache_size_mb': 1024
        }
        
        # Load existing settings
        self.load_settings()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get setting value"""
        return self._settings.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set setting value"""
        self._settings[key] = value
        self.save_settings()
    
    def get_plot_settings(self) -> PlotSettings:
        """Get plot settings"""
        return self.plot_settings
    
    def set_plot_settings(self, settings: PlotSettings) -> None:
        """Set plot settings"""
        self.plot_settings = settings
        self.save_settings()
    
    def get_processing_settings(self) -> ProcessingSettings:
        """Get processing settings"""
        return self.processing_settings
    
    def set_processing_settings(self, settings: ProcessingSettings) -> None:
        """Set processing settings"""
        self.processing_settings = settings
        self.save_settings()
    
    def get_peak_detection_settings(self) -> PeakDetectionSettings:
        """Get peak detection settings"""
        return self.peak_detection_settings
    
    def set_peak_detection_settings(self, settings: PeakDetectionSettings) -> None:
        """Set peak detection settings"""
        self.peak_detection_settings = settings
        self.save_settings()
    
    def get_export_settings(self) -> ExportSettings:
        """Get export settings"""
        return self.export_settings
    
    def set_export_settings(self, settings: ExportSettings) -> None:
        """Set export settings"""
        self.export_settings = settings
        self.save_settings()
    
    def add_recent_file(self, file_path: str) -> None:
        """Add file to recent files list"""
        recent_files = self._settings.get('recent_files', [])
        
        # Remove if already exists
        if file_path in recent_files:
            recent_files.remove(file_path)
        
        # Add to beginning
        recent_files.insert(0, file_path)
        
        # Limit size
        max_files = self._settings.get('max_recent_files', 10)
        recent_files = recent_files[:max_files]
        
        self._settings['recent_files'] = recent_files
        self.save_settings()
    
    def get_recent_files(self) -> List[str]:
        """Get recent files list"""
        recent_files = self._settings.get('recent_files', [])
        # Filter out non-existent files
        existing_files = [f for f in recent_files if Path(f).exists()]
        if len(existing_files) != len(recent_files):
            self._settings['recent_files'] = existing_files
            self.save_settings()
        return existing_files
    
    def clear_recent_files(self) -> None:
        """Clear recent files list"""
        self._settings['recent_files'] = []
        self.save_settings()
    
    def save_settings(self) -> None:
        """Save settings to file"""
        try:
            settings_dict = {
                'general': self._settings,
                'plot_settings': asdict(self.plot_settings),
                'processing_settings': asdict(self.processing_settings),
                'peak_detection_settings': asdict(self.peak_detection_settings),
                'export_settings': asdict(self.export_settings)
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(settings_dict, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.error(f"Failed to save settings: {e}")
    
    def load_settings(self) -> None:
        """Load settings from file"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    settings_dict = json.load(f)
                
                # Load general settings
                if 'general' in settings_dict:
                    self._settings.update(settings_dict['general'])
                
                # Load plot settings
                if 'plot_settings' in settings_dict:
                    plot_dict = settings_dict['plot_settings']
                    self.plot_settings = PlotSettings(**plot_dict)
                
                # Load processing settings
                if 'processing_settings' in settings_dict:
                    proc_dict = settings_dict['processing_settings']
                    self.processing_settings = ProcessingSettings(**proc_dict)
                
                # Load peak detection settings
                if 'peak_detection_settings' in settings_dict:
                    peak_dict = settings_dict['peak_detection_settings']
                    self.peak_detection_settings = PeakDetectionSettings(**peak_dict)
                
                # Load export settings
                if 'export_settings' in settings_dict:
                    export_dict = settings_dict['export_settings']
                    self.export_settings = ExportSettings(**export_dict)
                    
        except Exception as e:
            self.logger.warning(f"Failed to load settings, using defaults: {e}")
    
    def reset_to_defaults(self) -> None:
        """Reset all settings to defaults"""
        self.plot_settings = PlotSettings()
        self.processing_settings = ProcessingSettings()
        self.peak_detection_settings = PeakDetectionSettings()
        self.export_settings = ExportSettings()
        
        self._settings = {
            'theme': self.DEFAULT_THEME,
            'auto_save': True,
            'auto_update': False,
            'max_recent_files': 10,
            'recent_files': [],
            'window_geometry': None,
            'dock_states': {},
            'last_directory': str(Path.home()),
            'log_level': 'INFO',
            'performance_mode': False,
            'parallel_processing': True,
            'memory_limit_mb': 8192,
            'cache_size_mb': 1024
        }
        
        self.save_settings()
    
    def validate_settings(self) -> bool:
        """Validate current settings"""
        try:
            # Validate theme
            if self._settings.get('theme') not in self.AVAILABLE_THEMES:
                self._settings['theme'] = self.DEFAULT_THEME
            
            # Validate numeric ranges
            if not 100 <= self.plot_settings.dpi <= 600:
                self.plot_settings.dpi = 100
            
            if not 0.1 <= self.plot_settings.line_width <= 10.0:
                self.plot_settings.line_width = 1.5
            
            if not 1 <= self.processing_settings.smoothing_window <= 101:
                self.processing_settings.smoothing_window = 5
            
            if not 0.0 <= self.peak_detection_settings.height_threshold <= 1.0:
                self.peak_detection_settings.height_threshold = 0.1
            
            return True
            
        except Exception as e:
            self.logger.error(f"Settings validation failed: {e}")
            return False
    
    def export_settings(self, export_path: Union[str, Path]) -> bool:
        """Export settings to external file"""
        try:
            export_path = Path(export_path)
            
            settings_dict = {
                'general': self._settings,
                'plot_settings': asdict(self.plot_settings),
                'processing_settings': asdict(self.processing_settings),
                'peak_detection_settings': asdict(self.peak_detection_settings),
                'export_settings': asdict(self.export_settings),
                'export_info': {
                    'version': '2.0.0',
                    'export_date': str(Path().cwd()),
                    'description': 'DRS Analyzer settings export'
                }
            }
            
            with open(export_path, 'w') as f:
                json.dump(settings_dict, f, indent=2, default=str)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Settings export failed: {e}")
            return False
    
    def import_settings(self, import_path: Union[str, Path]) -> bool:
        """Import settings from external file"""
        try:
            import_path = Path(import_path)
            
            if not import_path.exists():
                raise FileNotFoundError(f"Settings file not found: {import_path}")
            
            # Backup current settings
            backup_path = self.config_file.with_suffix('.backup')
            if self.config_file.exists():
                shutil.copy2(self.config_file, backup_path)
            
            # Load new settings
            with open(import_path, 'r') as f:
                settings_dict = json.load(f)
            
            # Update settings (similar to load_settings)
            if 'general' in settings_dict:
                self._settings.update(settings_dict['general'])
            
            if 'plot_settings' in settings_dict:
                plot_dict = settings_dict['plot_settings']
                self.plot_settings = PlotSettings(**plot_dict)
            
            if 'processing_settings' in settings_dict:
                proc_dict = settings_dict['processing_settings']
                self.processing_settings = ProcessingSettings(**proc_dict)
            
            if 'peak_detection_settings' in settings_dict:
                peak_dict = settings_dict['peak_detection_settings']
                self.peak_detection_settings = PeakDetectionSettings(**peak_dict)
            
            if 'export_settings' in settings_dict:
                export_dict = settings_dict['export_settings']
                self.export_settings = ExportSettings(**export_dict)
            
            # Validate and save
            if self.validate_settings():
                self.save_settings()
                return True
            else:
                # Restore backup if validation fails
                if backup_path.exists():
                    shutil.copy2(backup_path, self.config_file)
                    self.load_settings()
                return False
                
        except Exception as e:
            self.logger.error(f"Settings import failed: {e}")
            return False
    
    def get_memory_limit(self) -> int:
        """Get memory limit in MB"""
        return self._settings.get('memory_limit_mb', 8192)
    
    def get_cache_size(self) -> int:
        """Get cache size in MB"""
        return self._settings.get('cache_size_mb', 1024)
    
    def is_performance_mode(self) -> bool:
        """Check if performance mode is enabled"""
        return self._settings.get('performance_mode', False)
    
    def is_parallel_processing_enabled(self) -> bool:
        """Check if parallel processing is enabled"""
        return self._settings.get('parallel_processing', True)
    
    def get_log_level(self) -> str:
        """Get logging level"""
        return self._settings.get('log_level', 'INFO')
    
    def __str__(self) -> str:
        """String representation of settings"""
        return f"AppSettings(theme={self.get('theme')}, auto_save={self.get('auto_save')})"
    
    def __repr__(self) -> str:
        """Detailed representation of settings"""
        return f"AppSettings(config_file={self.config_file}, settings_count={len(self._settings)})"