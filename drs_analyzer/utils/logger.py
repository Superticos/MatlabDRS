"""
Logging utilities for DRS Analyzer
"""
import logging
import sys
from pathlib import Path
from typing import Union, Optional
from logging.handlers import RotatingFileHandler
import os

def setup_logging(level: Union[str, int] = logging.INFO,
                 log_file: Optional[str] = None,
                 log_dir: Optional[str] = None,
                 max_file_size: int = 10 * 1024 * 1024,
                 backup_count: int = 5,
                 console_output: bool = True) -> logging.Logger:
    """
    Setup logging configuration for DRS Analyzer
    
    Args:
        level: Logging level
        log_file: Specific log file path
        log_dir: Log directory (if log_file not specified)
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        console_output: Whether to output to console
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger('drs_analyzer')
    logger.setLevel(level)
    
    # Clear existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler (if enabled)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler (if requested)
    if log_file or log_dir:
        try:
            if log_file:
                log_path = Path(log_file)
            else:
                log_dir_path = Path(log_dir) if log_dir else Path.home() / '.drs_analyzer' / 'logs'
                log_dir_path.mkdir(parents=True, exist_ok=True)
                log_path = log_dir_path / 'drs_analyzer.log'
            
            # Ensure parent directory exists
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if we can write to the log file
            if log_path.exists() and not os.access(log_path, os.W_OK):
                print(f"Warning: Cannot write to log file {log_path}, skipping file logging")
            else:
                file_handler = RotatingFileHandler(
                    log_path, 
                    maxBytes=max_file_size,
                    backupCount=backup_count,
                    encoding='utf-8'
                )
                file_handler.setLevel(level)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)
                
        except Exception as e:
            print(f"Warning: Failed to setup file logging: {e}")
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger

def get_logger(name: str = None) -> logging.Logger:
    """Get logger instance"""
    return logging.getLogger(name or 'drs_analyzer')

def set_log_level(level: Union[str, int]):
    """Set logging level for all DRS loggers"""
    logger = logging.getLogger('drs_analyzer')
    logger.setLevel(level)
    
    # Update all handlers
    for handler in logger.handlers:
        handler.setLevel(level)

def add_file_handler(log_file: Union[str, Path], 
                    level: Union[str, int] = logging.INFO,
                    max_file_size: int = 10 * 1024 * 1024,
                    backup_count: int = 5) -> bool:
    """Add file handler to existing logger"""
    try:
        logger = logging.getLogger('drs_analyzer')
        log_path = Path(log_file)
        
        # Ensure parent directory exists
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        
        logger.addHandler(file_handler)
        return True
        
    except Exception as e:
        print(f"Failed to add file handler: {e}")
        return False

def remove_console_output():
    """Remove console output from logger"""
    logger = logging.getLogger('drs_analyzer')
    handlers_to_remove = []
    
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            handlers_to_remove.append(handler)
    
    for handler in handlers_to_remove:
        logger.removeHandler(handler)

class LoggerMixin:
    """Mixin class to add logging capabilities to any class"""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class"""
        if not hasattr(self, '_logger'):
            self._logger = get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        return self._logger