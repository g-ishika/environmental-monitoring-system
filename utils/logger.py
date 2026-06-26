"""Logging utilities for environmental monitoring"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
import json


def setup_logger(name: str = "environmental_monitor", 
                 log_level: str = "INFO",
                 log_file: str = None) -> logging.Logger:
    """Setup logger with file and console handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create formatters
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{datetime.now().strftime('%Y%m%d')}_environmental_monitor.log"
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10_000_000,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            'timestamp': self.formatTime(record),
            'name': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        return json.dumps(log_entry)


class Logger:
    """Enhanced logger with structured logging support"""
    
    def __init__(self, name: str = "environmental_monitor", 
                 log_file: str = None,
                 json_logs: bool = False):
        self.logger = setup_logger(name, log_file=log_file)
        self.json_logs = json_logs
        
        if json_logs:
            # Replace formatter with JSON formatter
            for handler in self.logger.handlers:
                handler.setFormatter(JSONFormatter())
    
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra={'extra': kwargs})
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra={'extra': kwargs})
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra={'extra': kwargs})
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra={'extra': kwargs})
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, extra={'extra': kwargs})
    
    def exception(self, message: str, **kwargs):
        self.logger.exception(message, extra={'extra': kwargs})


# Default logger instance
default_logger = Logger()

# Convenience functions
def get_logger(name: str = "environmental_monitor") -> logging.Logger:
    """Get logger instance"""
    return setup_logger(name)