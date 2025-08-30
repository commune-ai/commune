"""
Standardized logging utilities for the Polaris validator system.
"""
import logging
import os
import sys
import traceback
from typing import Optional, Callable, Any

# Configure root logger
def configure_logging(
    log_level: int = logging.INFO,
    log_file: Optional[str] = "validator.log",
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> logging.Logger:
    """
    Configure logging for the validator.
    
    Args:
        log_level: The logging level to use
        log_file: Path to the log file, or None to disable file logging
        log_format: Format string for log messages
        
    Returns:
        The configured root logger
    """
    # Create handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure root logger
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger()

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        name: The name of the module
        
    Returns:
        A logger instance for the module
    """
    return logging.getLogger(name)

def log_exception(
    logger: logging.Logger,
    msg: str,
    exc: Exception,
    include_traceback: bool = True,
    level: int = logging.ERROR,
) -> None:
    """
    Log an exception with a consistent format.
    
    Args:
        logger: The logger to use
        msg: A descriptive message about what failed
        exc: The exception that was caught
        include_traceback: Whether to include the traceback in the log
        level: The logging level to use
    """
    if include_traceback:
        logger.log(level, f"{msg}: {exc}\n{traceback.format_exc()}")
    else:
        logger.log(level, f"{msg}: {exc}")

def exception_handler(
    logger: logging.Logger,
    msg: str,
    fallback_value: Any = None,
    include_traceback: bool = True,
    level: int = logging.ERROR,
) -> Callable:
    """
    Create a decorator for handling exceptions in a consistent way.
    
    Args:
        logger: The logger to use
        msg: A descriptive message about what failed
        fallback_value: Value to return if an exception occurs
        include_traceback: Whether to include the traceback in the log
        level: The logging level to use
        
    Returns:
        A decorator for handling exceptions
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_exception(logger, f"{msg} in {func.__name__}", e, include_traceback, level)
                return fallback_value
        return wrapper
    return decorator 