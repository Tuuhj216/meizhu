#!/usr/bin/env python3
"""
Logging utilities for GPIO controller system
"""
import logging
import os
from datetime import datetime

def setup_logger(name, log_file=None, level=logging.INFO):
    """
    Setup logger with console and optionally file output

    Args:
        name: Logger name
        log_file: Optional log file path
        level: Logging level

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

def get_process_logger(process_name):
    """
    Get logger for a specific process

    Args:
        process_name: Name of the process

    Returns:
        Logger instance
    """
    log_file = f"logs/{process_name}_{datetime.now().strftime('%Y%m%d')}.log"
    return setup_logger(process_name, log_file)