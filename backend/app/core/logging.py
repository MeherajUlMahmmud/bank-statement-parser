import logging
import logging.handlers
import sys
from pathlib import Path

from app.core.config import settings

# Ensure the logs directory exists
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Define log formats
DETAILED_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
SIMPLE_FORMAT = "%(levelname)s - %(message)s"

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "detailed": {
            "format": DETAILED_FORMAT,
        },
        "simple": {
            "format": SIMPLE_FORMAT,
        },
    },
    "handlers": {
        # Console handler for development
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "simple",
            "stream": sys.stdout,
        },
        # Rotating file handler for the main application logs
        "rotating_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "app.log"),
            "maxBytes": 5 * 1024 * 1024,  # 5 MB
            "backupCount": 5,             # Keep 5 backup files
            "encoding": "utf8",
        },
        # A separate rotating file for errors
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "detailed",
            "filename": str(LOG_DIR / "error.log"),
            "maxBytes": 5 * 1024 * 1024,  # 5 MB
            "backupCount": 5,             # Keep 5 backup files
            "encoding": "utf8",
        },
    },
    "loggers": {
        # The root logger
        "": {
            "level": "WARNING",
            "handlers": ["console", "rotating_file", "error_file"],
        },
        # The logger for our application
        "app": {
            "level": "DEBUG" if settings.DEBUG else "INFO",
            "handlers": ["console", "rotating_file", "error_file"],
            "propagate": False,  # Don't pass messages to the root logger
        },
        # Control uvicorn's logging output
        "uvicorn": {
            "level": "INFO",
            "handlers": ["console", "rotating_file"],
            "propagate": False,
        },
        "uvicorn.error": {
            "level": "INFO",
            "handlers": ["console", "rotating_file"],
            "propagate": False,
        },
        "uvicorn.access": {
            "level": "INFO",  # Set to WARNING to reduce noise
            "handlers": ["console", "rotating_file"],
            "propagate": False,
        },
        # Control httpx's logging output (for external API calls)
        "httpx": {
            "level": "WARNING",
            "handlers": ["console", "rotating_file"],
            "propagate": False,
        },
    }
}
