import logging
import logging.config
import os


DEFAULT_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": DEFAULT_LOG_LEVEL,  # Change to WARNING, ERROR, or CRITICAL
        },
    },
    "root": {
        "handlers": ["console"],
        "level": DEFAULT_LOG_LEVEL,  # Set the default log level for all loggers
    },
}
logging.config.dictConfig(LOGGING_CONFIG)
