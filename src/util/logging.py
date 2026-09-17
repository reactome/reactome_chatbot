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

# Importing this module configures logging as a side effect, and callers write
# `from util.logging import logging` so that the configuration is guaranteed to
# have run before they take a logger. That re-export is the point of the module,
# so declare it rather than leaving it implicit.
__all__ = ["DEFAULT_LOG_LEVEL", "LOGGING_CONFIG", "logging"]
