import logging
import logging.config
import os
import re

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
    "loggers": {
        # httpx logs every request URL at INFO. Analysis Service URLs carry
        # the reader's analysis token -- a bearer capability for their full
        # result -- in the path, so at INFO every summary and handoff wrote
        # tokens into the logs (review, area 1a).
        "httpx": {"level": "WARNING"},
        "httpcore": {"level": "WARNING"},
    },
    "root": {
        "handlers": ["console"],
        "level": DEFAULT_LOG_LEVEL,  # Set the default log level for all loggers
    },
}
logging.config.dictConfig(LOGGING_CONFIG)


_SESSION_ID = re.compile(r"(session_?id=)[^&\s\"]+", re.IGNORECASE)


class RedactSessionIds(logging.Filter):
    """Chainlit puts the session id in upload and download URLs, and the
    access log recorded them. For a guest that id is the whole credential: a
    second socket presenting it took the session over (review, area 1b)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                _SESSION_ID.sub(r"\1[redacted]", a) if isinstance(a, str) else a
                for a in record.args
            )
        return True


logging.getLogger("uvicorn.access").addFilter(RedactSessionIds())

# Importing this module configures logging as a side effect, and callers write
# `from util.logging import logging` so that the configuration is guaranteed to
# have run before they take a logger. That re-export is the point of the module,
# so declare it rather than leaving it implicit.
__all__ = ["DEFAULT_LOG_LEVEL", "LOGGING_CONFIG", "RedactSessionIds", "logging"]
