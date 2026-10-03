"""Request URLs that carry analysis tokens must not reach the logs.

httpx logs every request URL at INFO, and Analysis Service URLs carry the
reader's analysis token -- a bearer capability for their full result. At the
default level every summary and handoff wrote tokens into the logs (review,
area 1a).
"""

import logging

import util.logging


def test_httpx_request_lines_are_not_logged_at_info() -> None:
    for name in ("httpx", "httpcore"):
        assert not logging.getLogger(name).isEnabledFor(logging.INFO)


def test_our_own_info_logs_still_are() -> None:
    # The fix must lower only the request logger, not logging generally.
    root = logging.getLogger()
    assert root.isEnabledFor(logging.getLevelName(util.logging.DEFAULT_LOG_LEVEL))


def test_session_ids_are_redacted_from_the_access_log() -> None:
    # The access log line uvicorn writes for a Chainlit file download.
    record = logging.LogRecord(
        "uvicorn.access",
        logging.INFO,
        __file__,
        0,
        '%s - "%s %s HTTP/%s" %d',
        (
            "1.2.3.4:5",
            "GET",
            "/chat/guest/project/file/abc?session_id=SECRET-SID",
            "1.1",
            200,
        ),
        None,
    )
    assert logging.getLogger("uvicorn.access").filter(record)
    line = record.getMessage()
    assert "SECRET-SID" not in line
    assert "session_id=[redacted]" in line
