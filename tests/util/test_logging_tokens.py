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
