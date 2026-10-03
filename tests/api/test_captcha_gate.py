"""The human check, driven through the real app over HTTP and websocket.

Review of the public surface (area 1b, 2026-10-03): the gate was HTTP
middleware, which never sees a websocket, so a client opening the chat's
socket directly got the whole chat with no captcha. Nothing caught it because
no test drove the real gate -- the existing ones grep source or test a copied
snippet. These run `gate_probe.py`, which imports `bin/chat-fastapi.py` in its
own process (Chainlit's configuration is global) and reports what happened.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROBE = Path(__file__).with_name("gate_probe.py")


@pytest.fixture(scope="module")
def gate() -> dict[str, Any]:
    done = subprocess.run(  # noqa: S603
        [sys.executable, str(PROBE)],
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    return dict(json.loads(done.stdout.strip().splitlines()[-1]))


GATED = [307, "/chat/guest/verify_captcha_page", False]


@pytest.mark.parametrize("case", ["ws_no_cookie", "ws_old_format", "ws_expired"])
def test_the_websocket_is_gated(gate: dict[str, Any], case: str) -> None:
    assert gate[case] == "refused:1008"


def test_a_passed_check_opens_the_websocket(gate: dict[str, Any]) -> None:
    # The control: without it, a gate that refused everything would pass.
    assert gate["ws_fresh"] == "open:0"


@pytest.mark.parametrize(
    "case", ["http_no_cookie", "http_old_format", "http_expired", "http_api_lookalike"]
)
def test_http_without_a_current_pass_is_sent_to_the_check(
    gate: dict[str, Any], case: str
) -> None:
    assert gate[case] == GATED


def test_a_current_pass_is_let_through_and_renewed_when_ageing(
    gate: dict[str, Any],
) -> None:
    assert gate["http_fresh"] == [200, "", False]
    assert gate["http_renewed"] == [200, "", True]


def test_a_reader_from_a_plain_http_page_is_let_in(gate: dict[str, Any]) -> None:
    # The Referer check returned 400 to anyone following a link from an http
    # page, and stopped no one.
    assert gate["http_from_plain_http_page"] == [200, "", False]


def test_the_check_form_takes_no_files(gate: dict[str, Any]) -> None:
    assert gate["verify_with_a_file"] == 400
