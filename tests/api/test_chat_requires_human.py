"""The chat must not be served without a human check by accident.

Adam asked for a human check on the regular chat, not only the search-page
endpoint. The obstacle was not the check -- the middleware existed -- but that a
deployment with no Turnstile key skipped it and said nothing, so "there is
captcha middleware" and "the chat is gated" were different statements with
nothing to tell them apart.

This pins the decision being explicit in both directions: no key means refuse to
start, and a deployment that genuinely wants no check must say so.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent.parent
SNIPPET = """
import os, sys
sys.path.insert(0, "src")
required = os.getenv("CHAT_REQUIRES_HUMAN", "1").strip() not in {"0", "false", "no"}
secret = os.getenv("CLOUDFLARE_SECRET_KEY") or ""
if required and not secret:
    raise SystemExit("REFUSED")
print("STARTED")
"""


def _run(env: dict[str, str]) -> str:
    """The startup decision, in isolation.

    Importing bin/chat-fastapi.py itself would build a graph and need an OpenAI
    key, so the guard's logic is exercised rather than the module. The test for
    the two staying in step is below.
    """
    result = subprocess.run(  # noqa: S603 - fixed argv, no shell, no user input
        [sys.executable, "-c", SNIPPET],
        capture_output=True,
        text=True,
        cwd=REPO,
        env={"PATH": "/usr/bin:/bin", **env},
    )
    return (result.stdout + result.stderr).strip()


def test_no_key_and_no_opt_out_refuses_to_start() -> None:
    assert "REFUSED" in _run({})


def test_a_key_starts_the_chat() -> None:
    assert "STARTED" in _run({"CLOUDFLARE_SECRET_KEY": "0x-secret"})


@pytest.mark.parametrize("value", ["0", "false", "no"])
def test_an_explicit_opt_out_starts_without_a_key(value: str) -> None:
    """Beta ran ungated deliberately. That stays possible -- it just has to be
    written down rather than inferred from an absent value."""
    assert "STARTED" in _run({"CHAT_REQUIRES_HUMAN": value})


@pytest.mark.parametrize("value", ["1", "true", "yes", "", "anything-else"])
def test_anything_but_an_explicit_off_still_requires_a_key(value: str) -> None:
    """A typo in the opt-out must not silently disable the check."""
    assert "REFUSED" in _run({"CHAT_REQUIRES_HUMAN": value})


def test_the_guard_in_the_app_matches_the_one_tested_here() -> None:
    """The snippet above is a copy, so pin that it has not drifted."""
    source = (REPO / "bin" / "chat-fastapi.py").read_text()
    assert 'os.getenv("CHAT_REQUIRES_HUMAN", "1").strip() not in {' in source
    for off in ('"0",', '"false",', '"no",'):
        assert off in source
    assert "if CHAT_REQUIRES_HUMAN and not CLOUDFLARE_SECRET_KEY:" in source
