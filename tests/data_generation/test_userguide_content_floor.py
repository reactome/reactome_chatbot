"""A 200 is not evidence that a page has content.

beta and the internal Angular app both answer 200 for /userguide and return a
client-rendered shell: the guide is assembled in the browser, so a plain fetch
gets inlined CSS and font declarations. Measured 2026-09-17 -- about 1,240 words
of visible text against production's 2,372 of documentation.

A bundle built from that would pass every check we have (ten files, correct
embedding dimensions, non-zero document count) and answer nothing. Two sessions
checked status codes and paths and both concluded it was fine, which is exactly
why the check is on content rather than on the response.
"""

from pathlib import Path

import pytest
import requests

from data_generation.userguide.fetch import MIN_PAGE_WORDS, fetch_userguide_pages


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text
        self.status_code = 200
        self.encoding = "utf-8"

    def raise_for_status(self) -> None:
        pass


class _Session:
    def __init__(self, text: str) -> None:
        self.text = text
        self.headers: dict[str, str] = {}

    def get(self, _url: str, timeout: int = 0) -> _Response:
        return _Response(self.text)


SHELL = (
    "<html><head><style>"
    + "@font-face{font-family:'Roboto';src:url(https://fonts.gstatic.com/x.woff2);} "
    * 400
    + "</style></head><body><app-root></app-root></body></html>"
)
REAL = (
    "<html><body><h1>The Pathway Browser</h1><p>"
    + ("word " * 600)
    + "</p></body></html>"
)


def _fetch(
    tmp_path: Path, html: str, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Path]:
    monkeypatch.setattr(requests, "Session", lambda: _Session(html))
    return fetch_userguide_pages(
        ("https://example.invalid/userguide",), cache_dir=tmp_path, force=True
    )


def test_a_single_page_app_shell_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="words of visible text"):
        _fetch(tmp_path, SHELL, monkeypatch)


def test_the_shell_would_otherwise_have_looked_fine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """It is large, and it is a 200. Only the *visible* text gives it away."""
    assert len(SHELL) > 20_000
    assert len(SHELL.split()) > MIN_PAGE_WORDS


def test_a_real_page_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = _fetch(tmp_path, REAL, monkeypatch)
    assert len(paths) == 1
    assert next(iter(paths.values())).exists()


def test_nothing_is_cached_when_the_content_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Caching a shell would make the next run succeed against bad content."""
    with pytest.raises(RuntimeError):
        _fetch(tmp_path, SHELL, monkeypatch)
    assert list(tmp_path.glob("*.html")) == []
