"""A handoff has to survive the Turnstile gate.

A visitor without the captcha cookie goes /chat/guest/#handoff=<id> -> 307 ->
captcha page -> form POST -> 302 -> /chat/guest/. The 307 keeps the fragment;
the form POST cannot carry it. Without a workaround, most people arriving
from the website -- anyone who has not passed the captcha in the last hour --
would land in a chat without the context they clicked for, and nothing would
say so, because there would be no handoff left to claim.

Found by driving the real gate with Cloudflare's always-pass test keys, over
HTTPS (the gate rejects an `http:` Referer, so a plain-HTTP test could not
have exercised it). The fix is split across two files, which is why this
pins that they agree: the captcha page stashes the fragment in
sessionStorage, and custom.js restores it. Rename the key in one and the
handoff is silently dropped again.

Importing bin/chat-fastapi.py builds the graph, so, as in
test_chat_requires_human.py, the page is checked in its source.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KEY = "reactome-handoff-fragment"


def captcha_page_source() -> str:
    source = (REPO / "bin" / "chat-fastapi.py").read_text()
    start = source.index("async def captcha_page()")
    return source[start : source.index("\n@app.", start)]


def test_the_captcha_page_stashes_the_handoff() -> None:
    page = captcha_page_source()
    assert f"sessionStorage.setItem('{KEY}'" in page
    # Stashed from the fragment, and only when there is a handoff in it.
    assert "window.location.hash" in page
    assert "handoff=" in page


def test_custom_js_restores_it_under_the_same_key() -> None:
    js = (REPO / "public" / "custom.js").read_text()
    assert f"const STASH = '{KEY}'" in js
    assert "sessionStorage.getItem(STASH)" in js
    # Cleared once read, so a later plain visit in the same tab cannot
    # claim it again.
    assert "sessionStorage.removeItem(STASH)" in js
    # Put back into the URL, so a reload keeps working (FR-007).
    assert "history.replaceState" in js


def test_both_halves_use_one_key() -> None:
    page_keys = set(
        re.findall(r"sessionStorage\.setItem\('([^']+)'", captcha_page_source())
    )
    js = (REPO / "public" / "custom.js").read_text()
    js_keys = set(re.findall(r"const STASH = '([^']+)'", js))
    assert page_keys == js_keys == {KEY}
