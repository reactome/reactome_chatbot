"""Drive the real captcha gate, in its own process, and print what happened.

Run by `test_captcha_gate.py`. A separate process because importing
`bin/chat-fastapi.py` mounts Chainlit, whose configuration is global and would
leak into the rest of the suite. The lifespan (which builds the graph) is never
entered, so this takes seconds, not minutes.
"""

import hashlib
import hmac
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SECRET = "test-secret-not-a-real-key"  # noqa: S105

os.environ.update(
    {
        "CLOUDFLARE_SECRET_KEY": SECRET,
        "CLOUDFLARE_SITE_KEY": "test-site-key",
        "CHAINLIT_URI": "/chat/guest",
        "CHAINLIT_URL": "https://testserver",
        "OPENAI_API_KEY": "sk-test",
        "LOG_LEVEL": "error",
    }
)
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

spec = importlib.util.spec_from_file_location(
    "chat_fastapi", REPO / "bin/chat-fastapi.py"
)
assert spec is not None
assert spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

from fastapi.testclient import TestClient  # noqa: E402
from starlette.websockets import WebSocketDisconnect  # noqa: E402

from util import captcha_cookie  # noqa: E402

client = TestClient(module.app, base_url="https://testserver")
WS = "/chat/guest/ws/socket.io/?EIO=4&transport=websocket"
results: dict[str, object] = {}


def fresh() -> str:
    return captcha_cookie.mint(SECRET, time.time())


def aged(seconds: float) -> str:
    return captcha_cookie.mint(SECRET, time.time() - seconds)


def old_format() -> str:
    value = "a-turnstile-token-from-2024"
    return f"{value}|{hmac.new(SECRET.encode(), value.encode(), hashlib.sha256).hexdigest()}"


def http(path: str, cookie: str | None = None, **headers: str) -> tuple[int, str, bool]:
    client.cookies.clear()
    if cookie:
        client.cookies.set(captcha_cookie.COOKIE_NAME, cookie)
    response = client.get(path, headers=headers, follow_redirects=False)
    renewed = captcha_cookie.COOKIE_NAME in response.headers.get("set-cookie", "")
    return response.status_code, response.headers.get("location", ""), renewed


def ws(cookie: str | None) -> str:
    client.cookies.clear()
    # engine.io refuses a TestClient websocket without these.
    headers = {"upgrade": "websocket", "connection": "Upgrade"}
    if cookie:
        headers["cookie"] = f"{captcha_cookie.COOKIE_NAME}={cookie}"
    try:
        with client.websocket_connect(WS, headers=headers) as socket:
            return "open:" + socket.receive_text()[:1]
    except WebSocketDisconnect as closed:
        return f"refused:{closed.code}"


results["http_no_cookie"] = http("/chat/guest/")
results["http_fresh"] = http("/chat/guest/", fresh())
results["http_old_format"] = http("/chat/guest/", old_format())
results["http_expired"] = http(
    "/chat/guest/", aged(captcha_cookie.MAX_AGE_SECONDS + 60)
)
results["http_renewed"] = http(
    "/chat/guest/", aged(captcha_cookie.RENEW_AFTER_SECONDS + 60)
)
results["http_from_plain_http_page"] = http(
    "/chat/guest/", fresh(), referer="http://example.org/blog"
)
results["http_api_lookalike"] = http("/chat/guest/api-anything")
results["ws_no_cookie"] = ws(None)
results["ws_old_format"] = ws(old_format())
results["ws_expired"] = ws(aged(captcha_cookie.MAX_AGE_SECONDS + 60))
results["ws_fresh"] = ws(fresh())

client.cookies.clear()
upload = client.post(
    "/chat/guest/verify_captcha",
    files={"big": ("big.bin", b"x" * 200_000)},
    follow_redirects=False,
)
results["verify_with_a_file"] = upload.status_code

print(json.dumps(results))
