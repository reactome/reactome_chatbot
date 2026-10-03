"""The cookie that says this browser passed the human check.

Review of the public surface (area 1b, 2026-10-03) found the first version was
`value|HMAC(secret, value)` with no issue time. `max_age` is only a hint to the
browser, so one solved challenge was a pass for ever, for any client, and --
since beta signs with production's Turnstile secret -- on production too.

Now the signed value carries its issue time and a random nonce:

- **It expires on the server.** A cookie older than `MAX_AGE_SECONDS` fails,
  whatever the browser was told.
- **It renews while in use.** One older than `RENEW_AFTER_SECONDS` is re-issued
  on the next HTTP response, so an open tab keeps working past the first hour
  instead of its uploads and buttons starting to fail.
- **The nonce names one solve.** Rate limits key on it: unlike Chainlit's
  session id, the client cannot mint a new one without solving another
  challenge.

Pure functions over the secret and the clock, so the rules are tested here; the
gate in `bin/chat-fastapi.py` is wiring.
"""

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from http.cookies import CookieError, SimpleCookie

COOKIE_NAME = "captcha_verified"
VERSION = "v2"
MAX_AGE_SECONDS = 12 * 60 * 60
RENEW_AFTER_SECONDS = 30 * 60
#: A clock a little behind ours must not make a fresh cookie look from the
#: future and fail.
FUTURE_SKEW_SECONDS = 60


@dataclass(frozen=True)
class Check:
    ok: bool
    #: Which solve this is, for rate limiting. Empty unless `ok`.
    nonce: str = ""
    #: Old enough that the response should carry a fresh cookie.
    renew: bool = False


def _sign(secret: str, value: str) -> str:
    return hmac.new(secret.encode(), value.encode(), hashlib.sha256).hexdigest()


def mint(secret: str, now: float) -> str:
    """A fresh cookie value: version, issue time, nonce, signature."""
    value = f"{VERSION}.{int(now)}.{secrets.token_urlsafe(16)}"
    return f"{value}|{_sign(secret, value)}"


def check(cookie_value: str | None, secret: str, now: float) -> Check:
    """Whether a cookie value is a current pass. Never raises."""
    if not cookie_value or not secret:
        return Check(ok=False)
    value, _, signature = cookie_value.partition("|")
    if not hmac.compare_digest(signature, _sign(secret, value)):
        return Check(ok=False)
    parts = value.split(".")
    if len(parts) != 3 or parts[0] != VERSION or not parts[1].isdigit():
        # Signed but in the old, timeless format: a pass that never expired.
        return Check(ok=False)
    age = now - int(parts[1])
    if age > MAX_AGE_SECONDS or age < -FUTURE_SKEW_SECONDS:
        return Check(ok=False)
    return Check(ok=True, nonce=parts[2], renew=age > RENEW_AFTER_SECONDS)


def from_cookie_header(header: str | None) -> str | None:
    """This cookie's value from a raw `Cookie:` header, or None."""
    if not header:
        return None
    jar: SimpleCookie = SimpleCookie()
    try:
        jar.load(header)
    except CookieError:
        return None
    morsel = jar.get(COOKIE_NAME)
    return morsel.value if morsel else None
