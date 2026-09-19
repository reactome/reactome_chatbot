"""A per-caller request limit for the answer endpoint (FR-008).

A backstop, not the budget. The website enforces the real one before a call
reaches here; this exists so a leaked or shared token cannot run up an unbounded
bill against a service whose every answer costs six model calls.

In process and in memory, because there is one process serving this. If the
service is ever scaled out, a shared store has to replace this, and the limit
becomes per instance until it is.
"""

import hashlib
import os
import time
from collections import deque


def _positive_int(name: str, default: int) -> int:
    """Configuration that is absent, empty or nonsense falls back to the default."""
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def identity_of(claims: dict[str, object], token: str) -> str:
    """Who to count against.

    `sub` then `jti` when present, else a hash of the token itself.

    D1 is settled now: `sub` does arrive, and for a reader who has passed the
    Turnstile challenge it is the identity cookie's subject -- browser-scoped
    for the life of that cookie, not per-visit. So this counts a person across
    visits rather than within one, which is the stronger backstop and is what
    has been running since the gate shipped. Worth knowing before reasoning
    about what a burst here means.

    Hashed, never raw: this lands in a dict that lives as long as the process, and
    a bearer token is a credential.
    """
    for claim in ("sub", "jti"):
        value = claims.get(claim)
        if isinstance(value, str) and value:
            return f"{claim}:{value}"
    return "token:" + hashlib.sha256(token.encode()).hexdigest()[:32]


class SlidingWindowLimiter:
    """Allow `limit` requests per `window` seconds, per key.

    No lock. Every mutation happens between awaits on one event loop, so a
    request cannot be interleaved mid-update. Adding an await inside `allow`
    would break that, which is why it does no I/O.
    """

    def __init__(self, limit: int, window: float) -> None:
        self.limit = limit
        self.window = window
        self._hits: dict[str, deque[float]] = {}
        self._last_sweep = 0.0

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        self._sweep(now)
        hits = self._hits.setdefault(key, deque())
        cutoff = now - self.window
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True

    def _sweep(self, now: float) -> None:
        """Drop keys with nothing left in the window.

        Without this the dict grows with every distinct token forever, which on a
        search page is every visitor. Swept once per window rather than per
        request, so the cost is amortised.
        """
        if now - self._last_sweep < self.window:
            return
        self._last_sweep = now
        cutoff = now - self.window
        self._hits = {
            key: hits for key, hits in self._hits.items() if hits and hits[-1] > cutoff
        }


def limiter_from_env() -> SlidingWindowLimiter:
    """30 requests per 10 minutes by default.

    Generous for a person -- an answer takes 20-40 seconds, so thirty is far more
    than anyone reads -- and it still caps a leaked token at 180 an hour.
    """
    return SlidingWindowLimiter(
        limit=_positive_int("ANSWER_RATE_LIMIT", 30),
        window=float(_positive_int("ANSWER_RATE_WINDOW_SECONDS", 600)),
    )
