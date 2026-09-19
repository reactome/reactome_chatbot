"""Verify the token the website mints for each call. Signature, expiry, audience.

Agreed with the website session on 2026-09-18, which corrected the premise this
started from. There is **no human gate on their search path** and there is not
going to be one -- nobody solves a captcha to run a search. Their only hCaptcha
belongs to the contact form, and its response is spent on submit. So a token here
cannot honestly assert that a human is present, and this module does not claim it
does.

What it does assert is **caller identity**: this request came from the Reactome
website's server, for one visit. They mint server-side per request, EdDSA, with
`iss`, `aud`, `iat`, `exp` at +120s, and `sub` -- 128 random bits, not derived
from anything about the reader.

**`sub` is browser-scoped for a gated reader, not per-visit**, and this said
otherwise until 2026-09-19. Their `callerSubject()` prefers the Turnstile
identity cookie's subject whenever the reader has passed a challenge, and falls
back to a per-visit cookie only when they have not. So the backstop limit below
has been keyed on an identifier lasting as long as that cookie -- the stronger
throttle, and the behaviour in production since the gate shipped. It is kept
deliberately; what was wrong was this description of it, which named the
fallback as though it were the only case. Abuse control is
theirs: the panel is opt-in behind a click, so a crawled search never reaches a
model, and their proxy rate limits by address.

**Asymmetric on purpose.** They hold the signing key; this holds only the public
half. A shared secret would let either side mint, and this is the side reachable
from a search page if the proxy is ever bypassed.

**The audience is enforced, not optional.** They asked for it, and it is what
stops a token minted for this service being replayed at a different consumer
later -- or one minted for something else being presented here.

**Stateless on purpose.** The query budget is theirs, enforced before the call
reaches here. What this service keeps is a backstop limit keyed on `sub`.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import jwt

ALGORITHMS = ["EdDSA", "RS256"]
"""What the website may sign with. Both are asymmetric; no HS* symmetric option is
offered, because accepting one would let a leaked verifying key mint tokens."""

KEY_PATH_ENV = "CALLER_TOKEN_PUBLIC_KEY_PATH"


@dataclass(frozen=True)
class TokenRejectedError(Exception):
    """Why a token was refused. The reason is logged, never returned to the caller."""

    reason: str


def load_verifying_key(path: str | None = None) -> str:
    """The public key, or refuse to start.

    Principle IV. An endpoint that accepts everything because its key is missing is
    the worst outcome available, and it would test clean -- every request would
    succeed.
    """
    configured = path or os.getenv(KEY_PATH_ENV)
    if not configured:
        raise RuntimeError(
            f"{KEY_PATH_ENV} is not set. The answer endpoint verifies a signed "
            "caller token and cannot run without a verifying key; starting "
            "without one would accept every request."
        )
    key_file = Path(configured)
    try:
        key = key_file.read_text().strip()
    except OSError as exc:
        raise RuntimeError(
            f"Cannot read the verifying key at {key_file}: {exc}"
        ) from exc
    if not key:
        raise RuntimeError(f"The verifying key at {key_file} is empty.")
    return key


AUDIENCE_ENV = "CALLER_TOKEN_AUDIENCE"
DEFAULT_AUDIENCE = "reactome-chatbot"


def expected_audience() -> str:
    """Who tokens must be minted for. Configurable, but never empty.

    An empty audience would not mean "accept anything" -- PyJWT rejects a token
    that carries `aud` when none is expected -- so a blank setting would refuse
    every real token instead of loosening the check. Falling back to the agreed
    value keeps a misconfiguration from looking like a signing problem.
    """
    return os.getenv(AUDIENCE_ENV, "").strip() or DEFAULT_AUDIENCE


def verify(token: str, verifying_key: str, *, audience: str | None = None) -> dict:
    """Return the token's claims, or raise TokenRejectedError.

    Every failure path refuses. There is deliberately no branch that returns claims
    for an unverified token, however malformed.

    `audience` defaults to the configured one and is required: `aud` is in the
    required claims, so a token without it is refused by name rather than slipping
    through. Passing an explicit audience is for tests.
    """
    if not token:
        raise TokenRejectedError("no token presented")
    expected = audience or expected_audience()
    try:
        return dict(
            jwt.decode(
                token,
                verifying_key,
                algorithms=ALGORITHMS,
                audience=expected,
                # "aud" here is belt-and-braces, and deliberately kept despite
                # being redundant today: PyJWT already raises
                # MissingRequiredClaimError for an absent `aud` when an audience
                # is expected, so removing it fails no test. It is here so that
                # behaviour changing in a future PyJWT cannot quietly turn "no
                # audience" into "nothing to check".
                options={"require": ["exp", "aud"]},
            )
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenRejectedError("token expired") from exc
    except jwt.InvalidAudienceError as exc:
        raise TokenRejectedError("token minted for a different audience") from exc
    except jwt.InvalidSignatureError as exc:
        raise TokenRejectedError("signature does not verify") from exc
    except jwt.MissingRequiredClaimError as exc:
        # Named, not assumed: this used to report "no expiry" for whatever was
        # missing, which would now misreport a token with no audience. A token
        # with no expiry is a permanent credential, which is what the short
        # lifetime exists to avoid; one with no audience cannot be checked
        # against the consumer it was minted for.
        raise TokenRejectedError(
            f"token is missing a required claim: {exc.claim}"
        ) from exc
    except jwt.InvalidTokenError as exc:
        raise TokenRejectedError(f"invalid token: {type(exc).__name__}") from exc
