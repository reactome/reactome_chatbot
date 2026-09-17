"""Verify that the caller proved someone is human. Signature and expiry, nothing else.

Agreed with the website session on 2026-09-17. They run their own captcha, mint a
short-lived token, and proxy every request presenting it. This service verifies the
signature and refuses otherwise.

**Asymmetric on purpose.** They hold the signing key; this holds only a public
verifying key. A shared secret would let either side mint, and this is the side
reachable from a search page if the proxy is ever bypassed -- so compromising it must
not produce valid tokens.

**Stateless on purpose.** The query budget is theirs, enforced before the call
reaches here, because they proxy every request. Counting consumption here would mean
state and a second counter that can disagree with theirs.

Worth knowing, because it looks like an inconsistency: this repo verifies Cloudflare
Turnstile for the chat UI and the website's search page uses hCaptcha. Under a shared
cookie those two would have to be reconciled. Under minting this service never sees a
captcha at all, so there is nothing to reconcile -- do not "unify" them.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import jwt

ALGORITHMS = ["EdDSA", "RS256"]
"""What the website may sign with. Both are asymmetric; no HS* symmetric option is
offered, because accepting one would let a leaked verifying key mint tokens."""

KEY_PATH_ENV = "HUMAN_TOKEN_PUBLIC_KEY_PATH"


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
            "proof-of-human token and cannot run without a verifying key; starting "
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


def verify(token: str, verifying_key: str, *, audience: str | None = None) -> dict:
    """Return the token's claims, or raise TokenRejectedError.

    Every failure path refuses. There is deliberately no branch that returns claims
    for an unverified token, however malformed.
    """
    if not token:
        raise TokenRejectedError("no token presented")
    try:
        return dict(
            jwt.decode(
                token,
                verifying_key,
                algorithms=ALGORITHMS,
                audience=audience,
                options={"require": ["exp"]},
            )
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenRejectedError("token expired") from exc
    except jwt.InvalidAudienceError as exc:
        raise TokenRejectedError("token minted for a different audience") from exc
    except jwt.InvalidSignatureError as exc:
        raise TokenRejectedError("signature does not verify") from exc
    except jwt.MissingRequiredClaimError as exc:
        # A token with no expiry is a permanent credential, which is the thing the
        # short lifetime exists to avoid.
        raise TokenRejectedError("token has no expiry") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenRejectedError(f"invalid token: {type(exc).__name__}") from exc
