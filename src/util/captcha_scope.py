"""Which paths the captcha middleware lets through untouched.

Extracted from `bin/chat-fastapi.py` so it can be tested without importing the
app. That import is 15.8s even now that the graph is built at startup rather
than at import, which is too slow to sit in front of a rule this fiddly -- and
the rule is where a real bug lived: the middleware read `os.environ` for a
secret that comes from `get_secret`, so a deployment mounting it as a Docker
secret had the captcha silently disabled.

Spec 010 adds a route beside this one, so the rule is about to gain a case.
"""

from collections.abc import Iterable

STATIC_PREFIX = "/static"


def captcha_exempt_paths(chainlit_uri: str | None) -> list[str]:
    """Paths the captcha must not guard, or it would guard its own pages."""
    if not chainlit_uri:
        return ["/chat/"]
    return [
        "/chat/",
        f"{chainlit_uri}/verify_captcha",
        f"{chainlit_uri}/verify_captcha_page",
        f"{chainlit_uri}/static",
    ]


def is_captcha_exempt(
    path: str,
    *,
    chainlit_uri: str | None,
    captcha_configured: bool,
    extra_prefixes: Iterable[str] = (),
) -> bool:
    """True when `path` should skip the captcha check.

    `captcha_configured` is passed rather than read here, and it must come from
    the same place the signing secret does. Reading `os.environ` directly is
    what silently disabled the captcha for a deployment that mounted the secret.

    `extra_prefixes` exists for spec 010's answer endpoint, which verifies its
    own caller and must not be redirected to a captcha page.
    """
    if path in captcha_exempt_paths(chainlit_uri):
        return True
    if path.startswith(STATIC_PREFIX):
        return True
    if any(path.startswith(prefix) for prefix in extra_prefixes):
        return True
    # No captcha configured means no captcha to enforce. Deliberate: beta runs
    # without one because the Turnstile site key is bound to reactome.org.
    if not captcha_configured:
        return True
    # Anything outside the Chainlit app is not ours to guard.
    return bool(chainlit_uri) and not path.startswith(str(chainlit_uri))
