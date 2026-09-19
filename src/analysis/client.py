"""Read a completed analysis result from the Analysis Service.

Beta, never production -- the standing instruction for this repository, and
`ANALYSIS_BASE_URL` exists so a deployment can point elsewhere without the
default ever being production.

Three things here were measured against beta rather than read from the
OpenAPI, because the OpenAPI is wrong or silent about all three.
"""

import os
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from util.logging import logging

logger = logging.getLogger(__name__)

BASE_URL_ENV = "ANALYSIS_BASE_URL"
DEFAULT_BASE_URL = "https://beta.reactome.org/AnalysisService"

# Measured 2026-09-19: the site's automation blocking answers a library
# user-agent with 403 and an HTML body, on every endpoint including
# /database/version. That looks exactly like an auth failure and is not one.
BROWSER_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0 Safari/537.36"
)

TIMEOUT_SECONDS = 20.0

#: Everything this module can conclude. `gone` is deliberately distinct from
#: `not_found`: one has an action attached, the other is a dead end.
Outcome = Literal["ok", "not_found", "gone", "unsupported", "failed"]


def base_url() -> str:
    return os.getenv(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")


@dataclass(frozen=True)
class Fetched:
    outcome: Outcome
    result: dict[str, Any] | None = None


def _headers(accept: str = "application/json") -> dict[str, str]:
    return {"User-Agent": BROWSER_USER_AGENT, "Accept": accept}


# `/database/version` answers with the bare number as text/plain and rejects
# an `Accept: application/json` request with **406 Not Acceptable** -- measured
# against beta 2026-09-19, after the mocked test passed because a mock cannot
# refuse a header it was never told about.
VERSION_ACCEPT = "text/plain, */*"


def is_gsa(result: dict[str, Any]) -> bool:
    """A ReactomeGSA result, which this does not model and will not summarise.

    GSA is a separate service on a different host with its own result shape.
    Recognising it costs one field check and prevents the worst outcome: a
    confident summary of something we do not actually understand.
    """
    summary = result.get("summary") or {}
    return bool(summary.get("gsaMethod") or summary.get("gsaToken"))


async def fetch_result(
    token: str, *, client: httpx.AsyncClient | None = None
) -> Fetched:
    """The completed result for `token`, or why there isn't one.

    404, 410 and 500 are all *negative outcomes*, not faults:

    - 404: no result for that token.
    - 410: the result was deleted by a new release. Distinct on purpose --
      the user can re-run, where 404 is a dead end.
    - 500: **undocumented, and what a malformed token actually returns**
      (measured against beta: `x` and `%20` both give 500). Treating it as a
      service fault would produce a `failed` state, or a retry loop against a
      service that will answer identically every time.
    """
    url = f"{base_url()}/token/{token}"
    owned = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
    try:
        response = await client.get(url, headers=_headers())
    except Exception as exc:
        logger.warning("analysis lookup failed for a token: %s", type(exc).__name__)
        return Fetched("failed")
    finally:
        if owned:
            await client.aclose()

    if response.status_code == 404 or response.status_code == 500:
        return Fetched("not_found")
    if response.status_code == 410:
        return Fetched("gone")
    if response.status_code != 200:
        logger.warning("analysis service answered %s", response.status_code)
        return Fetched("failed")

    try:
        result = response.json()
    except ValueError:
        # A 200 with a non-JSON body is the automation block, or a proxy.
        logger.warning("analysis service returned a non-JSON 200")
        return Fetched("failed")

    if is_gsa(result):
        return Fetched("unsupported")
    return Fetched("ok", result)


async def current_release(*, client: httpx.AsyncClient | None = None) -> str | None:
    """The release the Analysis Service is serving, or None.

    Read, never hardcoded (Principle V), because it is both the release the
    summary reports *and* the cache-invalidation key: the service deletes
    results on a release change, so a summary stored against an old release
    describes something that no longer exists.
    """
    owned = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
    try:
        response = await client.get(
            f"{base_url()}/database/version", headers=_headers(VERSION_ACCEPT)
        )
        if response.status_code != 200:
            logger.warning("release lookup answered %s", response.status_code)
            return None
        return response.text.strip() or None
    except Exception as exc:
        logger.warning("release lookup failed: %s", type(exc).__name__)
        return None
    finally:
        if owned:
            await client.aclose()
