"""Read a completed analysis result from the Analysis Service.

Beta, never production -- the standing instruction for this repository, and
`ANALYSIS_BASE_URL` exists so a deployment can point elsewhere without the
default ever being production.

Three things here were measured against beta rather than read from the
OpenAPI, because the OpenAPI is wrong or silent about all three.
"""

import os
import re
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

# An analysis token is base64 with percent-encoded padding, as the service
# issues it: `MjAyNjA5MTkxODExNDJfMTE%3D`. Nothing else is accepted, and the
# reason is a disclosure bypass rather than tidiness.
#
# The token is interpolated into a URL path, so a caller-supplied
# `<real token>/notFound` addresses `GET /token/{token}/notFound` -- the
# endpoint that returns the user's *unmatched identifiers*. That is the
# identifier tier, which the aggregate tier promises never to fetch, reached
# by a caller who only ever asked for an aggregate summary. Demonstrated
# against beta on 2026-09-19; it returned the submitted identifiers.
#
# Rejecting here rather than escaping: the token arrives already
# percent-encoded, so quoting it again would break every valid token, and
# there is nothing to gain by letting an invalid one reach the network. A
# rejected token yields `not_found`, which is exactly what the service
# returns for a malformed one anyway (measured: 500, mapped to not_found).
_TOKEN = re.compile(r"\A[A-Za-z0-9_-]{1,200}(?:%3D|=){0,2}\Z", re.IGNORECASE)


def is_well_formed(token: str) -> bool:
    return bool(_TOKEN.match(token))


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


#: The analysis types this service does not model. Read from the API's own
#: enum on 2026-09-20 -- `ExternalAnalysisSummary.type` is
#: SPECIES_COMPARISON, OVERREPRESENTATION, EXPRESSION, GSA_REGULATION,
#: GSA_STATISTICS, GSVA -- rather than guessed at, because the first version
#: recognised GSA by its `gsaMethod` field alone and a result carrying the
#: type without that field would have been summarised confidently. That is
#: precisely the outcome D8 exists to prevent.
GSA_TYPES = frozenset({"GSA_REGULATION", "GSA_STATISTICS", "GSVA"})


def is_gsa(result: dict[str, Any]) -> bool:
    """A ReactomeGSA result, which this does not model and will not summarise.

    GSA is a separate service on a different host with its own result shape.
    Recognising it prevents the worst outcome: a confident summary of
    something we do not actually understand.

    Two independent signals, because either alone has a gap. The fields catch
    a result whose type is unset or new; the type catches one that carries no
    `gsaMethod`. Neither is known to be sufficient on its own and there is no
    cost to checking both.
    """
    summary = result.get("summary") or {}
    if summary.get("gsaMethod") or summary.get("gsaToken"):
        return True
    return str(summary.get("type") or "").upper() in GSA_TYPES


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
    if not is_well_formed(token):
        # Not an error the caller must special-case: a malformed token is a
        # normal negative outcome (FR-009), and this is the same answer the
        # service gives for one.
        logger.info("rejected a malformed analysis token before any request")
        return Fetched("not_found")
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

    if not isinstance(result, dict):
        # `/token/{t}/notFound` returns a list, and so would anything else a
        # path escape reached. Before this check that raised AttributeError
        # straight out of a function whose contract is to never raise.
        logger.warning(
            "analysis service returned %s, not an object", type(result).__name__
        )
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


async def fetch_not_found(
    token: str, *, limit: int = 50, client: httpx.AsyncClient | None = None
) -> list[str] | None:
    """The user's unmatched identifiers. **Identifier tier only.**

    This is the endpoint that returns the reader's own submitted data, so it
    is a separate function taking a separate decision rather than a flag on
    `fetch_result`. A flag acquires a default, and a default here is a
    disclosure nobody chose.

    Bounded: a list of thousands would be sent to a model provider and tell
    the reader nothing a sample does not.
    """
    if not is_well_formed(token):
        return None
    owned = client is None
    client = client or httpx.AsyncClient(timeout=TIMEOUT_SECONDS)
    try:
        response = await client.get(
            f"{base_url()}/token/{token}/notFound",
            headers=_headers(),
            params={"pageSize": limit, "page": 1},
        )
        if response.status_code != 200:
            return None
        payload = response.json()
    except Exception as exc:
        logger.warning("not-found lookup failed: %s", type(exc).__name__)
        return None
    finally:
        if owned:
            await client.aclose()
    if not isinstance(payload, list):
        return None
    return [
        str(entry["id"])
        for entry in payload
        if isinstance(entry, dict) and entry.get("id")
    ][:limit]
