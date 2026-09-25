"""Talk to ReactomeGSA, the service that runs gene set analysis.

A different service from the Analysis Service, doing a different thing:

    AnalysisService  over-representation over a LIST of identifiers
    ReactomeGSA      gene set analysis over an EXPRESSION MATRIX

Everything asserted here was measured against the live service on
2026-09-21, because the swagger is silent on the parts that matter.

**The matrix is always inline.** `POST /analysis` requires
`datasets[].data`, the whole tab-delimited matrix as a string. There is no
by-reference variant, not even for a dataset the service itself just loaded.
Measured: 1.2 MB for the 16-sample melanoma example, a 1.5 MB submission,
and a 2.0 MB result holding 2,679 pathways. Nothing of that size may reach
the model or cross an MCP tool call, so this module exists to keep it on the
server.

**A public dataset needs no upload.** `POST /data/load/{resourceId}` takes an
identifier -- Expression Atlas, Single Cell Expression Atlas, GREIN, GEO, or
the bundled examples -- and `GET /data/summary/{id}` then returns the sample
IDs and factors a user needs in order to name a comparison.

**A 200 from /analysis means accepted, not succeeded.** Measured: a
submission returned 200 and then failed with `CONNECTION_FORCED - broker
forced connection closure` while the service was updating its Reactome
version. The failure was visible only through `GET /status`. Treat the
submission as a receipt and the status as the truth.

**No browser user-agent here, unlike `analysis/client.py`.** That one needs
one because reactome.org sits behind automation blocking that answers a
library client with 403. `gsa.reactome.org` serves hypercorn directly and
answers `python-httpx` with 200 -- verified rather than assumed, because the
opposite mistake (testing only with curl, which is exempt from that block)
is how a sibling service was published and believed to work.
"""

import os
import re
from dataclasses import dataclass
from typing import Any

import httpx

from util.logging import logging

logger = logging.getLogger(__name__)

BASE_URL_ENV = "REACTOME_GSA_URL"
DEFAULT_BASE_URL = "https://gsa.reactome.org/0.1"

#: Submitting carries the matrix and the service answers only once it is
#: queued, so it needs far longer than a catalogue read.
TIMEOUT_SECONDS = 30.0
SUBMIT_TIMEOUT_SECONDS = 180.0

#: Identifiers are interpolated into URL paths. A dataset ID comes from the
#: model, which means it comes from the user, so it is constrained to what
#: real ones look like -- `EXAMPLE_MEL_RNA`, `GSE12345`, `E-MTAB-2770`.
#: Anything else is rejected rather than escaped: a value containing `/`
#: addresses a different endpoint, which is a bypass rather than a 404.
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class GsaError(RuntimeError):
    """The service refused, or answered in a shape this code cannot use."""


class GsaNotReadyError(GsaError):
    """The analysis has not finished, so there is no result yet.

    `GET /result` answers 406 for this, which is an expected state on the
    happy path rather than a fault. Collapsing it into a generic failure
    would make "still running" indistinguishable from "broken", and the
    caller polls on exactly that distinction.
    """


def _identifier_from(body: str, kind: str) -> str:
    """Read an ID out of a response body, refusing anything else.

    **Both POSTs answer `text/plain`: the bare identifier, unquoted.** The
    swagger says so (`produces: text/plain`, example `Analysis00371643`),
    and the live service returns a bare UUID.

    The first version of this function said the opposite -- "the bare ID as a
    quoted JSON string" -- and called `response.json()` on it. That was never
    measured, and it meant every submission failed: the job was accepted with
    a 200, the ID could not be parsed, and the user was told nothing had run.
    The upload feature shipped that way and was never able to work. Every
    test stubbed `submit()` at the method level, returning a Python string,
    so the one line that was wrong was the one line no test executed.

    Quotes are still stripped, so a JSON-quoted body would also be accepted;
    what matters is that the result is validated as an identifier before it
    becomes a URL path segment. An error body, an HTML page from a proxy, or
    an empty response produces a plausible string that would otherwise fail
    far from here, as a 404 that reads like a missing analysis.
    """
    value = body.strip().strip('"').strip()
    if not _IDENTIFIER.match(value):
        raise GsaError(f"{kind} is not a valid identifier: {value[:120]!r}")
    return value


def base_url() -> str:
    return os.environ.get(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")


def _checked(kind: str, value: str) -> str:
    if not _IDENTIFIER.match(value):
        raise GsaError(f"{kind} is not a valid identifier: {value!r}")
    return value


#: Statuses that mean the service has stopped working on it. Listed once so
#: a loop cannot disagree with the dataclass about what "done" means.
TERMINAL_STATUSES = frozenset({"complete", "failed"})


@dataclass(frozen=True)
class LoadingStatus:
    """Progress of `POST /data/load`, which is not instant."""

    status: str
    description: str
    completed: float
    dataset_id: str | None

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def failed(self) -> bool:
        return self.finished and self.status != "complete"


@dataclass(frozen=True)
class DatasetSummary:
    """What a user needs in order to name a comparison.

    Small enough to show: the measured example is 16 samples and three
    factors. The matrix it describes is 1.2 MB and is not here.
    """

    dataset_id: str
    title: str
    type: str
    samples: list[str]
    #: factor name -> that factor's value for each sample, in sample order
    factors: dict[str, list[str]]

    def groups(self, factor: str) -> list[str]:
        return sorted(set(self.factors.get(factor, [])))


@dataclass(frozen=True)
class AnalysisStatus:
    status: str
    description: str
    completed: float

    @property
    def finished(self) -> bool:
        return self.status in TERMINAL_STATUSES

    @property
    def failed(self) -> bool:
        # Anything terminal that is not success. Derived rather than
        # `== "failed"`, so a status added to TERMINAL_STATUSES later is
        # treated as a failure by default instead of being silently
        # reported as a completed analysis with no results.
        return self.finished and self.status != "complete"


class GsaClient:
    """Thin async client. Holds no state about a running analysis."""

    def __init__(
        self,
        url: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._base = (url or base_url()).rstrip("/")
        # Injectable so the HTTP layer itself can be tested. Without it the
        # only way to test this client was to stub its methods, which is how
        # a `response.json()` on a `text/plain` body reached production.
        self._transport = transport

    async def _get(self, path: str, *, timeout: float = TIMEOUT_SECONDS) -> Any:
        async with httpx.AsyncClient(
            timeout=timeout, transport=self._transport
        ) as client:
            response = await client.get(f"{self._base}{path}")
        if response.status_code == 406:
            raise GsaNotReadyError(f"GET {path}: analysis is not complete")
        if response.status_code != 200:
            raise GsaError(f"GET {path} returned {response.status_code}")
        return response.json()

    async def methods(self) -> list[dict[str, Any]]:
        result = await self._get("/methods")
        return list(result) if isinstance(result, list) else []

    async def load_public_dataset(self, resource_id: str, dataset_id: str) -> str:
        """Ask the service to load a public dataset. Returns a loading ID.

        The body is a list of name/value parameters rather than an object --
        the shape the service wants, confirmed by a successful load.
        """
        _checked("resource", resource_id)
        _checked("dataset", dataset_id)
        body = [{"name": "dataset_id", "value": dataset_id}]
        async with httpx.AsyncClient(
            timeout=TIMEOUT_SECONDS, transport=self._transport
        ) as client:
            response = await client.post(
                f"{self._base}/data/load/{resource_id}", json=body
            )
        if response.status_code not in (200, 202):
            raise GsaError(
                f"loading {dataset_id} from {resource_id} returned "
                f"{response.status_code}: {response.text[:200]}"
            )
        return _identifier_from(response.text, "loading id")

    async def loading_status(self, loading_id: str) -> LoadingStatus:
        data = await self._get(f"/data/status/{_checked('loading id', loading_id)}")
        return LoadingStatus(
            status=str(data.get("status", "")),
            description=str(data.get("description", "")),
            completed=float(data.get("completed") or 0.0),
            dataset_id=data.get("dataset_id"),
        )

    async def dataset_summary(self, dataset_id: str) -> DatasetSummary:
        data = await self._get(f"/data/summary/{_checked('dataset', dataset_id)}")
        factors = {
            str(entry.get("name")): [str(v) for v in entry.get("values", [])]
            for entry in data.get("sample_metadata") or []
            if entry.get("name")
        }
        return DatasetSummary(
            dataset_id=dataset_id,
            title=str(data.get("title") or dataset_id),
            type=str(data.get("type") or ""),
            samples=[str(s) for s in data.get("sample_ids") or []],
            factors=factors,
        )

    async def download_matrix(self, dataset_id: str) -> str:
        """The expression matrix, as a tab-delimited string.

        Measured at 1.2 MB for a 16-sample dataset. It exists only to be
        handed straight back to `submit`; it must not be logged, returned to
        a caller that could show it, or put in a prompt. `format=expr` is
        required -- omitting it is a 400, and `tsv` is not one of the
        accepted values.
        """
        url = f"{self._base}/data/download/{_checked('dataset', dataset_id)}"
        async with httpx.AsyncClient(
            timeout=SUBMIT_TIMEOUT_SECONDS, transport=self._transport
        ) as client:
            response = await client.get(url, params={"format": "expr"})
        if response.status_code != 200:
            raise GsaError(f"downloading {dataset_id} returned {response.status_code}")
        logger.info(
            "gsa matrix downloaded",
            extra={"dataset": dataset_id, "bytes": len(response.content)},
        )
        return response.text

    async def submit(
        self,
        *,
        method: str,
        dataset_name: str,
        dataset_type: str,
        matrix: str,
        samples: list[str],
        analysis_group: list[str],
        group1: str,
        group2: str,
    ) -> str:
        """Start an analysis. Returns an analysis ID.

        The ID is a *receipt*. The analysis may still fail, and it will say
        so through `analysis_status`, not here.
        """
        body = {
            "methodName": method,
            "datasets": [
                {
                    "name": dataset_name,
                    "type": dataset_type,
                    "data": matrix,
                    "design": {
                        "samples": samples,
                        "analysisGroup": analysis_group,
                        "comparison": {"group1": group1, "group2": group2},
                    },
                }
            ],
        }
        async with httpx.AsyncClient(
            timeout=SUBMIT_TIMEOUT_SECONDS, transport=self._transport
        ) as client:
            response = await client.post(f"{self._base}/analysis", json=body)
        if response.status_code != 200:
            raise GsaError(
                f"submitting returned {response.status_code}: {response.text[:200]}"
            )
        analysis_id = _identifier_from(response.text, "analysis id")
        logger.info(
            "gsa analysis submitted",
            extra={
                "analysis": analysis_id,
                "method": method,
                "matrix_bytes": len(matrix),
            },
        )
        return analysis_id

    async def analysis_status(self, analysis_id: str) -> AnalysisStatus:
        data = await self._get(f"/status/{_checked('analysis', analysis_id)}")
        return AnalysisStatus(
            status=str(data.get("status", "")),
            description=str(data.get("description", "")),
            completed=float(data.get("completed") or 0.0),
        )

    async def result(self, analysis_id: str) -> dict[str, Any]:
        """The finished result: ~2 MB, 2,679 pathways in the measured run.

        Returned whole because the caller needs the table to write a file.
        It must be bounded before any of it reaches a model.
        """
        data = await self._get(
            f"/result/{_checked('analysis', analysis_id)}",
            timeout=SUBMIT_TIMEOUT_SECONDS,
        )
        return data if isinstance(data, dict) else {}
