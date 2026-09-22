"""Run one analysis from start to finished file.

Sits between the client, which knows the service, and the chat, which knows
the user. Nothing here imports Chainlit: the awkward parts -- polling a job
that outlasts a chat turn, a submission that succeeds and then fails, a
result that has to be two different sizes for two different audiences --
are the parts worth testing, and a browser is not needed to test them.

**The submission is a receipt.** Measured: `POST /analysis` returned 200 and
the analysis then failed with `CONNECTION_FORCED - broker forced connection
closure`, visible only through `/status`. So `submit_*` returns an ID and
promises nothing, and `await_result` is where success or failure is decided.
"""

import contextlib
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from gsa import results as gsa_results
from gsa.client import AnalysisStatus, GsaClient, GsaError
from gsa.upload import Matrix, discard
from util.logging import logging

logger = logging.getLogger(__name__)

#: A PADOG run with 1,000 permutations took minutes in the measured run.
#: This is a ceiling on waiting, not an expectation.
DEFAULT_DEADLINE_SECONDS = 30 * 60
POLL_INTERVAL_SECONDS = 10.0

#: A hard ceiling on how many times the service is asked, independent of the
#: interval.
#:
#: Two bounds are needed, and the second one took two attempts. The
#: wall-clock deadline assumes each turn of the loop waits; the moment
#: sleeping does not sleep -- a patched `_sleep`, a zero interval from a
#: caller -- elapsed time never advances and the loop hammers the service.
#: My first fix derived the limit as `deadline / interval`, which computes
#: the bound from the quantity that is degenerate: `poll_interval=0` gave a
#: limit of 1.8 million, so the guard against a runaway loop was itself
#: unbounded in exactly the case it existed for. It left three pytest
#: processes spinning at 100% CPU on a shared host.
#:
#: At the real 10s interval a 30-minute deadline is 180 polls, so this is
#: generous.
MAX_POLLS = 2_000

#: The service's own default, and the method the round trip was measured
#: with. Camera is faster; choosing between them wants evidence rather than
#: preference, so it is recorded as an open question in spec 012.
DEFAULT_METHOD = "PADOG"

ProgressCallback = Callable[[AnalysisStatus], Awaitable[None]]


class AnalysisFailedError(GsaError):
    """The service accepted the analysis and then could not finish it."""


@dataclass(frozen=True)
class Finished:
    """A completed analysis, split by audience.

    Three fields, three destinations, kept apart deliberately:

        for_model   bounded, allow-listed, safe to put in a prompt
        links       the user's Pathway Browser view -- a capability URL
                    that must not reach the model
        table_path  the full table, for the user to download
    """

    analysis_id: str
    for_model: dict[str, object]
    links: list[tuple[str, str]]
    table_path: Path


async def submit_public_dataset(
    client: GsaClient,
    *,
    resource_id: str,
    dataset_id: str,
    factor: str,
    group1: str,
    group2: str,
    method: str = DEFAULT_METHOD,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
) -> str:
    """Load a public dataset and submit it. Returns an analysis ID.

    The matrix is downloaded here and handed straight to `submit`. It is
    1.2 MB for a small dataset and never leaves this function.
    """
    loading_id = await client.load_public_dataset(resource_id, dataset_id)
    # Both bounds, same as `await_result`. This loop had only the
    # wall-clock one -- the bug I had just fixed in its sibling, in the same
    # file, during the same review. Fixing a loop is not fixing the loops.
    deadline = time.monotonic() + deadline_seconds
    polls = 0
    while True:
        status = await client.loading_status(loading_id)
        polls += 1
        if status.failed:
            raise AnalysisFailedError(
                f"loading {dataset_id} failed: {status.description}"
            )
        if status.finished:
            break
        if polls >= MAX_POLLS or time.monotonic() > deadline:
            raise AnalysisFailedError(f"loading {dataset_id} did not finish in time")
        await _sleep(POLL_INTERVAL_SECONDS)

    summary = await client.dataset_summary(dataset_id)
    groups = summary.factors.get(factor)
    if not groups:
        raise AnalysisFailedError(
            f"{dataset_id} has no factor called {factor!r}. "
            f"It has: {', '.join(sorted(summary.factors)) or 'none'}."
        )
    _check_comparable(groups, factor, group1, group2)

    matrix = await client.download_matrix(dataset_id)
    return await client.submit(
        method=method,
        dataset_name=dataset_id,
        dataset_type=summary.type,
        matrix=matrix,
        samples=summary.samples,
        analysis_group=groups,
        group1=group1,
        group2=group2,
    )


async def submit_uploaded_matrix(
    client: GsaClient,
    *,
    matrix: Matrix,
    dataset_type: str,
    analysis_group: list[str],
    group1: str,
    group2: str,
    method: str = DEFAULT_METHOD,
) -> str:
    """Submit a user's own matrix, then delete their file.

    The file goes whether or not the analysis then succeeds: the service has
    its own copy by then, and this host does not have room to keep ours.
    `dataset_name` is deliberately not the user's filename -- that is one of
    the strings the disclosure rules exist to keep out of a prompt.
    """
    # Everything, including the validation, inside the `try`.
    #
    # The checks used to sit above it, so a wrong group name or a
    # miscounted label list -- the two mistakes a user is most likely to
    # make -- returned an error and left their file on disk. The disk leak
    # happened on exactly the paths people take most often.
    try:
        _check_comparable(analysis_group, "the grouping you gave", group1, group2)
        if len(analysis_group) != len(matrix.samples):
            raise AnalysisFailedError(
                f"You gave {len(analysis_group)} group labels for "
                f"{len(matrix.samples)} samples. There must be one label per "
                f"sample, in the order the columns appear."
            )
        return await client.submit(
            method=method,
            dataset_name="uploaded",
            dataset_type=dataset_type,
            matrix=matrix.text,
            samples=matrix.samples,
            analysis_group=analysis_group,
            group1=group1,
            group2=group2,
        )
    finally:
        discard(matrix.path)


def _check_comparable(groups: list[str], label: str, group1: str, group2: str) -> None:
    distinct = set(groups)
    missing = [g for g in (group1, group2) if g not in distinct]
    if missing:
        raise AnalysisFailedError(
            f"{', '.join(missing)} is not a value of {label}. "
            f"It has: {', '.join(sorted(distinct))}."
        )
    if group1 == group2:
        raise AnalysisFailedError("The two groups to compare must be different.")


#: How long a written result table is kept, and how much of them in total.
#:
#: The upload is deleted the moment it is submitted, and then the *output*
#: was kept forever -- each table is up to ~2 MB, on a host with 4.7 GB
#: free. Deleting the input and hoarding the output is not a disk policy.
#:
#: The window only has to outlast a user downloading their own results.
RESULT_MAX_AGE_SECONDS = 24 * 60 * 60
RESULT_DIR_MAX_BYTES = 200 * 1024 * 1024


def prune_results(
    out_dir: Path,
    *,
    max_age_seconds: float = RESULT_MAX_AGE_SECONDS,
    max_total_bytes: int = RESULT_DIR_MAX_BYTES,
    now: float | None = None,
) -> int:
    """Delete old result tables. Returns how many were removed.

    Called before each write, so the directory bounds itself and there is
    no cron job to forget. Age first, then oldest-first until the total
    fits: age alone leaves a burst of results unbounded, and size alone
    keeps one stale file forever on a quiet week.

    Only ever touches files this module wrote, matched by name. A cleanup
    that globs a directory it does not own is one misconfiguration away
    from deleting something else.
    """
    moment = time.time() if now is None else now
    ours = sorted(
        (path for path in out_dir.glob("reactome-gsa-*.tsv") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
    )

    removed = 0
    surviving: list[Path] = []
    for path in ours:
        if moment - path.stat().st_mtime > max_age_seconds:
            with contextlib.suppress(OSError):
                path.unlink()
                removed += 1
        else:
            surviving.append(path)

    total = sum(path.stat().st_size for path in surviving if path.exists())
    for path in surviving:
        if total <= max_total_bytes:
            break
        try:
            size = path.stat().st_size
            path.unlink()
        except OSError:
            continue
        total -= size
        removed += 1

    if removed:
        logger.info("pruned gsa result tables", extra={"removed": removed})
    return removed


async def await_result(
    client: GsaClient,
    analysis_id: str,
    *,
    out_dir: Path,
    on_progress: ProgressCallback | None = None,
    deadline_seconds: float = DEFAULT_DEADLINE_SECONDS,
    poll_interval: float = POLL_INTERVAL_SECONDS,
) -> Finished:
    """Poll until the analysis finishes, then write the table and return.

    Raises `AnalysisFailedError` when the service says so -- which is the only
    place it ever says so.
    """
    deadline = time.monotonic() + deadline_seconds
    max_polls = min(
        MAX_POLLS, max(2, int(deadline_seconds / max(poll_interval, 1.0)) + 2)
    )
    polls = 0

    while True:
        status = await client.analysis_status(analysis_id)
        polls += 1
        if on_progress is not None:
            await on_progress(status)
        if status.failed:
            raise AnalysisFailedError(status.description or "the analysis failed")
        if status.finished:
            # Terminal and not failed, so: complete. Asking the status
            # object rather than comparing the string again keeps one
            # definition of "done" across both loops and the dataclass.
            break
        if polls >= max_polls or time.monotonic() > deadline:
            raise AnalysisFailedError(
                f"the analysis did not finish within "
                f"{deadline_seconds / 60:.0f} minutes; it may still be "
                f"running as {analysis_id}"
            )
        await _sleep(poll_interval)

    parsed = gsa_results.parse(await client.result(analysis_id))
    if not parsed.pathways:
        # Complete, and yet nothing to report. Better to say so than to
        # hand back an empty file and an exact-sounding zero.
        raise AnalysisFailedError("the analysis finished but returned no pathway table")

    out_dir.mkdir(parents=True, exist_ok=True)
    prune_results(out_dir)
    table_path = out_dir / f"reactome-gsa-{analysis_id}.tsv"
    table_path.write_text(gsa_results.as_tsv(parsed))
    logger.info(
        "gsa result written",
        extra={
            "analysis": analysis_id,
            "pathways": len(parsed.pathways),
            "bytes": table_path.stat().st_size,
        },
    )

    return Finished(
        analysis_id=analysis_id,
        for_model=gsa_results.for_model(parsed),
        links=gsa_results.for_user(parsed),
        table_path=table_path,
    )


async def _sleep(seconds: float) -> None:
    # Indirected so a test can run the poll loop without waiting for it.
    import asyncio

    await asyncio.sleep(seconds)
