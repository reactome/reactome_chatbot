"""The Chainlit side of running a gene set analysis.

Thin on purpose. Every decision -- what a usable reply is, what the person
is told, what the model may see -- lives in `gsa.chat` and is tested there.
What is left here is the part a browser has to verify: reading an attached
file, asking a follow-up question, streaming progress, sending a download.

**Why this is not a tool the agent calls.** The run takes minutes, which is
longer than a chat turn, and the matrix is 1.2 MB, which must never enter
the model's context. A tool call would put the model in the middle of both
problems. So an attached file routes here *instead of* the graph, and the
model is handed only the bounded result at the end, if the user asks for a
summary at all.
"""

from pathlib import Path
from typing import Any, Protocol

from gsa import chat
from gsa.client import GsaClient
from gsa.job import AnalysisFailedError, await_result, submit_uploaded_matrix
from gsa.upload import UploadRejectedError, discard, validate
from util.logging import logging

logger = logging.getLogger(__name__)

#: Where result tables are written. Bounded by `prune_results` on each run.
#:
#: Not a bare path under a world-writable `/tmp`: these are a user's own
#: analysis results, and on a shared host anyone could pre-create the
#: directory or replace a file in it. `results_dir()` creates it 0700 and
#: owned by this process, and `GSA_RESULTS_DIR` lets a deployment put it
#: somewhere with a real quota.
RESULTS_DIR_ENV = "GSA_RESULTS_DIR"


def results_dir() -> Path:
    """The results directory, created private to this user."""
    import os
    import tempfile

    configured = os.environ.get(RESULTS_DIR_ENV)
    base = (
        Path(configured) if configured else Path(tempfile.gettempdir()) / "reactome-gsa"
    )
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    return base


#: Extensions we will try to read as a matrix. Anything else attached is
#: almost certainly meant for a different conversation.
MATRIX_SUFFIXES = {".tsv", ".csv", ".txt"}


class Attachment(Protocol):
    """The part of a Chainlit file element this needs."""

    name: str
    path: str


def matrix_attachment(elements: list[Any] | None) -> Attachment | None:
    """The first attachment that could be an expression matrix, if any.

    Returns None for a message with no attachments, which is every ordinary
    message -- so the caller can use this as the routing decision without
    knowing anything about Chainlit.
    """
    for element in elements or []:
        name = getattr(element, "name", "") or ""
        path = getattr(element, "path", None)
        if path and Path(name).suffix.lower() in MATRIX_SUFFIXES:
            found: Attachment = element
            return found
    return None


async def run_analysis(
    attachment: Attachment,
    *,
    ask_for_grouping: Any,
    send: Any,
    update_progress: Any,
    send_file: Any,
    client: GsaClient | None = None,
) -> None:
    """Validate, ask for groups, submit, poll, deliver.

    The callables are passed in rather than imported so this can be driven
    without Chainlit. They are the four things a chat has to be able to do:
    ask a question and wait, say something, revise what was said, and hand
    over a file.
    """
    path = Path(attachment.path)

    try:
        matrix = validate(path)
    except UploadRejectedError as refusal:
        discard(path)
        await send(str(refusal))
        return

    await send(chat.describe_matrix(matrix))

    reply = await ask_for_grouping()
    if not reply:
        discard(path)
        await send(
            "No labels arrived, so I have not run anything. The file is deleted."
        )
        return

    try:
        grouping = chat.parse_grouping(reply, len(matrix.samples))
    except chat.ReplyUnusableError as unusable:
        discard(path)
        await send(f"{unusable} Send the file again when you are ready.")
        return

    gsa = client or GsaClient()
    try:
        # `submit_uploaded_matrix` deletes the file itself, on every path.
        analysis_id = await submit_uploaded_matrix(
            gsa,
            matrix=matrix,
            dataset_type="rnaseq_counts",
            analysis_group=grouping.labels,
            group1=grouping.group1,
            group2=grouping.group2,
        )
    except AnalysisFailedError as failure:
        await send(f"I could not start the analysis: {failure}")
        return
    except Exception:
        logger.exception("gsa submission failed")
        await send("I could not reach the analysis service. Nothing was run.")
        return

    await send(
        f"Started. This usually takes a few minutes — "
        f"comparing **{grouping.group1}** with **{grouping.group2}**."
    )

    try:
        finished = await await_result(
            gsa,
            analysis_id,
            out_dir=results_dir(),
            on_progress=lambda status: update_progress(chat.describe_progress(status)),
        )
    except AnalysisFailedError as failure:
        # The service accepts and then fails, and says so only here.
        await send(f"The analysis did not finish: {failure}")
        return
    except Exception:
        logger.exception("gsa analysis failed", extra={"analysis": analysis_id})
        await send("Something went wrong while waiting for the analysis.")
        return

    # `finished.for_model` is deliberately not used here.
    #
    # Nothing about this analysis reaches OpenAI. The result is described
    # from the table and handed over as a file, both of which are the
    # user's own data going back to the user. A model-written summary is
    # spec 012's Story 3 the other way round -- opt-in, behind the existing
    # warning -- and until that exists, the honest state is that the
    # allow-listed view is computed and sent nowhere.
    #
    # It is computed rather than skipped so the disclosure rules stay
    # exercised by the tests; if a summary is added later, the bounded view
    # is what it must be given, not the result.
    await send(chat.describe_result(finished))
    await send_file(finished.table_path)
