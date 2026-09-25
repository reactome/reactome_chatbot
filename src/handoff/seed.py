"""What a claimed handoff puts into the conversation (spec 013).

Two things, for two audiences, the same split `gsa` makes:

- **The reader** sees the summary they clicked from, verbatim.
- **The model** gets that summary as its own previous turn, *plus the data it
  was built from*, so a follow-up like "which of these involve TP53?" can be
  answered from the analysis rather than from general knowledge.

The seeded turn has the shape every real turn has -- `[HumanMessage,
AIMessage]` -- so nothing downstream has to know it was not typed.

**The data is rebuilt at the handoff's tier, with the same functions the
summary endpoint uses** (`for_tier`, `prompt_input`, `fetch_not_found`), not a
copy of them. The chat model therefore receives exactly the allow-listed view
the summary was built from and nothing wider (FR-003). Rebuilding is safe
where regenerating the *text* would not be: an analysis result is a fixed
artefact, so the same token at the same tier yields the same data.
"""

import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from analysis.client import Fetched, fetch_not_found, fetch_result
from analysis.disclosure import for_tier
from analysis.summarise import prompt_input
from handoff.store import DEFAULT_TTL_SECONDS, Handoff

#: What the reader asked for on the website, stated as what happened.
HUMAN_TURN = (
    "Summarise my Reactome pathway analysis. (Asked on the analysis results page.)"
)

FetchResult = Callable[[str], Awaitable[Fetched]]
FetchUnmatched = Callable[[str], Awaitable[list[str] | None]]


async def analysis_data(
    handoff: Handoff,
    *,
    fetch: FetchResult = fetch_result,
    fetch_unmatched: FetchUnmatched = fetch_not_found,
) -> dict[str, Any] | None:
    """The allow-listed data the summary was built from, or None if gone.

    None when the Analysis Service no longer has the result -- it deletes
    results on a new release -- in which case the chat still continues the
    summary, and says it cannot see the underlying numbers.
    """
    fetched = await fetch(handoff.token)
    if fetched.outcome != "ok" or fetched.result is None:
        return None
    data = prompt_input(for_tier(fetched.result, handoff.tier))
    if handoff.tier == "identifiers":
        # Only at the tier the reader chose on the website, and only then.
        unmatched = await fetch_unmatched(handoff.token)
        if unmatched:
            data["identifiers_not_found_names"] = unmatched
    return data


def seeded_turn(handoff: Handoff, data: dict[str, Any] | None) -> list[BaseMessage]:
    """The conversation turn the chat starts from."""
    if data is None:
        appendix = (
            "\n\n(The analysis result behind this summary is no longer available "
            "from Reactome, so only the summary above is known.)"
        )
    else:
        appendix = (
            "\n\nThe analysis data this summary was built from "
            f"(disclosure tier: {handoff.tier}):\n"
            f"```json\n{json.dumps(data, indent=1, default=str)}\n```"
        )
    return [HumanMessage(HUMAN_TURN), AIMessage(handoff.summary + appendix)]


def shown_to_reader(handoff: Handoff) -> str:
    """What the reader sees when the tab opens. Their own summary, verbatim."""
    return (
        "Continuing from your analysis summary:\n\n"
        f"{handoff.summary}\n\n"
        "---\nAsk a follow-up question about this analysis."
    )


UNAVAILABLE = (
    "I couldn't load the summary you came from — the link may have expired "
    f"(they last {DEFAULT_TTL_SECONDS // 60:.0f} minutes). You can still ask me "
    "anything here, or go back to the analysis page and choose *Continue in "
    "chat* again."
)
