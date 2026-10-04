"""How much of a conversation goes back to the model each turn.

None of it was trimmed: every turn resent the whole thread to the rephraser,
the answer model and the live loop. A thread of 8,000-character messages
overflowed the context window at turn 47, short questions at turn 138, and a
message of 8,000 emoji -- about 24k tokens, under every character cap -- by
turn 6; after that every turn failed, and a logged-in thread resumed broken
(review, area 3).

Recent turns are kept, by count and by size, and so is a handoff's seeded
first turn: it is the summary and data the whole conversation is about, and
the rules for reading them.
"""

from collections.abc import Sequence

from langchain_core.messages import BaseMessage

MAX_MESSAGES = 40
#: Characters, not tokens: no tokenizer is needed to bound it, and at ~1-4
#: characters a token this stays well inside a 128k window.
MAX_CHARS = 60_000
SEED_MARK = "reactome_analysis_seed"


def _size(message: BaseMessage) -> int:
    return len(str(message.content))


def recent(history: Sequence[BaseMessage] | None) -> list[BaseMessage]:
    """The recent part of a conversation, plus a seeded first turn."""
    messages = list(history or [])
    seeded = (
        messages[:2]
        if any(getattr(m, "additional_kwargs", {}).get(SEED_MARK) for m in messages[:2])
        else []
    )
    rest = messages[len(seeded) :]
    budget = MAX_CHARS - sum(_size(m) for m in seeded)
    kept: list[BaseMessage] = []
    for message in reversed(rest):
        if len(kept) >= MAX_MESSAGES or _size(message) > budget:
            break
        kept.append(message)
        budget -= _size(message)
    return seeded + kept[::-1]
