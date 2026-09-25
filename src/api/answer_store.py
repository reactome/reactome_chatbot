"""Answers the search page was shown, kept briefly so they can be continued.

Spec 013, Story 2. "Continue in chat" beside a search-page answer must open
the chat on *that* answer. It cannot be regenerated: the same question
through the same endpoint scores ~0.33 similarity run to run, so the reader
would be greeted with different text from the one they clicked from.

**Keyed by a per-answer ID, not by the question.** Two readers asking the
same question get different answers; a cache keyed by question would let
one reader continue another's. The endpoint issues an ID in its `done`
event, and that is what the website hands back.

**What is kept is exactly what the reader saw** -- the text after anchors and
the trailing sources list were stripped, and the citations as sent -- not
the raw model output.

Bounded and process-local, like the analysis summary cache. An answer is
public pathway text, so keeping it briefly carries no disclosure; the window
only has to outlast a reader deciding to click.
"""

import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field

DEFAULT_TTL_SECONDS = 60 * 60
DEFAULT_MAX_ENTRIES = 2048


@dataclass(frozen=True)
class StoredAnswer:
    question: str
    text: str
    #: `(identifier, display_name)`, where identifier is an st_id or a URL.
    citations: tuple[tuple[str, str], ...]
    release: int | None
    created_at: float


@dataclass
class AnswerStore:
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    max_entries: int = DEFAULT_MAX_ENTRIES
    _entries: OrderedDict[str, StoredAnswer] = field(default_factory=OrderedDict)

    def put(self, answer: StoredAnswer) -> str | None:
        """Keep an answer; returns its ID, or None if there was nothing to keep.

        An empty answer is never stored: continuing it would open a chat on
        nothing, and say it was the reader's answer.
        """
        if not answer.text.strip():
            return None
        answer_id = secrets.token_urlsafe(18)
        self._entries[answer_id] = answer
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return answer_id

    def get(self, answer_id: str, *, now: float | None = None) -> StoredAnswer | None:
        found = self._entries.get(answer_id)
        if found is None:
            return None
        if (time.time() if now is None else now) - found.created_at > self.ttl_seconds:
            del self._entries[answer_id]
            return None
        return found


#: Written by `/api/answer`, read by `/api/handoff`.
answers = AnswerStore()
