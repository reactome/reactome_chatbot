"""Summaries kept so the same analysis reads the same way twice.

FR-014 wants a token to yield the same summary request after request.
Generation cannot provide that: measured on this repository, the same
question through the same surface twice scores 0.33 similarity. Storage can,
because an analysis result is a fixed artefact -- so stability here is
**reuse**, not determinism, and FR-015 requires the interface to say which.

**Keyed by `(token, release, tier)`, and all three matter.**

`release`, because the Analysis Service *deletes results on a new release*.
Without it a stored summary outlives the result it describes and we serve a
confident account of an analysis that no longer exists.

`tier`, because an aggregate summary and a disclosing one are different
artefacts. Sharing a key would let a reader who chose the default be served
a summary built from their identifiers, or the reverse -- one of them a
disclosure nobody asked for.

**In process, and lost on deploy.** Beta sets no `POSTGRES_LANGGRAPH_DB` and
LangGraph already falls back to `MemorySaver`, so nothing on that host
persists across a restart. This satisfies FR-014 within a process lifetime
and is honest only because `cached` tells the reader when a summary was
regenerated. A durable store is follow-up work, recorded in research D4.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
from time import time

#: Bounded because this lives for the life of the process. Summaries are a
#: couple of kilobytes, so this is a few megabytes at worst, and the oldest
#: is dropped rather than the newest refused -- a reader whose summary was
#: evicted regenerates, which is the documented behaviour anyway.
DEFAULT_MAX_ENTRIES = 512


@dataclass(frozen=True)
class Stored:
    text: str
    citations: tuple[tuple[str, str], ...]
    generated_at: float


@dataclass
class SummaryStore:
    max_entries: int = DEFAULT_MAX_ENTRIES
    _entries: OrderedDict[tuple[str, str, str], Stored] = field(
        default_factory=OrderedDict
    )

    def get(self, token: str, release: str, tier: str) -> Stored | None:
        key = (token, release, tier)
        found = self._entries.get(key)
        if found is not None:
            self._entries.move_to_end(key)
        return found

    def put(
        self,
        token: str,
        release: str,
        tier: str,
        text: str,
        citations: tuple[tuple[str, str], ...],
    ) -> Stored:
        """Store a summary. An empty one is never stored.

        A failed or abandoned generation leaves no text, and storing that
        would serve the emptiness back forever as though it were the answer.
        """
        stored = Stored(text=text, citations=citations, generated_at=time())
        if not text.strip():
            return stored
        key = (token, release, tier)
        self._entries[key] = stored
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return stored

    def __len__(self) -> int:
        return len(self._entries)
