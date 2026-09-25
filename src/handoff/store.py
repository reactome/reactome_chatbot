"""Handoffs waiting to be claimed by a chat tab (spec 013).

A handoff is minted by the website for a summary the reader has just seen,
and claimed by the chat tab it opens. It carries a **copy of that summary's
text**, taken when it was minted. Two reasons:

- The chat must continue the summary the reader saw, not regenerate one
  (FR-002). Generation here is not reproducible -- the same question scores
  ~0.33 similarity run to run -- so regenerating would greet the reader with
  different text from the one they clicked from.
- The summary cache is bounded and process-local. Copying means the chat
  still has the text if that cache has evicted it in the minutes between
  mint and claim, and means the chat never reaches into another endpoint's
  private state.

**Redeemable repeatedly, for a short window** (FR-007). A single-use handoff
would give an empty chat when the tab is reloaded. Instead it can be claimed
any number of times until it expires. What it grants is read access to one
summary the reader already has, so the window is short and the ID cannot be
guessed.

**In process, lost on deploy**, like the summary cache it copies from. The
API that mints and the Chainlit app that claims run in the same process
(`mount_chainlit` in `bin/chat-fastapi.py`), so module state is shared.
"""

import secrets
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Literal

from analysis.disclosure import Tier

#: Long enough to open the tab and reload it; short enough that a link which
#: escapes -- pasted into a message, left in history -- stops working soon.
DEFAULT_TTL_SECONDS = 15 * 60

#: Bounded because it lives for the life of the process.
DEFAULT_MAX_ENTRIES = 2048


@dataclass(frozen=True)
class AnalysisHandoff:
    kind: Literal["analysis"]
    #: The analysis token the summary describes.
    token: str
    release: str
    #: The disclosure tier the summary was *built* at. The chat continues at
    #: exactly this tier (FR-003) -- never wider than the reader agreed to.
    tier: Tier
    summary: str
    citations: tuple[tuple[str, str], ...]
    created_at: float


@dataclass(frozen=True)
class SearchHandoff:
    """A search-page answer (Story 2). No tier: it is public pathway text."""

    kind: Literal["search"]
    question: str
    #: The answer exactly as the page rendered it.
    summary: str
    #: `(st_id or url, display_name)`, as the page received them.
    citations: tuple[tuple[str, str], ...]
    created_at: float


#: Two types rather than one with optional fields, so a search handoff cannot
#: be built carrying a disclosure tier that means nothing for it.
Handoff = AnalysisHandoff | SearchHandoff


def new_id() -> str:
    """192 bits, URL-safe. Unguessable, and fits `handoff.window._ID`."""
    return secrets.token_urlsafe(24)


@dataclass
class HandoffStore:
    ttl_seconds: float = DEFAULT_TTL_SECONDS
    max_entries: int = DEFAULT_MAX_ENTRIES
    _entries: OrderedDict[str, Handoff] = field(default_factory=OrderedDict)

    def put(self, handoff: Handoff) -> str:
        handoff_id = new_id()
        self._entries[handoff_id] = handoff
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return handoff_id

    def get(self, handoff_id: str, *, now: float | None = None) -> Handoff | None:
        """The handoff, or None if unknown or expired. Expired ones are dropped."""
        found = self._entries.get(handoff_id)
        if found is None:
            return None
        if (time.time() if now is None else now) - found.created_at > self.ttl_seconds:
            del self._entries[handoff_id]
            return None
        return found


#: Shared by the API that mints and the chat hook that claims.
handoffs = HandoffStore()
