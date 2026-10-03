"""Gene-list analyses the chat has offered and the reader has not yet answered.

In process memory, not in Chainlit's `user_session`. The session is saved
into the thread's metadata as JSON: `cl.Action` objects became `null`, so
resuming a thread and typing "yes" raised on `None.remove()`, and five
60K-character messages would have been written into every thread's metadata
besides. Nothing here needs to outlive the process -- after a restart the
buttons say the offer has lapsed, and the reader sends the list again.

Plain data only, and bounded twice: offers per session, and sessions.
"""

import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from analysis.client import MAX_SUBMITTED_IDENTIFIERS

MAX_PER_SESSION = 5
MAX_SESSIONS = 2000


@dataclass(frozen=True)
class Proposal:
    #: The reader's message, for the model if they decline.
    text: str
    message_id: str
    identifiers: tuple[str, ...]
    #: `(name, id)` of each button, so they can be taken away once answered.
    actions: tuple[tuple[str, str], ...]


@dataclass
class _Session:
    offers: OrderedDict[str, Proposal] = field(default_factory=OrderedDict)
    #: The offer a typed "yes" means: the one just made, and only that.
    latest: str | None = None
    #: The chat has just told the reader to send their genes.
    invited: bool = False
    #: The last list the reader sent, for "analyse the list I gave you".
    last_list: tuple[str, ...] | None = None


@dataclass
class ProposalStore:
    max_per_session: int = MAX_PER_SESSION
    max_sessions: int = MAX_SESSIONS
    _sessions: OrderedDict[str, _Session] = field(default_factory=OrderedDict)

    def _session(self, session_id: str) -> _Session:
        found = self._sessions.get(session_id)
        if found is None:
            found = self._sessions[session_id] = _Session()
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
        self._sessions.move_to_end(session_id)
        return found

    @staticmethod
    def new_id() -> str:
        return uuid.uuid4().hex

    def put(
        self, session_id: str, proposal_id: str, proposal: Proposal
    ) -> list[Proposal]:
        """Keep an offer; returns any pushed out, whose buttons should go."""
        session = self._session(session_id)
        session.offers[proposal_id] = proposal
        session.latest = proposal_id
        evicted: list[Proposal] = []
        while len(session.offers) > self.max_per_session:
            evicted.append(session.offers.popitem(last=False)[1])
        return evicted

    def take(self, session_id: str, proposal_id: str | None) -> Proposal | None:
        """Claim an offer, once. No await between the lookup and the pop, so
        a double click or a click plus a typed "yes" cannot run it twice."""
        session = self._sessions.get(session_id)
        if session is None or not proposal_id:
            return None
        if session.latest == proposal_id:
            session.latest = None
        return session.offers.pop(proposal_id, None)

    def take_latest(self, session_id: str) -> str | None:
        """The offer a "yes" in this message would mean -- and forget it, so
        only the message straight after an offer can confirm it."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        latest, session.latest = session.latest, None
        return latest

    def remember_list(self, session_id: str, identifiers: list[str]) -> None:
        # Bounded like a submission: 2,000 sessions of 10K strings is not.
        self._session(session_id).last_list = tuple(
            identifiers[:MAX_SUBMITTED_IDENTIFIERS]
        )

    def last_list(self, session_id: str) -> list[str] | None:
        session = self._sessions.get(session_id)
        if session is None or session.last_list is None:
            return None
        return list(session.last_list)

    def invite(self, session_id: str) -> None:
        """The chat asked for a gene list; the next message may be one."""
        self._session(session_id).invited = True

    def take_invited(self, session_id: str) -> bool:
        """Whether this message answers that invitation -- and forget it, so
        only the message straight after counts."""
        session = self._sessions.get(session_id)
        if session is None:
            return False
        invited, session.invited = session.invited, False
        return invited

    def drop_session(self, session_id: str) -> None:
        """Forget everything about a session that has ended."""
        self._sessions.pop(session_id, None)


proposals = ProposalStore()
