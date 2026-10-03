"""Uploaded matrices waiting for their sample labels.

The labels used to be asked for with Chainlit's `AskUserMessage`, which is
tied to the socket open when it was asked. A reconnect -- a laptop sleeping,
a network change, a backgrounded phone tab -- left it waiting on a socket
that was gone: the reader's "control, control, treated, treated" was answered
by the model as an ordinary question, and ten minutes later the chat said no
labels had arrived (review, area 2). Now the matrix waits here, and the
reader's next message that reads as labels runs it.

In process memory, like the gene-list offers: plain data, bounded, gone on
restart -- after which the reader attaches the file again.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass, field

from gsa.upload import Matrix, discard

#: How long an uploaded matrix waits for labels.
WAIT_SECONDS = 30 * 60
MAX_SESSIONS = 500


@dataclass
class PendingMatrices:
    wait_seconds: float = WAIT_SECONDS
    max_sessions: int = MAX_SESSIONS
    _waiting: OrderedDict[str, tuple[Matrix, float]] = field(
        default_factory=OrderedDict
    )

    def put(self, session_id: str, matrix: Matrix, now: float | None = None) -> None:
        """Keep a matrix for this session; one per session, newest wins.

        Putting the same matrix back -- after a reply that was not usable
        labels -- must not delete its file, which dropping the old entry would.
        """
        previous = self._waiting.pop(session_id, None)
        if previous is not None and previous[0].path != matrix.path:
            discard(previous[0].path)
        self._waiting[session_id] = (matrix, time.time() if now is None else now)
        while len(self._waiting) > self.max_sessions:
            _, (old, _) = self._waiting.popitem(last=False)
            discard(old.path)

    def peek(self, session_id: str, now: float | None = None) -> Matrix | None:
        """The matrix waiting for this session, if it has not expired."""
        found = self._waiting.get(session_id)
        if found is None:
            return None
        matrix, since = found
        if (time.time() if now is None else now) - since > self.wait_seconds:
            self.drop(session_id)
            return None
        return matrix

    def take(self, session_id: str, now: float | None = None) -> Matrix | None:
        matrix = self.peek(session_id, now)
        if matrix is not None:
            self._waiting.pop(session_id, None)
        return matrix

    def drop(self, session_id: str) -> None:
        """Forget it and delete its file."""
        found = self._waiting.pop(session_id, None)
        if found is not None:
            discard(found[0].path)


pending_matrices = PendingMatrices()
