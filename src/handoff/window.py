"""The message a chat tab sends to claim its handoff, and the reply.

Spec 013. The website opens the chat in a new tab at
`/chat/guest/#handoff=<id>`. The ID is in the fragment because browsers never
send a fragment to a server: it cannot reach nginx logs or a `Referer`.

Chainlit 2.11 does not give the app the page URL on connect -- its connect
handler reads cookies only -- and a cookie is shared by every tab, so two
handoffs opened in quick succession could seed the wrong one. Instead, a
script in the tab (`public/custom.js`) reads the fragment and posts the ID
with `window.postMessage`. Chainlit's page forwards `event.data` of *every*
message posted in the window to `@cl.on_window_message`, over that tab's own
socket. So the ID is bound to the tab that carried it.

Two properties of that route shape this module:

- **It forwards anything.** There is no origin or source check in Chainlit's
  listener, and our own acknowledgement comes back through the same listener
  (the server's reply is posted to `window.parent`, which for a top-level tab
  is the tab itself). So every message is parsed strictly and anything that
  is not a well-formed claim is ignored -- including our own ack.
- **It can drop messages.** If the socket is not connected yet, the page
  discards the post. So the script retries until acknowledged, which means
  the same claim can arrive more than once, and redeeming it must be
  idempotent within a session.
"""

import re
from typing import Any

CLAIM_TYPE = "reactome-handoff"
ACK_TYPE = "reactome-handoff-ack"

#: At least 128 bits of URL-safe randomness (22 base64url characters), and a
#: ceiling so a hostile page cannot post something enormous.
_ID = re.compile(r"^[A-Za-z0-9_-]{22,128}$")


def claimed_id(message: Any) -> str | None:
    """The handoff ID in a window message, or None if it is not a claim.

    Returns None for everything that is not exactly a claim -- including the
    acknowledgement this server sends, which Chainlit forwards straight back.
    """
    if not isinstance(message, dict):
        return None
    if message.get("type") != CLAIM_TYPE:
        return None
    value = message.get("id")
    if not isinstance(value, str) or not _ID.match(value):
        return None
    return value


def acknowledgement(handoff_id: str) -> dict[str, str]:
    """What tells the tab's script to stop retrying."""
    return {"type": ACK_TYPE, "id": handoff_id}
