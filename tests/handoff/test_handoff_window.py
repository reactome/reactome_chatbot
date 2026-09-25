"""The claim a chat tab posts, and what the server accepts as one.

Chainlit forwards every message posted in the page -- including this
server's own acknowledgement, which comes straight back through the same
listener -- so the parser's job is mostly to say no.

The transport itself was verified in a headless browser against the real
Chainlit build: one claim per tab despite retries; two tabs opened together
in one browser each get only their own ID; a reload re-claims; and a single
post in the first second after load is lost 15 times out of 15, which is why
`public/custom.js` retries until acknowledged.
"""

import secrets

import pytest

from handoff import window

VALID = secrets.token_urlsafe(24)


def test_a_well_formed_claim_yields_its_id() -> None:
    assert window.claimed_id({"type": window.CLAIM_TYPE, "id": VALID}) == VALID


def test_our_own_acknowledgement_is_not_a_claim() -> None:
    # It comes back through Chainlit's listener. Treating it as a claim would
    # re-seed the thread on every ack.
    assert window.claimed_id(window.acknowledgement(VALID)) is None


@pytest.mark.parametrize(
    "message",
    [
        None,
        "reactome-handoff",
        ["reactome-handoff", VALID],
        {"id": VALID},
        {"type": "something-else", "id": VALID},
        {"type": window.CLAIM_TYPE},
        {"type": window.CLAIM_TYPE, "id": 12345},
        {"type": window.CLAIM_TYPE, "id": "short"},
        {"type": window.CLAIM_TYPE, "id": "x" * 500},
        {"type": window.CLAIM_TYPE, "id": VALID + "/../../etc"},
        {"type": window.CLAIM_TYPE, "id": VALID + "<script>"},
    ],
)
def test_anything_else_is_ignored(message: object) -> None:
    # The listener has no origin check, so any page able to post to this
    # window can send anything. None of it may be taken for a claim.
    assert window.claimed_id(message) is None


def test_the_id_floor_is_128_bits() -> None:
    # 22 base64url characters is 132 bits: an ID cannot be guessed.
    assert window.claimed_id({"type": window.CLAIM_TYPE, "id": "a" * 21}) is None
    assert window.claimed_id({"type": window.CLAIM_TYPE, "id": "a" * 22}) == "a" * 22


def test_the_acknowledgement_names_the_id_it_answers() -> None:
    # Two tabs may be retrying at once; each stops only on its own ack.
    ack = window.acknowledgement(VALID)
    assert ack == {"type": window.ACK_TYPE, "id": VALID}
