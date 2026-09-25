"""Offers the chat has made and not yet had answered.

Round three of review found the first version of this state crashed on
resume: it lived in Chainlit's user_session, which is saved as JSON, and the
button objects came back as null.
"""

import json
from dataclasses import asdict

from analysis.proposals import Proposal, ProposalStore


def offer(text: str = "run ORA on TP53, MDM2") -> Proposal:
    return Proposal(
        text=text,
        message_id="m1",
        identifiers=("TP53", "MDM2"),
        actions=(("gene_list_run", "a1"), ("gene_list_no", "a2")),
    )


def test_an_offer_is_plain_data() -> None:
    # Whatever holds it, it survives JSON unchanged -- the failure was
    # cl.Action objects turning into null.
    data = asdict(offer())
    assert json.loads(json.dumps(data)) == {
        **data,
        "identifiers": ["TP53", "MDM2"],
        "actions": [["gene_list_run", "a1"], ["gene_list_no", "a2"]],
    }


def test_an_offer_is_claimed_once() -> None:
    store = ProposalStore()
    store.put("s", "p", offer())
    assert store.take("s", "p") == offer()
    assert store.take("s", "p") is None  # double click, or click then "yes"


def test_offers_are_per_session() -> None:
    store = ProposalStore()
    store.put("s1", "p", offer())
    assert store.take("s2", "p") is None
    assert store.take("s1", "p") is not None


def test_yes_means_only_the_offer_just_made() -> None:
    store = ProposalStore()
    store.put("s", "p1", offer())
    store.put("s", "p2", offer())
    assert store.take_latest("s") == "p2"
    # Read once: the next message is no longer "straight after" the offer.
    assert store.take_latest("s") is None


def test_clicking_the_latest_offer_ends_its_yes() -> None:
    store = ProposalStore()
    store.put("s", "p", offer())
    store.take("s", "p")
    assert store.take_latest("s") is None


def test_older_offers_are_evicted_and_returned_for_their_buttons() -> None:
    store = ProposalStore(max_per_session=2)
    first = offer("first")
    store.put("s", "p1", first)
    store.put("s", "p2", offer("second"))
    assert store.put("s", "p3", offer("third")) == [first]
    assert store.take("s", "p1") is None
    assert store.take("s", "p3") is not None


def test_sessions_are_bounded() -> None:
    store = ProposalStore(max_sessions=2)
    for name in ("a", "b", "c"):
        store.put(name, "p", offer())
    assert store.take("a", "p") is None
    assert store.take("c", "p") is not None


def test_nothing_to_take_is_none_not_an_error() -> None:
    store = ProposalStore()
    assert store.take("unknown", "p") is None
    assert store.take("unknown", None) is None
    assert store.take_latest("unknown") is None
