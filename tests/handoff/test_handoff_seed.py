"""What a claimed handoff puts into the conversation, and what it must not."""

import asyncio
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from analysis.client import Fetched
from handoff import seed
from handoff.store import Handoff, HandoffStore

RESULT: dict[str, Any] = {
    "summary": {"token": "T", "type": "OVERREPRESENTATION"},
    "identifiersNotFound": 2,
    "pathwaysFound": 3,
    "pathways": [
        {
            "stId": "R-HSA-1",
            "name": "Cell Cycle",
            "species": {"name": "Homo sapiens"},
            "entities": {"found": 5, "total": 10, "pValue": 1e-6, "fdr": 1e-4},
        }
    ],
}


def handoff(tier: str = "aggregate", **overrides: Any) -> Handoff:
    fields: dict[str, Any] = {
        "kind": "analysis",
        "token": "T",
        "release": "97",
        "tier": tier,
        "summary": "Cell Cycle is the strongest signal.",
        "citations": (),
        "created_at": time.time(),
    }
    fields.update(overrides)
    return Handoff(**fields)


class Spy:
    def __init__(self, returns: Any) -> None:
        self.calls = 0
        self.returns = returns

    async def __call__(self, _token: str) -> Any:
        self.calls += 1
        return self.returns


def data_for(h: Handoff, fetch: Spy, unmatched: Spy) -> Any:
    return asyncio.run(seed.analysis_data(h, fetch=fetch, fetch_unmatched=unmatched))


class TestDisclosure:
    def test_an_aggregate_handoff_never_fetches_the_readers_identifiers(self) -> None:
        """FR-003, the one that matters. The reader chose aggregate on the
        website; continuing in chat must not quietly go and get the
        identifiers they declined to share."""
        unmatched = Spy(["MYSTERY1", "MYSTERY2"])
        data = data_for(handoff("aggregate"), Spy(Fetched("ok", RESULT)), unmatched)

        assert unmatched.calls == 0
        assert "identifiers_not_found_names" not in (data or {})

    def test_an_identifiers_handoff_includes_them(self) -> None:
        # The control: without it the test above would pass against code that
        # never fetches identifiers at all.
        unmatched = Spy(["MYSTERY1", "MYSTERY2"])
        data = data_for(handoff("identifiers"), Spy(Fetched("ok", RESULT)), unmatched)

        assert unmatched.calls == 1
        assert data["identifiers_not_found_names"] == ["MYSTERY1", "MYSTERY2"]


class TestTheSeededTurn:
    def test_has_the_shape_of_a_real_turn(self) -> None:
        turn = seed.seeded_turn(handoff(), {"pathways": []})
        assert [type(m) for m in turn] == [HumanMessage, AIMessage]

    def test_the_model_turn_carries_the_summary_the_reader_saw_verbatim(self) -> None:
        turn = seed.seeded_turn(handoff(), {"pathways": []})
        assert str(turn[1].content).startswith("Cell Cycle is the strongest signal.")

    def test_the_model_turn_carries_the_data_it_was_built_from(self) -> None:
        data = asyncio.run(
            seed.analysis_data(
                handoff(), fetch=Spy(Fetched("ok", RESULT)), fetch_unmatched=Spy(None)
            )
        )
        text = str(seed.seeded_turn(handoff(), data)[1].content)
        assert "Cell Cycle" in text
        assert "aggregate" in text

    def test_a_result_that_is_gone_is_said_not_papered_over(self) -> None:
        # The Analysis Service deletes results on a new release.
        data = data_for(handoff(), Spy(Fetched("gone", None)), Spy(None))
        assert data is None
        text = str(seed.seeded_turn(handoff(), None)[1].content)
        assert "no longer available" in text


class TestTheReader:
    def test_sees_their_summary_verbatim(self) -> None:
        assert "Cell Cycle is the strongest signal." in seed.shown_to_reader(handoff())

    def test_an_expired_link_says_so(self) -> None:
        # FR-009: not a silent empty chat the reader assumes has the context.
        assert "expired" in seed.UNAVAILABLE


class TestTheStore:
    def test_a_handoff_can_be_claimed_repeatedly_within_its_window(self) -> None:
        # FR-007: a reload re-claims, so single-use would empty the chat.
        store = HandoffStore()
        handoff_id = store.put(handoff())
        assert store.get(handoff_id) is not None
        assert store.get(handoff_id) is not None

    def test_expires(self) -> None:
        store = HandoffStore(ttl_seconds=60)
        handoff_id = store.put(handoff(created_at=time.time() - 61))
        assert store.get(handoff_id) is None

    def test_an_unknown_id_is_none(self) -> None:
        assert HandoffStore().get("never-issued") is None

    def test_ids_fit_what_the_tab_script_will_carry(self) -> None:
        from handoff.window import claimed_id

        handoff_id = HandoffStore().put(handoff())
        assert claimed_id({"type": "reactome-handoff", "id": handoff_id}) == handoff_id

    def test_is_bounded(self) -> None:
        store = HandoffStore(max_entries=3)
        ids = [store.put(handoff()) for _ in range(5)]
        assert store.get(ids[0]) is None
        assert store.get(ids[-1]) is not None
