"""Which temperature gets sent, and to which models.

The gpt-5.5/5.6 families and gpt-6-astra accept only their own default of 1.
Every other value is a 400 on the *first request*, not at construction, so
nothing catches it until a user asks a question.

There is no way to send no temperature at all: omitting the argument makes
ChatOpenAI send its own pydantic default of 0.7, which those models reject just
as firmly as 0.0. An earlier version of this change tried exactly that, and
`model_dump(exclude_unset=True)` agreed the field was unset -- while the request
still carried 0.7. So the assertions below are about the value chosen, and the
live behaviour was verified by hand against the API.
"""

import pytest

pytest.importorskip("langchain_openai", reason="LLM stack not installed")

from langchain_openai.chat_models.base import ChatOpenAI  # noqa: E402

from agent.graph import FIXED_TEMPERATURE, resolve_temperature  # noqa: E402
from agent.models import get_llm  # noqa: E402


def _built(model: str, **kwargs: float) -> ChatOpenAI:
    """get_llm is typed BaseChatModel; the temperature lives on ChatOpenAI."""
    llm = get_llm("openai", model, **kwargs)  # type: ignore[arg-type]
    assert isinstance(llm, ChatOpenAI)
    return llm


@pytest.fixture(autouse=True)
def _no_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's LLM_TEMPERATURE must not decide what these tests assert."""
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)


@pytest.mark.parametrize(
    "model",
    ["gpt-5.5", "gpt-5.5-2026-04-23", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-6-astra"],
)
def test_models_that_refuse_zero_get_their_only_supported_value(model: str) -> None:
    assert resolve_temperature(model) == FIXED_TEMPERATURE == 1.0


@pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4.1", "gpt-5", "gpt-5.4-mini"])
def test_every_other_model_still_gets_zero(model: str) -> None:
    """Determinism stays the default; only the models that refuse it lose it.

    gpt-5 and gpt-5.4-mini are in this list deliberately: they are newer than
    gpt-4o-mini and they do accept 0.0, so the rule is not "new models".
    """
    assert resolve_temperature(model) == 0.0


def test_the_prefix_table_covers_dated_snapshots() -> None:
    """gpt-5.6-luna and a dated pin of it must resolve the same way.

    Matching on prefix is why this table needs no edit each time OpenAI pins a
    snapshot of a family already listed.
    """
    assert resolve_temperature("gpt-5.6-luna-2026-06-01") == FIXED_TEMPERATURE


def test_the_override_wins_over_the_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """The escape hatch, for a model the table has not met."""
    monkeypatch.setenv("LLM_TEMPERATURE", "1")
    assert resolve_temperature("gpt-4o-mini") == 1.0
    monkeypatch.setenv("LLM_TEMPERATURE", "0")
    assert resolve_temperature("gpt-5.6-luna") == 0.0


@pytest.mark.parametrize("value", ["", "   "])
def test_an_empty_override_is_treated_as_unset(
    value: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty env var in a compose file means "not configured", not 0.0."""
    monkeypatch.setenv("LLM_TEMPERATURE", value)
    assert resolve_temperature("gpt-4o-mini") == 0.0


def test_an_unparseable_override_stops_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Article IV: a temperature nobody can honour must not fall back silently."""
    monkeypatch.setenv("LLM_TEMPERATURE", "warm")
    with pytest.raises(SystemExit, match="not a number"):
        resolve_temperature("gpt-4o-mini")


def test_get_llm_sends_the_temperature_it_is_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")  # construction only
    assert _built("gpt-4o-mini", temperature=0.0).temperature == 0.0
    assert _built("gpt-5.6-luna", temperature=1.0).temperature == 1.0


def test_get_llm_still_defaults_to_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Callers that never heard of this change keep the old behaviour."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-not-a-real-key")
    assert _built("gpt-4o-mini").temperature == 0.0
