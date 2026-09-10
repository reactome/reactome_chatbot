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

from agent.graph import (  # noqa: E402
    FIXED_TEMPERATURE,
    FIXED_TEMPERATURE_MODELS,
    resolve_temperature,
)
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


# Every verdict below was measured against the API by ./bin/probe_model_temperature
# on 2026-09-09, not inferred from the name. See test_the_set_is_not_a_name_pattern.
REFUSES_ZERO = [
    "chat-latest",
    "gpt-5",
    "gpt-5-2025-08-07",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-5.5",
    "gpt-5.5-2026-04-23",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "o3",
    "o3-2025-04-16",
    "o4-mini",
    "o4-mini-2025-04-16",
]
ACCEPTS_ZERO = [
    "gpt-3.5-turbo",
    "gpt-4",
    "gpt-4o-mini",
    "gpt-4.1",
    "gpt-4.1-nano",
    "gpt-5.1",
    "gpt-5.2",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano-2026-03-17",
]


@pytest.mark.parametrize("model", REFUSES_ZERO)
def test_models_that_refuse_zero_get_their_only_supported_value(model: str) -> None:
    assert resolve_temperature(model) == FIXED_TEMPERATURE == 1.0


@pytest.mark.parametrize("model", ACCEPTS_ZERO)
def test_every_other_model_still_gets_zero(model: str) -> None:
    """Determinism stays the default; only the models that refuse it lose it."""
    assert resolve_temperature(model) == 0.0


def test_the_set_is_not_a_name_pattern() -> None:
    """The reason this is an exact-match set and not a prefix match.

    The behaviour interleaves inside one family: gpt-5 refuses 0.0, gpt-5.1,
    gpt-5.2 and gpt-5.4 accept it, gpt-5.5 and gpt-5.6 refuse it again. A "gpt-5"
    prefix -- which this file used to have -- also matches gpt-5.1, and was wrong
    for eleven models.
    """
    assert resolve_temperature("gpt-5") == FIXED_TEMPERATURE
    for accepted in ("gpt-5.1", "gpt-5.2", "gpt-5.4"):
        assert (
            resolve_temperature(accepted) == 0.0
        ), f"{accepted} accepts 0.0; a gpt-5 prefix would have caught it"
    assert resolve_temperature("gpt-5.5") == FIXED_TEMPERATURE


def test_dated_snapshots_resolve_as_the_model_they_pin() -> None:
    """`<model>-YYYY-MM-DD` behaves as `<model>`, so the suffix is stripped.

    This is what keeps the set from needing an edit every time OpenAI pins one.
    """
    assert resolve_temperature("gpt-5.6-luna-2026-06-01") == FIXED_TEMPERATURE
    assert resolve_temperature("gpt-5.4-mini-2026-03-17") == 0.0


def test_every_listed_model_is_bare_of_a_snapshot_suffix() -> None:
    """A dated id in the set would be dead: the suffix is stripped before lookup."""
    for model in FIXED_TEMPERATURE_MODELS:
        assert (
            resolve_temperature(model) == FIXED_TEMPERATURE
        ), f"{model} is in the set but does not resolve through it"


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


# --- configured temperature, from config.yml (spec 003, US2) -------------------


def test_a_configured_temperature_a_model_refuses_stops_startup() -> None:
    """FR-006, and the reason spec 003 is a specification and not a bump.

    Without this the mistake surfaces as a 400 on a user's first question --
    visible to a user, attributed to the chatbot, diagnosable only from logs.
    """
    with pytest.raises(SystemExit) as exc:
        resolve_temperature("gpt-5.6-luna", configured=0.0)

    message = str(exc.value)
    assert "gpt-5.6-luna" in message, "name the model"
    assert "0.0" in message, "name the value"
    assert "LLM_TEMPERATURE" in message, "name the fix"


def test_a_configured_temperature_a_model_accepts_is_used() -> None:
    assert resolve_temperature("gpt-4o-mini", configured=0.7) == 0.7


def test_the_only_supported_value_is_accepted_when_configured() -> None:
    """Setting 1.0 explicitly for luna is redundant but not wrong."""
    assert resolve_temperature("gpt-5.6-luna", configured=1.0) == FIXED_TEMPERATURE


def test_an_unknown_model_is_not_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-007. The table is empirical and always behind; an unknown model must
    not block startup. It fails on the first request with OpenAI's own 404,
    which is unambiguous."""
    assert resolve_temperature("gpt-7-unreleased", configured=0.0) == 0.0


def test_the_environment_still_wins_over_a_configured_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM_TEMPERATURE is the escape hatch, so it must outrank config.yml --
    including the guard above, which is what makes it an escape hatch."""
    monkeypatch.setenv("LLM_TEMPERATURE", "1")
    assert resolve_temperature("gpt-5.6-luna", configured=0.0) == 1.0
