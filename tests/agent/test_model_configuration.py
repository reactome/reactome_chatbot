"""Which model answers, and where that choice comes from.

Precedence is LLM_MODEL, then config.yml, then the built-in default. That is the
reverse of util/secrets.py, where a mounted Docker secret beats the environment --
the same rule from opposite sides: the more specific source wins. A secret is
mounted BY a deployment and should beat a committed file; LLM_MODEL is how one
container overrides a committed config.yml.
"""

import pytest

pytest.importorskip("langchain_openai", reason="LLM stack not installed")

from pydantic import ValidationError  # noqa: E402

from agent.graph import DEFAULT_LLM_MODEL, resolve_llm_model  # noqa: E402
from util.config_yml.models import LLMConfig  # noqa: E402


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's LLM_MODEL must not decide what these tests assert."""
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)


def test_no_configuration_behaves_exactly_as_before() -> None:
    """FR-002, and the scenario every existing deployment is in.

    The most important test here: adding this feature must be invisible to a
    config.yml that does not use it.
    """
    assert resolve_llm_model(None) == ("openai", DEFAULT_LLM_MODEL, None)


def test_an_empty_llm_section_also_changes_nothing() -> None:
    """`llm: {}` is a section that sets nothing, not a request for something."""
    assert resolve_llm_model(LLMConfig()) == ("openai", DEFAULT_LLM_MODEL, None)


def test_a_configured_model_is_the_one_selected() -> None:
    provider, model, base_url = resolve_llm_model(LLMConfig(model="gpt-5.6-luna"))
    assert (provider, model, base_url) == ("openai", "gpt-5.6-luna", None)


def test_the_environment_overrides_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """FR-003. How one container is pointed elsewhere without editing a file."""
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    _, model, _ = resolve_llm_model(LLMConfig(model="gpt-5.6-luna"))
    assert model == "gpt-4o-mini"


def test_the_environment_overrides_the_base_url_too(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://from-the-environment/v1")
    _, _, base_url = resolve_llm_model(LLMConfig(base_url="https://from-the-file/v1"))
    assert base_url == "https://from-the-environment/v1"


def test_a_self_hosted_endpoint_can_be_configured() -> None:
    """Plant Reactome serves a model from its own OpenAI-compatible endpoint.

    This is why the config is named fields rather than a "provider/model"
    string, as #151 proposed -- base_url has nowhere to live in a flat string.
    """
    provider, model, base_url = resolve_llm_model(
        LLMConfig(provider="ollama", model="bge-m3", base_url="http://localhost:11434")
    )
    assert (provider, model, base_url) == (
        "ollama",
        "bge-m3",
        "http://localhost:11434",
    )


def test_the_config_model_cannot_name_an_embedding_model() -> None:
    """FR-004, the one part of #112 and #151 deliberately not taken.

    The embedding model is derived from the bundle that built the vectors. A
    query embedded with a different model returns nonsense rather than an error,
    so there is no safe way to configure it.
    """
    assert "embedding" not in LLMConfig.model_fields

    # And naming one is fatal rather than ignored. Pydantic's default is to drop
    # unknown keys silently, which would leave an operator believing they had set
    # something; Config.from_yaml turns this ValidationError into a refusal to start.
    with pytest.raises(ValidationError, match="embedding_model"):
        LLMConfig(embedding_model="text-embedding-3-large")  # type: ignore[call-arg]


def test_no_configuration_file_can_name_an_embedding_model() -> None:
    """SC-004, checked against the files an operator actually edits.

    The schema and the shipped default are what a person copies from. If either
    showed an embedding model, someone would set it.
    """
    from pathlib import Path

    repo = Path(__file__).parent.parent.parent
    for name in (".config.schema.yaml", "config_default.yml"):
        text = repo.joinpath(name).read_text()
        offending = [
            line
            for line in text.splitlines()
            if "embedding" in line.lower() and not line.lstrip().startswith("#")
        ]
        assert not offending, f"{name} offers an embedding setting: {offending}"
