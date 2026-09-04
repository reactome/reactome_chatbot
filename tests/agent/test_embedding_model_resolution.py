"""The query embedding model must match the one that built the bundle.

Queries are embedded and compared against vectors already in Chroma. If the two
models differ the comparison is meaningless -- and if the model does not exist at
the provider, the first query 404s.

The default used to be a literal "bge-m3", which OpenAI has no such model for, so
any deployment that did not set EMBEDDING_MODEL was broken. That reached main in
the plantreactome merge.
"""

from pathlib import Path

import pytest

import util.embedding_environment as ee
from agent.graph import DEFAULT_EMBEDDING_MODEL, resolve_embedding_model

BUNDLE = "openai/text-embedding-3-large/reactome/Release95"


@pytest.fixture
def _installed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ee, "EM_ARCHIVE", tmp_path)
    monkeypatch.setattr(ee, "EM_CURRENT", tmp_path / "current")
    (tmp_path / "current").write_text(BUNDLE)


@pytest.mark.usefixtures("_installed")
def test_defaults_to_the_model_that_built_the_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert resolve_embedding_model() == "text-embedding-3-large"


@pytest.mark.usefixtures("_installed")
def test_never_defaults_to_a_model_the_provider_does_not_have(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The specific regression: "bge-m3" is not an OpenAI model."""
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert resolve_embedding_model() != "bge-m3"


@pytest.mark.usefixtures("_installed")
def test_env_override_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    assert resolve_embedding_model() == "text-embedding-3-small"


@pytest.mark.usefixtures("_installed")
def test_mismatch_is_reported(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A silent mismatch produces meaningless retrieval, so it must be loud."""
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    with caplog.at_level("ERROR"):
        resolve_embedding_model()
    assert "text-embedding-3-large" in caplog.text
    assert "meaningless" in caplog.text


def test_falls_back_when_no_bundle_is_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(ee, "EM_ARCHIVE", tmp_path)
    monkeypatch.setattr(ee, "EM_CURRENT", tmp_path / "current")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    assert resolve_embedding_model() == DEFAULT_EMBEDDING_MODEL
