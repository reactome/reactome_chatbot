"""The query embedding model must match the one that built the bundle.

Queries are embedded and compared against vectors already in Chroma. If the two
models differ the comparison is meaningless, and if the model does not exist at
the configured endpoint the first query fails outright.

The default used to be the literal "bge-m3". That is correct for the Plant
Reactome deployment, which serves it from a self-hosted OpenAI-compatible
endpoint via OPENAI_BASE_URL, and wrong for every bundle published for Reactome,
where it 404s against api.openai.com. Hardcoding either breaks the other.
"""

from pathlib import Path

import pytest

import util.embedding_environment as ee
from agent.graph import DEFAULT_EMBEDDING_MODEL, resolve_embedding_model

REACTOME = "openai/text-embedding-3-large/reactome/Release95"
PLANT = "openai/bge-m3/plantreactome/Release68"


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ee, "EM_ARCHIVE", tmp_path)
    monkeypatch.setattr(ee, "EM_CURRENT", tmp_path / "current")
    monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
    return tmp_path


def test_reactome_bundle_resolves_to_its_own_model(archive: Path) -> None:
    (archive / "current").write_text(REACTOME)
    assert resolve_embedding_model() == "text-embedding-3-large"


def test_plantreactome_bundle_resolves_to_bge_m3(archive: Path) -> None:
    """Plant Reactome legitimately uses bge-m3; it must not be overridden."""
    (archive / "current").write_text(PLANT)
    assert resolve_embedding_model() == "bge-m3"


def test_never_silently_defaults_to_a_model_no_bundle_used(archive: Path) -> None:
    """The regression: "bge-m3" was the default regardless of what was installed."""
    (archive / "current").write_text(REACTOME)
    assert resolve_embedding_model() != "bge-m3"


def test_env_override_is_respected(
    archive: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (archive / "current").write_text(REACTOME)
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    assert resolve_embedding_model() == "text-embedding-3-small"


def test_override_that_disagrees_with_the_bundle_is_reported(
    archive: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    (archive / "current").write_text(REACTOME)
    monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
    with caplog.at_level("ERROR"):
        resolve_embedding_model()
    assert "text-embedding-3-large" in caplog.text


def test_bundles_built_with_different_models_are_reported(
    archive: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """One embedding is shared by every profile, so the bundles must agree."""
    (archive / "current").write_text(f"{REACTOME}:{PLANT}")
    with caplog.at_level("ERROR"):
        resolve_embedding_model()
    assert "different embedding models" in caplog.text


def test_falls_back_when_nothing_is_installed(archive: Path) -> None:
    assert resolve_embedding_model() == DEFAULT_EMBEDDING_MODEL
