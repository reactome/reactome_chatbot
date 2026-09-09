"""Characterization of embeddings path resolution.

`embeddings/current` is a single colon-separated line mapping each database to the
bundle in use. `bin/embeddings_manager use` writes it; the agent profiles read it
when they construct a RAG chain. They used to read it at import time, via a default
argument, which is why `require_dir` exists.
"""

from pathlib import Path

import pytest

import util.embedding_environment as ee
from util.embedding_environment import EmbeddingEnvironment

REACTOME = "openai/text-embedding-3-large/reactome/Release90"
UNIPROT = "openai/text-embedding-3-large/uniprot/Release90"


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(ee, "EM_ARCHIVE", tmp_path)
    monkeypatch.setattr(ee, "EM_CURRENT", tmp_path / "current")
    return tmp_path


def test_no_current_file_means_no_embeddings(archive: Path) -> None:
    assert EmbeddingEnvironment.get_dict() == {}
    assert EmbeddingEnvironment.get_dir("reactome") is None


def test_database_key_is_the_parent_directory_name(archive: Path) -> None:
    """The db key comes from the path, not the file -- `.../reactome/Release90` -> `reactome`."""
    (archive / "current").write_text(REACTOME)
    assert EmbeddingEnvironment.get_dict() == {"reactome": Path(REACTOME)}
    assert EmbeddingEnvironment.get_dir("reactome") == archive / REACTOME


def test_multiple_databases_are_colon_separated(archive: Path) -> None:
    (archive / "current").write_text(f"{REACTOME}:{UNIPROT}")
    assert set(EmbeddingEnvironment.get_dict()) == {"reactome", "uniprot"}


def test_get_model_strips_the_database_and_version(archive: Path) -> None:
    (archive / "current").write_text(REACTOME)
    assert EmbeddingEnvironment.get_model("reactome") == "openai/text-embedding-3-large"


def test_set_one_adds_without_disturbing_other_databases(archive: Path) -> None:
    (archive / "current").write_text(REACTOME)
    EmbeddingEnvironment.set_one(Path(UNIPROT))
    assert set(EmbeddingEnvironment.get_dict()) == {"reactome", "uniprot"}


def test_set_one_replaces_the_bundle_for_the_same_database(archive: Path) -> None:
    (archive / "current").write_text(REACTOME)
    newer = "openai/text-embedding-3-large/reactome/Release91"
    EmbeddingEnvironment.set_one(Path(newer))
    assert EmbeddingEnvironment.get_dict() == {"reactome": Path(newer)}


def test_empty_current_file_is_not_an_error(archive: Path) -> None:
    (archive / "current").write_text("")
    assert EmbeddingEnvironment.get_dict() == {}


def test_require_dir_names_what_is_installed_when_the_key_is_unknown(
    archive: Path,
) -> None:
    """The old path returned None here and passed it into a parameter typed Path."""
    (archive / "current").write_text(REACTOME)
    (archive / REACTOME).mkdir(parents=True)

    with pytest.raises(FileNotFoundError) as exc:
        EmbeddingEnvironment.require_dir("uniprot")

    message = str(exc.value)
    assert "uniprot" in message
    assert (
        "reactome" in message
    ), "say which bundles ARE installed, not just which is not"
    assert "embeddings_manager install" in message, "say what to do about it"


def test_require_dir_rejects_a_bundle_that_current_names_but_disk_lacks(
    archive: Path,
) -> None:
    """The failure that would otherwise be silent, and the reason for the is_dir check.

    `get_dir` builds a path without checking it exists, and Chroma CREATES a
    missing persist_directory rather than complaining. So a `current` left
    pointing at a deleted bundle produced a chatbot that answered every question
    from an empty collection -- confidently, and with no error anywhere.
    """
    (archive / "current").write_text(REACTOME)  # note: directory never created

    assert EmbeddingEnvironment.get_dir("reactome") is not None, "the old, quiet path"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        EmbeddingEnvironment.require_dir("reactome")


def test_require_dir_returns_the_directory_when_it_is_there(archive: Path) -> None:
    (archive / "current").write_text(REACTOME)
    (archive / REACTOME).mkdir(parents=True)
    assert EmbeddingEnvironment.require_dir("reactome") == archive / REACTOME
