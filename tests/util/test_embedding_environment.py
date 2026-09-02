"""Characterization of embeddings path resolution.

`embeddings/current` is a single colon-separated line mapping each database to the
bundle in use. `bin/embeddings_manager use` writes it; the retriever modules read it
at import time.
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
