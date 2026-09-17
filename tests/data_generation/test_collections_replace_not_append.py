"""Every generator must replace its collection, never add to it.

`Chroma.from_documents` appends to whatever is already in the persist directory.
Running a build twice therefore stores every document twice, with no error and
nothing in the output to suggest it. That is not hypothetical: the Release95
reactome bundle shipped with 33,498 reaction documents for 16,749 CSV rows --
exactly 2.00x -- and it went unnoticed until the bundle was rebuilt for
Release 97. The vector retriever over-fetches and de-duplicates on `st_id`, so
half of that over-fetch was being spent on duplicates of what it already had.

The same fault was present in three of the five generators, each written
separately. This is a source-level guard over all of them, because four of the
five have no executing tests at all and the real fix -- one shared persist
helper -- should not be attempted until something would catch it going wrong.
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src/data_generation"

GENERATORS = [
    "reactome/__init__.py",
    "alliance/__init__.py",
    "uniprot/__init__.py",
    "userguide/__init__.py",
    "disease_variant/__init__.py",
]


def _calls_from_documents(tree: ast.AST) -> bool:
    return any(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "from_documents"
        for n in ast.walk(tree)
    )


@pytest.mark.parametrize("relative", GENERATORS)
def test_generator_removes_the_collection_before_writing_it(relative: str) -> None:
    source = (SRC / relative).read_text()
    tree = ast.parse(source)
    if not _calls_from_documents(tree):
        pytest.skip(f"{relative} does not persist a Chroma collection")

    assert "rmtree" in source, (
        f"{relative} calls Chroma.from_documents without removing the existing "
        "collection first, so a second run doubles it silently"
    )
    # Removing the directory leaves chromadb's cached system client pointing at
    # a deleted sqlite file; the next write then fails with "attempt to write a
    # readonly database", which reads as corruption rather than as this.
    assert "clear_system_cache" in source, (
        f"{relative} removes the collection directory without clearing "
        "chromadb's cached client"
    )


def test_userguide_does_not_gate_the_removal_on_force() -> None:
    """`force` means re-fetch the HTML, not "rebuild the collection".

    Gating the removal on it meant an ordinary rebuild appended.
    """
    source = (SRC / "userguide/__init__.py").read_text()
    assert "if force and chroma_dir.exists():" not in source
