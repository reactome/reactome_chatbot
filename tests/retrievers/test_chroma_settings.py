"""Each Chroma store needs its own Settings object.

`langchain_chroma` mutates whatever settings it is handed:

    client_settings.persist_directory = (
        persist_directory or client_settings.persist_directory
    )

so a shared instance means the last store constructed rewrites the persist
directory for every earlier one, and chromadb returns a client keyed on those
settings.

With only the reactome bundle installed this was invisible -- every store
pointed into the same tree. Installing the user guide bundle gave them
different trees, and a *reactome* question went looking for its collection
inside the *user guide* directory:

    PermissionError: [Errno 13] Permission denied:
    '/app/embeddings/.../userguide/Release95/sections/9c574827-...'

Found on beta, in the deployed container, by asking "What does CDK5
phosphorylate in Alzheimer disease?" after a user guide question.
"""

from retrievers.csv_chroma import chroma_settings


def test_each_call_returns_a_distinct_object() -> None:
    first, second = chroma_settings(), chroma_settings()
    assert first is not second, (
        "a shared Settings object is mutated by langchain_chroma, so two stores "
        "would end up pointing at one directory"
    )


def test_mutating_one_does_not_reach_another() -> None:
    """This is the exact mutation langchain_chroma performs."""
    first = chroma_settings()
    first.persist_directory = "/bundles/userguide/sections"

    second = chroma_settings()
    assert second.persist_directory != "/bundles/userguide/sections"


def test_telemetry_stays_off() -> None:
    # The reason this object existed in the first place; do not lose it while
    # fixing the sharing.
    assert chroma_settings().anonymized_telemetry is False
