"""Citations come from whichever identifier a collection actually carries.

Reactome documents have `st_id`. Userguide documents never will -- they are
documentation pages whose identity is a URL -- and until 2026-09-18 that meant a
userguide-routed question returned no citations at all. Measured then: "How do I
use the pathway browser?" gave 0 citations and 452 tokens.
"""

from langchain_core.documents import Document

from agent.graph import _citation_for


def _doc(**metadata: str) -> Document:
    return Document(page_content="text", metadata=dict(metadata))


def test_a_reactome_document_is_cited_by_stable_id() -> None:
    citation = _citation_for(_doc(st_id="R-HSA-8863013", display_name="CDK5 binds p25"))
    assert citation is not None
    assert (citation.st_id, citation.display_name) == (
        "R-HSA-8863013",
        "CDK5 binds p25",
    )
    assert citation.url == "", "a Reactome citation must not also carry a url"


def test_a_userguide_document_is_cited_by_url() -> None:
    citation = _citation_for(
        _doc(
            source="https://reactome.org/userguide/pathway-browser",
            page_title="Pathway Browser",
            section_title="Introduction",
        )
    )
    assert citation is not None
    assert citation.url == "https://reactome.org/userguide/pathway-browser"
    assert citation.display_name == "Pathway Browser"
    assert citation.st_id == "", "a userguide citation must not carry a stable id"


def test_a_stable_id_wins_when_a_document_somehow_has_both() -> None:
    """No document should, but the caller's contract is one identifier."""
    citation = _citation_for(
        _doc(st_id="R-HSA-1", display_name="Real", source="https://example.invalid")
    )
    assert citation is not None
    assert citation.st_id == "R-HSA-1"
    assert citation.url == ""


def test_a_document_with_neither_is_not_cited() -> None:
    """Silence beats inventing an identifier that resolves to nothing."""
    assert _citation_for(_doc(page_title="orphan")) is None


def test_no_fabricated_stable_id_is_ever_produced() -> None:
    """The rule this design exists to keep.

    A made-up `R-` id would resolve to nothing, or worse to a real but wrong
    entity, and a reader cannot tell the difference from the link text.
    """
    citation = _citation_for(
        _doc(source="https://reactome.org/userguide", page_title="Userguide")
    )
    assert citation is not None
    assert not citation.st_id.startswith("R-")
    assert citation.st_id == ""


def test_a_local_file_path_is_never_cited_as_a_url() -> None:
    """`source` is a generic LangChain field, not necessarily a web address.

    The CSV loaders set it to the file they read. Without a scheme check, a
    Reactome document that happened to lack an `st_id` cited
    "/home/awright/git/reactome_chatbot/embeddings/openai/..." -- a server path,
    sent to a public website. Found by running a real question; the fixtures
    above are too clean to have caught it.
    """
    for path in (
        "/home/awright/git/reactome_chatbot/embeddings/openai/text-embedding-3-large",
        "embeddings/reactome/Release97/reactions.csv",
        "file:///etc/passwd",
        "",
    ):
        assert _citation_for(_doc(source=path, page_title="whatever")) is None, path


def test_an_http_url_is_still_cited() -> None:
    """The guard must not throw out what it exists to let through."""
    for url in ("https://reactome.org/userguide", "http://reactome.org/userguide"):
        citation = _citation_for(_doc(source=url, page_title="Userguide"))
        assert citation is not None
        assert citation.url == url
