from types import SimpleNamespace

from langchain_core.messages import AIMessage, AIMessageChunk

from util.answer_text import AnswerTextParser, answer_text


def test_plain_string_passes_through_unchanged() -> None:
    assert answer_text("plain token") == "plain token"


def test_a_chunk_is_read_through_to_its_content() -> None:
    assert answer_text(SimpleNamespace(content="hello")) == "hello"


def test_a_bare_reasoning_block_yields_nothing() -> None:
    assert answer_text({"type": "reasoning", "encrypted_content": "gAAAAAB"}) == ""


def test_only_text_blocks_are_kept() -> None:
    assert (
        answer_text(
            [
                {"type": "reasoning", "encrypted_content": "gAAAAAB"},
                {"type": "output_text", "text": "Arabidopsis "},
                {"type": "text", "text": "closes stomata."},
            ]
        )
        == "Arabidopsis closes stomata."
    )


def test_the_parser_reduces_a_message_to_text() -> None:
    """Goes through `invoke`, which is what LLMChain actually calls."""
    msg = AIMessage(content=[{"type": "output_text", "text": "stomata"}])
    assert AnswerTextParser().invoke(msg) == "stomata"


def test_the_parser_drops_reasoning_and_keeps_the_prose() -> None:
    """The production shape: a reasoning item ahead of the answer text.

    `invoke` does not hand the message to `parse`. It wraps it in a
    `ChatGeneration`, whose validator collapses the content to `message.text`,
    and `BaseMessage.text` keeps only `type == "text"` -- so an `output_text`
    answer arrives at `parse` as an empty string. A parser that overrides only
    `parse` cannot see the blocks and blanks the answer instead of fixing it.
    """
    msg = AIMessage(
        content=[
            {"type": "reasoning", "encrypted_content": "gAAAAAB", "id": "rs_1"},
            {"type": "output_text", "text": "CDK5 phosphorylates "},
            {"type": "output_text", "text": "Tau."},
        ]
    )
    assert AnswerTextParser().invoke(msg) == "CDK5 phosphorylates Tau."


def test_the_parser_streams_one_item_per_chunk() -> None:
    """`StrOutputParser` streamed; so must this.

    Extending plain `BaseOutputParser` would not fail -- `Runnable.transform`
    falls back to buffering the input and invoking once -- it would just yield
    the whole answer as a single item, which quietly takes token-at-a-time
    output away from the answer node.
    """
    chunks = [AIMessageChunk(content="alpha"), AIMessageChunk(content="beta")]
    assert list(AnswerTextParser().transform(iter(chunks))) == ["alpha", "beta"]
