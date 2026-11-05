"""Reduce a chat model's output to the text the reader should see.

Chat Completions returns `content` as a plain string. The Responses API returns
a list of blocks -- `{'type': 'output_text', 'text': ...}` interleaved with
`{'type': 'reasoning', 'encrypted_content': ...}`. Anything that treats `content`
as a string then renders a Python repr of base64 reasoning into the answer, and
that repr is what reaches the chat, the checkpoint and the `steps` row.

Lives here rather than in agent.graph because both the streaming boundary and the
RAG chain need it, and only one of them can filter by node.
"""

from typing import Any, override

from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.outputs import ChatGeneration, Generation

#: Block types that carry answer prose. `output_text` is what the raw Responses
#: API emits; `text` is what langchain normalises it to. Both are needed: an
#: endpoint that passes Responses-API items through unconverted uses the former,
#: and `BaseMessage.text` -- which knows only about `text` -- returns "" for it.
_TEXT_TYPES = ("text", "output_text")


def answer_text(value: Any) -> str:
    """The answer text in `value`, which may be a chunk, a message, or content.

    Accepts anything with a `content` attribute as well as raw content, so the
    streaming loop and the output parser share one implementation. A bare dict --
    which is what a chunk's content actually is here -- is handled; a list-only
    implementation would drop real answer text.
    """
    content = getattr(value, "content", value)
    if isinstance(content, str):
        return content
    blocks = content if isinstance(content, list) else [content]
    parts: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in _TEXT_TYPES:
            parts.append(block.get("text", ""))
    return "".join(parts)


class AnswerTextParser(BaseTransformOutputParser[str]):
    """`StrOutputParser`, except it survives a Responses-API response.

    Overriding `parse` alone is useless, and the reason is worth recording:

    `invoke()` and `transform()` do not call `parse` with the message. Both call
    `parse_result([ChatGeneration(message=...)])`, and `ChatGeneration` flattens
    the content in a model validator -- `text = message.text` -- before anything
    of ours sees it. `BaseMessage.text` keeps only blocks with `type == "text"`,
    so a `parse` override is handed an already-collapsed string and can never
    inspect a block. For `output_text` blocks that collapsed string is empty:
    the answer is silently blank rather than wrong, which is the worse failure.

    So the work happens in `parse_result`, which is the one choke point every
    path -- `invoke`, `ainvoke`, `transform`, `atransform` -- goes through, and
    which still has the message.

    Extends the *transform* base rather than `BaseOutputParser` because every
    call site this replaced used `StrOutputParser`, which does. Plain
    `BaseOutputParser` still streams -- `Runnable.transform` falls back to
    buffering the input and calling `ainvoke` -- but it streams as one chunk,
    which would quietly take token-at-a-time output away from the answer node.
    """

    @override
    def parse(self, text: str) -> str:
        return answer_text(text)

    @override
    def parse_result(self, result: list[Generation], *, partial: bool = False) -> str:
        first = result[0]
        # `first.text` is already flattened by this point, so a chat generation
        # has to be read through its message or the blocks are unrecoverable.
        if isinstance(first, ChatGeneration):
            return answer_text(first.message.content)
        return answer_text(first.text)

    @property
    def _type(self) -> str:
        return "answer_text"
