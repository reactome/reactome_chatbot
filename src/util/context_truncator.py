from langchain_core.documents import Document
import tiktoken


def truncate_to_token_limit(
    docs: list[Document],
    max_docs: int = 15,
    max_tokens: int = 12000,
    model: str = "gpt-4o",
) -> list[Document]:
    """
    Truncate document list to fit within token and count budgets.
    Docs must already be ranked from best to worst (e.g. WRR).
    Cuts from the bottom so least relevant docs are removed first.
    """
    encoder = tiktoken.encoding_for_model(model)
    result = []
    total_tokens = 0

    for doc in docs[:max_docs]:
        doc_tokens = len(encoder.encode(doc.page_content))

        if total_tokens + doc_tokens > max_tokens:
            # always include at least one doc even if it exceeds budget
            if not result:
                result.append(doc)
            break

        result.append(doc)
        total_tokens += doc_tokens

    return result