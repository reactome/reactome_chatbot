import importlib.util

import pytest

RETRIEVAL_STACK_MODULES = ("langchain", "langchain_chroma", "chromadb", "nltk")


def _stack_available() -> bool:
    return all(importlib.util.find_spec(m) is not None for m in RETRIEVAL_STACK_MODULES)


def pytest_runtest_setup(item: pytest.Item) -> None:
    if (
        list(item.iter_markers(name="requires_retrieval_stack"))
        and not _stack_available()
    ):
        pytest.skip("retrieval stack (langchain/chromadb/nltk) not installed")
