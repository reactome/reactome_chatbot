# Tests

These are **characterization tests**: they pin down what the code does *today*, so that
the dependency upgrade (LangChain 0.3 -> 1.x, Chroma <1.0 -> 1.x) has a tripwire. Where
current behaviour looks wrong, the test asserts the current behaviour and carries a
`BUG:` comment rather than asserting the desired behaviour — change the test and the code
together, deliberately.

## Running

    poetry run pytest                        # everything importable
    poetry run pytest -m "not requires_retrieval_stack"

## Markers

- `requires_retrieval_stack` — needs langchain/chromadb/torch importable.
- `requires_embeddings` — additionally needs an installed bundle (`./bin/embeddings_manager ls`).

## Why so little is covered

Most of `src/` cannot be imported without a provisioned environment: `load_dotenv()` and
`AgentGraph(...)` run at import time in the entry points, and
`EmbeddingEnvironment.get_dir(...)` is a *default argument* in the retriever modules, so
importing them requires an embeddings bundle on disk. Even `util.config_yml`, which is
pure configuration logic, transitively imports torch because it pulls one enum from
`agent.profiles`.

Breaking that import-time coupling is what unlocks real coverage here.
