# Quickstart: verifying model configuration

Operator checks for how the answering model is configured, end to end, the way a
deployment uses it. Written for the 2026-09-10 implementation and re-checked against
the code on 2026-10-21. Every step is a thing that can be observed failing, not a
unit test. Requirement IDs refer to [spec.md](./spec.md).

## Prerequisites

- an installed reactome bundle (`./bin/embeddings_manager which`)
- `OPENAI_API_KEY` in `.env`

## 1. Nothing configured behaves exactly as before (FR-002)

The most important scenario, because it is what every existing deployment does.

```bash
grep -c '^llm:' config.yml || echo "no llm section — good"
poetry run chainlit run bin/chat-chainlit.py
```

**Expect**: the server starts and the startup log reads `Answering with
openai/gpt-4o-mini at temperature 0.0` (`src/agent/graph.py:315-317`), unless
`LLM_MODEL` is set in the environment. Ask a question; it answers.

## 2. A model set in config.yml is the one that answers (FR-001, FR-008)

```yaml
# config.yml
llm:
  provider: openai
  model: gpt-5.6-luna
```

**Expect**: the startup log names `gpt-5.6-luna` at temperature `1.0`: that model
refuses `0.0`, and `resolve_temperature` derives `1.0` without being told.

## 3. The environment overrides the file (FR-003)

```bash
LLM_MODEL=gpt-4o-mini poetry run chainlit run bin/chat-chainlit.py
```

**Expect**: `gpt-4o-mini` in the log, despite `config.yml` naming luna. This is how
one container is overridden without editing a committed file.

## 4. A contradictory pair stops startup (FR-006, SC-003)

```yaml
llm:
  model: gpt-5.6-luna
  temperature: 0
```

**Expect**: the server **refuses to start**, naming the model, the value, and the
fix (`src/agent/graph.py:160-166`). Starting and failing later is the bug this guards
against (spec User Story 2). `LLM_TEMPERATURE=0` is not checked this way: it is the
escape hatch and is sent as given.

Compare with what happens without this feature: the server starts, and the first
user to ask a question gets an error that reads as the chatbot being broken.

## 5. A model the table has not met still starts (FR-007)

```yaml
llm:
  model: gpt-7-whatever
```

**Expect**: the server starts. An unknown model is not an error — the table is
empirical and always behind. It will fail on the first request with OpenAI's own
404, which is unambiguous, and `LLM_TEMPERATURE` remains the escape hatch.

## 6. No configuration file can name an embedding model (FR-004, SC-004)

```bash
grep -rn "embedding" .config.schema.yaml config_default.yml
```

**Expect**: matches only in comments, which explain why there is no such field; no
`embedding` model field. (`tests/agent/test_model_configuration.py:93-109` makes the
same check, skipping comment lines.) The embedding model is read from
the bundle path by `resolve_embedding_model()`, because a query embedded with a
different model than built the vectors returns nonsense rather than an error.

```bash
poetry run pytest tests/agent/test_embedding_model_resolution.py -q
```

## 7. The gates

```bash
poetry run ruff check . && poetry run mypy . && poetry run pytest
```

When this shipped, the existing tests in `tests/util/test_config.py` passed
unmodified (6f016dc), showing that adding the section did not change what the loader
does with an invalid file. The same PR then added tests for unknown top-level keys
(ce7cfef).
