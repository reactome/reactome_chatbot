# Quickstart: verifying model configuration

How to prove this feature works, end to end, the way a deployment uses it. Every
step here is a thing that can be observed failing, not a unit test.

## Prerequisites

- an installed reactome bundle (`./bin/embeddings_manager which`)
- `OPENAI_API_KEY` in `.env`

## 1. Nothing configured behaves exactly as before (FR-002)

The most important scenario, because it is what every existing deployment does.

```bash
grep -c '^llm:' config.yml || echo "no llm section — good"
poetry run chainlit run bin/chat-chainlit.py
```

**Expect**: the server starts and the startup log names `gpt-4o-mini`. Ask a
question; it answers.

## 2. A model set in config.yml is the one that answers (FR-001, FR-008)

```yaml
# config.yml
llm:
  provider: openai
  model: gpt-5.6-luna
```

**Expect**: the startup log names `gpt-5.6-luna`, and `resolve_temperature` has
silently sent `1.0` — that model refuses `0.0` and nobody had to know.

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
fix. It must not start and fail later — that is the entire point of Stage 2.

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

**Expect**: no `embedding` model field anywhere. The embedding model is read from
the bundle path by `resolve_embedding_model()`, because a query embedded with a
different model than built the vectors returns nonsense rather than an error.

```bash
poetry run pytest tests/agent/test_embedding_model_resolution.py -q
```

## 7. The gates

```bash
poetry run ruff check . && poetry run mypy . && poetry run pytest
```

`tests/util/test_config.py` must pass **untouched**: adding a section must not
change what the loader does with an invalid file.
