# Research: OpenAI reasoning effort (2026-11-03 to 2026-11-05)

These notes cover the installed libraries, langchain-openai 1.6.2 and openai 2.54.0,
and API facts measured on 2026-11-03 with the project's key using tiny requests
(about 150). The R numbers are cited from code (`src/agent/graph.py`,
`src/util/config_yml/models.py`) and are kept stable.

## R1 — What the API accepts (measured 2026-11-03)

gpt-6-luna:

| Request | Result |
|---|---|
| `temperature=0.0` | 400: "Only the default (1) value is supported." |
| `temperature=1.0` | accepted |
| `reasoning_effort` `none` / `low` / `medium` / `high` / `xhigh` | accepted |
| `reasoning_effort=minimal` | 400, listing the five above |
| the three graders (`json_schema`) at `reasoning_effort=medium` | valid output, one question each |
| streamed, with `stream_usage` | text chunks, and usage reports `output_token_details.reasoning` |

Across the 38 chat models `models.list()` returned (dated snapshots counted with their
base model), one request each at `medium`:

- gpt-4o, gpt-4o-mini, gpt-4.1 and gpt-4.1-mini reject it ("Unrecognized request
  argument supplied: reasoning_effort").
- gpt-3.5-turbo, gpt-4, gpt-4-turbo, gpt-4.1-nano, o1, o3-mini, o4-mini and the
  dated `*-chat-latest` models returned 404 (deprecated).
- Every other model accepted it. The levels each accepts:

| Model | none | minimal | low | medium | high | xhigh |
|---|---|---|---|---|---|---|
| chat-latest | – | – | – | ✓ | – | – |
| gpt-5, gpt-5-mini, gpt-5-nano | – | ✓ | ✓ | ✓ | ✓ | – |
| gpt-5.1 | ✓ | – | ✓ | ✓ | ✓ | – |
| gpt-5.2 to gpt-5.5, gpt-5.6-*, gpt-6-luna, gpt-6-sol, gpt-6.1-terra | ✓ | – | ✓ | ✓ | ✓ | ✓ |
| gpt-6-astra, gpt-6.1-sol, o3 | – | – | ✓ | ✓ | ✓ | ✓ |

Only `medium` is accepted everywhere. `low`, the non-answer default since 2026-11-05,
is accepted by every model except chat-latest.

A 16-token output cap made seven of these requests fail with a 400 that had nothing
to do with `reasoning_effort`. All seven passed when rerun with a 2048-token cap. So
a small cap is not a reliable way to probe a reasoning model.

## R2 — Sending no temperature

`ChatOpenAI(temperature=None, reasoning_effort=...)` leaves `temperature` out of the
request on 1.6.2. The old note in `get_llm`, "there is no way to send no temperature",
was true of 0.2.14 and became false with the 1.x upgrade (`4e607c8`).
`reasoning_effort` on its own keeps the request on `/v1/chat/completions`, which the
graders' `json_schema` method and the `astream_answer` streaming boundary both rely on.

As a side effect, this fixes a mismatch that was never reported. The library drops
any temperature other than 1 for `gpt-5*` models, so the `0.0` the table chose for
gpt-5.1, 5.2 and 5.4 was never actually sent.

## R3 — One rule per provider, no table

| Provider | Temperature | Reasoning effort |
|---|---|---|
| `openai` | none; a configured one or `LLM_TEMPERATURE` is refused | the configured level, otherwise `medium` for answer calls and `low` for the rest |
| `ollama` | `LLM_TEMPERATURE`, else configured, else `0.0`, as before | none; either key configured is refused |

Rejected alternatives:

- A per-model table of accepted levels. It would need editing for every new model,
  and would not check models missing from it.
- A hand-kept list of the older models to refuse. R4 catches them without a list.

## R4 — The startup request

`check_model` sends `"Reply with the single word OK."` through the same model object
the calls will use, so the parameters are exactly theirs. It has no output cap (R1).
A rejection from the API (any status other than 429 and 5xx) stops the process. The
message quotes the provider's error, and for a rejected level that error lists the
accepted ones. It then picks a fix from the error's `param` and `code` when the API
sends them, and from its text otherwise:

- "Unrecognized" or `unsupported_parameter` `reasoning_effort`: the model takes
  none, so choose a current one.
- Any other `reasoning_effort` error: set the key for that kind of call to a level the
  model accepts.
- Anything else: check `OPENAI_API_KEY`, `LLM_BASE_URL` and the model name.

No answer is not a rejection: a connection error or timeout, a 429, or a 5xx, after
the client's own retries. These say nothing about the settings, so they are logged
as a warning and startup continues. Refusing would keep the server down through a
provider outage (revised 2026-11-09; it used to refuse on any failure).

A refusal is a `ModelConfigError`, not a `SystemExit`. `AgentGraph` and the
command-line tools turn it into an exit at startup.

It runs once per process, before slow work:

- **The server**: in `AgentGraph.__init__`, before the roughly 85 seconds of BM25
  indexing. It checks the answer model only if its parameters differ.
- **The command-line tools**: the evaluator (each model and the judge), the routing
  probe and the retrieval baseline each check when they start.
- **Not checked**: `_summary_llm`, which runs inside a request and builds from
  `AgentGraph.answer_model`, the answer model startup already resolved and checked;
  and Ollama.

It costs one short request per process start, two when the two levels differ.

Rejected alternatives: checking on the first question, which makes a user find the
failure, and an off switch, which would end up off on the one deployment where it
mattered.

## R5 — One decision for every place that builds a model

`resolve_model_parameters` decides what to send and does no network I/O.
Each caller resolves once and passes the result to `build_llm`, which builds and runs
R4 when asked. Resolving once means `AgentGraph` logs and compares the same parameters
it builds with. `build_llm` and `get_llm` both take the result as a required argument,
so a caller that skips the decision fails type checking.

| Site | Model | `llm:` config | Checked |
|---|---|---|---|
| `AgentGraph.__init__` (two models) | deployment's | yes | yes |
| `_summary_llm` (answer model) | deployment's, from `AgentGraph.answer_model` | yes, as resolved at startup | no |
| evaluator, model under test (two models) | `--model`, else the default | yes | yes |
| evaluator, judge | `--judge-model`, else gpt-6-astra | no | yes |
| routing probe | deployment's (was the default, at `0.0`) | yes (was no) | yes |
| retrieval baseline, self-query | default (was gpt-4o-mini) | no | yes |

The judge reads no config because a deployment's levels are chosen for its answering
model, and may be ones gpt-6-astra rejects (`none`, R1). The baseline reads none so
that its results don't move when one deployment's config changes. The answer sweep
also ignored `config.yml`, and was fixed here.

## R6 — Streaming with a reasoning model

`astream_answer` decides where the answer starts by the order of events, not by
anything about the model. On chat completions a reasoning model streams plain text
(R1), so the code did not change. It was not checked end to end on gpt-6-luna.

## R7 — Measurement (not kept)

`src/evaluation/answer_cost.py` was written to record latency and per-call tokens on
the served path, with prices passed in and a check that each call was sent the right
level. It was left out of the repository on 2026-11-09 without being run. What
remains is `astream_answer`'s `callbacks` argument. The chat's own
`OpenAICallbackHandler.total_cost` was not usable, since it records $0 for models
missing from its price table.

## R8 — The evaluator's judge

The default judge changed from gpt-4o, which rejects a reasoning effort (R1), to
gpt-6-astra, which is not the answering default, so the "judge is under test" guard
still holds. The evaluator's default `--model` is `DEFAULT_LLM_MODEL`, not a model
name written into the code. Scores from the two judges cannot be compared.
