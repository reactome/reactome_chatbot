# RAGAS Evaluation Toolkit

Scores the answers the chatbot actually gives.

- `evaluator.py` (run it as `./bin/evaluate`) asks the **shipping RAG chain** a set
  of questions and scores the answers with ragas: faithfulness, answer relevancy,
  context utilization, and context recall when reference answers are supplied.
- `test_generator.py` synthesizes question/answer sets from the example corpora.
  **It targets the ragas 0.1 API and does not run against the pinned 0.2** — see
  the TODO in the file. `tests/golden/questions.txt` is what the evaluator uses by
  default and needs no generation.

## What changed, and why the old flags are gone

`--rag_type basic|advanced` and `--testset_dir` no longer exist.

The evaluator used to build its own retriever — `SelfQueryRetriever` +
`EnsembleRetriever` + `MergerRetriever`, over the `summations` collection alone,
with `k=7` and weights `[0.2, 0.8]` that appear nowhere in the product. The
retriever rewrite then removed `SelfQueryRetriever` from the pipeline, so the
evaluator was measuring a configuration that existed nowhere. `basic` and
`advanced` were two shapes of that private stack, not two shapes of the product.

It now calls `create_reactome_rag`, the same factory `bin/chat-chainlit.py` uses.
There is only one pipeline to measure, so there is no `--rag_type`.

Reference answers, needed only for `context_recall`, come from `--references`
as JSON rather than from spreadsheets:

```json
{ "What role does TP53 play in apoptosis?": "TP53 induces apoptosis by ..." }
```

## Requirements

- An installed reactome bundle (`./bin/embeddings_manager install ...`)
- `OPENAI_API_KEY`

`poetry run` is needed for the interpreter, not for the import path: the script
puts `src/` on `sys.path` itself, so it works from any directory.

## Usage

```bash
# one model over the golden questions
poetry run ./bin/evaluate --model gpt-4o-mini

# two models, same questions, same judge, side by side
poetry run ./bin/evaluate --model gpt-4o-mini --model gpt-5.6-luna

# three runs each, so the report can show the noise floor
poetry run ./bin/evaluate --model gpt-4o-mini --repeat 3 --out report.json
```

`--out` writes the full report: aggregate scores per run, seconds per question,
and every answer with its per-question scores — so a low score can be looked at
rather than guessed about.

## The judge

Scoring is done by a separate model, `gpt-4o` by default, pinned with
`--judge-model`. Two rules the tool enforces or documents:

- **The judge may not be a model under test.** The tool refuses. A model grading
  its own answers is not a measurement.
- **The judge must not change between runs being compared.** A moving judge makes
  two runs incomparable, which is the failure this tool exists to avoid.

The judge's embedding model is only used to compare a question against an answer;
it is unrelated to the vectors in the bundle, and it is pointed at
`api.openai.com` explicitly rather than following `OPENAI_BASE_URL`, which on the
Plant Reactome host points at a self-hosted endpoint that does not serve it.
`JUDGE_BASE_URL` overrides.

## Reading the output

A single run has no noise floor: retrieval is not deterministic (Chroma's ANN
search varies run to run), so a difference between two single runs cannot be told
apart from variance. Use `--repeat 3` and compare against the reported spread.
