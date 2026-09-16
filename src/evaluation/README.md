# RAGAS Evaluation Toolkit

Scores the answers the chatbot actually gives.

- `evaluator.py` (run it as `./bin/evaluate`) asks the **shipping RAG chain** a set
  of questions and scores the answers with ragas: faithfulness, answer relevancy,
  context utilization, and context recall when reference answers are supplied.
`test_generator.py` used to sit beside it and was **deleted** in the LangChain 1.x
upgrade. It targeted the ragas 0.1 API — `from_langchain(generator_llm=,
critic_llm=)`, `generate_with_langchain_docs(test_size=, distributions=)` — none
of which exists in the pinned 0.4, and its own TODO had said so since 0.2. It was
not a port away from working; it had not run for three major versions, and
nothing imported it. `git log -- src/evaluation/test_generator.py` has it.

Generating reference answers is still worth having — it is what would let
`context_recall` run. Rebuilding it against the current ragas synthesizer API is
a real piece of work, not a rename, and belongs with whoever wants that metric.
`tests/golden/questions.txt` is what the evaluator uses by default and needs no
generation.

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

### Surviving a run that goes wrong

A run is bought, question by question, and `--out` is written only at the very
end — after every model and every repeat. A rate limit on question 18 of 20 used
to discard the seventeen already paid for.

```bash
# keep every answer as it is produced
poetry run ./bin/evaluate --model gpt-4o-mini --transcript-log run.jsonl

# answer four questions at once
poetry run ./bin/evaluate --model gpt-4o-mini --concurrency 4
```

`--transcript-log` appends one JSON object per answer, flushed immediately, so
whatever was bought survives the process that bought it.

A question that fails no longer ends the run. It is named on stderr, listed
under `failed_questions` in the report, and the remaining questions are scored
without it — with the references re-aligned to the questions that survived. That
alignment is the part worth knowing about: dropping a question from the middle
and *not* re-aligning would score every later answer against the wrong
reference, which produces numbers rather than an error.

`--concurrency` raises the rate of calls to the provider, so it makes rate
limits more likely — which is survivable now, and was not before. It does not
change what is measured: questions are independent, and results are placed by
index rather than appended as they arrive.

## What it measures, exactly

The chain `create_reactome_rag` builds, asked the **rephrased** question — which
is what `generate_answer` passes to the RAG in production, never the raw one.
That step is not cosmetic: over the 20 golden questions, 15 come back changed,
including `signalling` → `signaling`, which moves BM25's lexical matching.

Not measured, because they do not change the answer text: the safety check,
intent classification (this always evaluates the reactome source), and
postprocessing, which appends web-search results as separate content rather than
rewriting the answer.

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

## A deprecation this file carries knowingly

The four metrics are imported from `ragas.metrics`, which warns on every run:

> Importing Faithfulness from 'ragas.metrics' is deprecated and will be removed
> in v1.0. Please use 'ragas.metrics.collections' instead.

Not a rename. The replacements exist but two are renamed — `ResponseRelevancy`
is `AnswerRelevancy`, `LLMContextRecall` is `ContextRecall` — and they take the
judge model as a **constructor** argument rather than through `evaluate()`:

```
TypeError: Faithfulness.__init__() missing 1 required positional argument: 'llm'
```

So moving is a rework of how metrics are built, and it may change the scores,
which for a measurement tool is the part that needs care rather than the import.
Worth doing before this tool has produced numbers anyone is comparing against —
which is now.

## Reading the output

A single run has no noise floor: retrieval is not deterministic (Chroma's ANN
search varies run to run), so a difference between two single runs cannot be told
apart from variance. Use `--repeat 3` and compare against the reported spread.

## The answer sweep

```bash
./bin/answer-sweep                  # all of them
./bin/answer-sweep --only gsea      # one
```

Eleven questions the chatbot has got wrong before, each with what a good answer
must and must not contain, run end to end through the compiled graph. Exits
non-zero on a failure, so it can gate a deploy.

This is not the evaluator. `evaluator.py` scores answer *quality* with ragas and
costs real money; this asks something cheaper — is the chatbot still doing the
thing it was fixed to do — and takes about two and a half minutes.

**Why it exists.** Every regression in the week of 2026-09-14 was found the same
way: someone asked beta a question and the answer was wrong. The safety checker
refusing "can you run gsea for me". Ordinary retrieval taken down for a day by a
shared Chroma settings object. A live answer that was correct and displayed
nothing. Each was caught by a person noticing, which is slow, and only happens
for questions people happen to ask.

`must_not` matters as much as `must`. Most of those failures produced confident,
plausible text: *"Reactome does not provide a specific tool"* is a fluent
sentence and a false one about the flagship feature.

**Transient upstream failures are retried once, and the retry is reported.** On
its first run against production this caught a real degradation — reactome.org
served a Cloudflare challenge to a burst of requests and the answer became "I
could not find out ... due to a service error". That is the error handling
working rather than a regression, and a sweep that cries wolf gets ignored. The
retry is deliberately narrow: only an exception, or an answer that says the
lookup failed.
