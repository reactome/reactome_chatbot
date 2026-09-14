"""Score the answers the chatbot actually gives, with ragas.

This measures **the shipping pipeline**. That is the whole point of the file and
was not true of it until now: it used to build its own `SelfQueryRetriever` +
`EnsembleRetriever` + `MergerRetriever` over the `summations` collection alone,
with `k=7` and weights `[0.2, 0.8]` that appear nowhere in the product. After the
retriever rewrite removed `SelfQueryRetriever` from the pipeline, it measured a
configuration that no longer existed anywhere -- producing numbers that looked
like an answer and were not.

It now calls `create_reactome_rag`, the same factory `bin/chat-chainlit.py`
reaches, so a change to retrieval cannot alter the product without altering the
measurement.

  # one model over the golden questions
  ./bin/evaluate --model gpt-4o-mini

  # two models, same questions, same judge, side by side
  ./bin/evaluate --model gpt-4o-mini --model gpt-5.6-luna

  # how much of a difference is just noise?
  ./bin/evaluate --model gpt-4o-mini --repeat 3

Requires an installed reactome bundle and OPENAI_API_KEY.
"""

import argparse
import json
import math
import os
import statistics
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple, cast

import nltk
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.dataset_schema import EvaluationResult, MultiTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    ContextUtilization,
    Faithfulness,
    LLMContextRecall,
    ResponseRelevancy,
)

from agent.graph import resolve_embedding_model, resolve_temperature
from agent.models import get_embedding, get_llm
from agent.tasks.rephrase import create_rephrase_chain
from retrievers.reactome.rag import create_reactome_rag
from util.embedding_environment import EmbeddingEnvironment

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_QUESTIONS = REPO_ROOT / "tests" / "golden" / "questions.txt"

# The judge is pinned and is NOT the model under test. Scoring gpt-5.6-luna with
# gpt-5.6-luna would ask a model to grade its own homework; and a judge that moves
# between two runs makes the two runs incomparable, which is the failure this
# whole file exists to avoid.
DEFAULT_JUDGE_MODEL = "gpt-4o"

# Used only to compare a question against an answer for answer_relevancy.
JUDGE_EMBEDDING_MODEL = "text-embedding-3-large"


def read_questions(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def read_references(path: Path) -> dict[str, str]:
    """Optional question -> reference answer map, as JSON.

    Only `context_recall` needs one. Without it that metric is dropped rather
    than scored against nothing.
    """
    data: dict[str, str] = json.loads(path.read_text())
    return data


def resolve_references(
    questions: list[str], references: dict[str, str]
) -> list[str | None]:
    """Line up reference answers with questions, by the ORIGINAL wording.

    A reference file is written against the questions a person typed. The
    evaluator retrieves and scores on the rephrased question, because that is
    what production does -- so the lookup has to happen before the rephrase, and
    the result travels positionally. Keying by the rephrased text instead matches
    nothing, and drops context_recall from every run while looking exactly like
    "no references were supplied".
    """
    return [references.get(question) for question in questions]


class QuestionFailed(NamedTuple):
    """A question that could not be answered, kept rather than thrown away."""

    # Not `index`: on a NamedTuple that shadows tuple.index.
    position: int
    question: str
    error: str


def answer_one(chain: Any, rephrase: Any, question: str) -> tuple[str, str, list[str]]:
    """Rephrase as production does, then ask the chain.

    The rephrase is not optional and not cosmetic. `generate_answer` passes
    `rephrased_input` to the RAG chain, never the raw question, so retrieval in
    production always happens on rewritten text. Measured over the 20 golden
    questions, **15 come back changed** -- including `signalling` -> `signaling`,
    which moves BM25's lexical matching outright.

    Skipping it would have left this file measuring a different retrieval from
    the one it claims to measure: the same defect as building a private
    retriever, one step further up.
    """
    standalone = rephrase.invoke({"user_input": question, "chat_history": []})
    response = chain.invoke({"input": standalone, "chat_history": []})
    return (
        standalone,
        response["answer"],
        [doc.page_content for doc in response["context"]],
    )


def answer_questions(
    chain: Any,
    rephrase: Any,
    questions: list[str],
    *,
    concurrency: int = 1,
    on_answered: Callable[[int, str, str, list[str]], None] | None = None,
) -> tuple[list[str], list[str], list[list[str]], float, list[QuestionFailed]]:
    """Answer every question, keeping what succeeded when something fails.

    Two properties this needs that the serial version did not have.

    **A failure must not cost the whole run.** Every answer here is paid for --
    a rephrase call, a retrieval, a generation -- and the previous version held
    all of it in memory until the last question returned, so a rate limit on
    question 18 of 20 discarded the seventeen already bought. Failures are now
    recorded and reported; the run continues.

    **Order must not depend on timing.** Results are placed by index, not
    appended as they arrive, so a concurrent run scores the same questions
    against the same references as a serial one. Getting this wrong would not
    crash; it would silently score answers against the wrong references, which
    is the kind of wrong that reads as a result.
    """
    total = len(questions)
    rephrased: list[str | None] = [None] * total
    answers: list[str | None] = [None] * total
    contexts: list[list[str] | None] = [None] * total
    failures: list[QuestionFailed] = []
    done = 0
    lock = threading.Lock()
    started = time.monotonic()

    def run(index: int) -> None:
        nonlocal done
        question = questions[index]
        try:
            standalone, answer, context = answer_one(chain, rephrase, question)
        # Deliberately broad: one bad question must not end the run.
        except Exception as exc:
            with lock:
                done += 1
                failures.append(QuestionFailed(index, question, repr(exc)))
                print(
                    f"    [{done}/{total}] FAILED {question[:60]}: {exc!r}",
                    file=sys.stderr,
                )
            return

        rephrased[index] = standalone
        answers[index] = answer
        contexts[index] = context
        with lock:
            done += 1
            print(f"    [{done}/{total}] {question[:70]}", file=sys.stderr)
            if standalone.strip() != question.strip():
                print(
                    f"          rephrased: {standalone.strip()[:70]}", file=sys.stderr
                )
            if on_answered is not None:
                on_answered(index, standalone, answer, context)

    if concurrency > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(run, range(total)))
    else:
        for index in range(total):
            run(index)

    elapsed = time.monotonic() - started

    # Drop the failures, keeping the three lists aligned with each other.
    kept = [i for i in range(total) if answers[i] is not None]
    return (
        [cast(str, rephrased[i]) for i in kept],
        [cast(str, answers[i]) for i in kept],
        [cast(list[str], contexts[i]) for i in kept],
        elapsed,
        failures,
    )


def _kept(index: int, failures: list[QuestionFailed]) -> bool:
    """Whether the question at this index produced an answer."""
    return all(failure.position != index for failure in failures)


def make_transcript_writer(
    path: Path | None, model: str, run: int, questions: list[str]
) -> Callable[[int, str, str, list[str]], None] | None:
    """Append each answer to a JSONL file as it is produced.

    The report is written once, at the end, after every model and every repeat.
    That is fine when nothing goes wrong and expensive when something does: a
    run that dies on the last question used to leave nothing at all, having paid
    for every answer before it.

    One line per answer, flushed immediately, so whatever was bought survives
    the process that bought it.
    """
    if path is None:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def write(index: int, rephrased: str, answer: str, context: list[str]) -> None:
        record = {
            "model": model,
            "run": run,
            "index": index,
            "question": questions[index],
            "rephrased": rephrased,
            "answer": answer,
            "documents_retrieved": len(context),
        }
        line = json.dumps(record, sort_keys=True) + "\n"
        # Serialised and flushed: concurrent workers share this file, and a
        # partial line is worse than a missing one.
        with lock, path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()

    return write


def build_chain(model: str, embeddings_dir: Path) -> tuple[Any, Any]:
    """The chain under test and the rephrase step, built as the application does.

    Production uses one LLM for both, so the model under test rephrases its own
    questions -- which is part of what is being compared.
    """
    llm: BaseChatModel = get_llm(
        "openai",
        model,
        request_timeout=360.0,
        temperature=resolve_temperature(model),
    )
    embedding = get_embedding("openai", resolve_embedding_model())
    return create_reactome_rag(llm, embedding, embeddings_dir), create_rephrase_chain(
        llm
    )


def score(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    references: list[str | None],
    judge: str,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Score answers. `questions` is what was retrieved on; `references` is
    positional, resolved by the caller from the ORIGINAL wording -- a reference
    file is written against the questions a person typed, not against whatever
    the rephrase step produced on the day."""
    judge_llm = LangchainLLMWrapper(
        get_llm("openai", judge, temperature=resolve_temperature(judge))
    )
    # answer_relevancy embeds the question and the answer to compare them. This
    # is the JUDGE's embedding and has nothing to do with the vectors in the
    # bundle, so it does not go through resolve_embedding_model -- but it must
    # not silently follow OPENAI_BASE_URL either. On the Plant Reactome host that
    # points at a self-hosted bge-m3 endpoint, and asking that for
    # text-embedding-3-large is a 404 in the middle of a run. api.openai.com is
    # named explicitly; JUDGE_BASE_URL overrides it.
    judge_embeddings = LangchainEmbeddingsWrapper(
        get_embedding(
            "openai",
            JUDGE_EMBEDDING_MODEL,
            base_url=os.getenv("JUDGE_BASE_URL", "https://api.openai.com/v1"),
        )
    )

    samples: list[SingleTurnSample | MultiTurnSample] = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
            reference=r,
        )
        for q, a, c, r in zip(questions, answers, contexts, references, strict=True)
    ]

    metrics: list[Any] = [Faithfulness(), ResponseRelevancy(), ContextUtilization()]
    if all(s.reference for s in samples):
        metrics.append(LLMContextRecall())
    else:
        print(
            "    (no reference answers, so context_recall is skipped rather than "
            "scored against nothing -- pass --references to include it)",
            file=sys.stderr,
        )

    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )
    # ragas 0.4 types evaluate() as EvaluationResult | Executor; it returns an
    # Executor only when asked to run asynchronously, which this does not do.
    # Checked rather than cast, so a future ragas that changes the default says
    # so here instead of failing on the next line with an AttributeError.
    if not isinstance(result, EvaluationResult):
        raise TypeError(
            f"ragas returned {type(result).__name__}, not EvaluationResult. "
            "evaluate() now defers by default; this tool expects a completed run."
        )
    # result.scores is a public field holding one dict of metric -> score per
    # sample. The aggregate used to come from result._repr_dict, which is private
    # and would break on a ragas upgrade without warning -- in a file whose whole
    # job is to stay trustworthy across upgrades.
    per_question: list[dict[str, float]] = [dict(s) for s in result.scores]

    # A metric that fails on one sample comes back NaN, and NaN propagates
    # through fmean -- so one bad question would turn the whole aggregate into
    # NaN, which prints as "nan" and looks like a broken tool rather than a
    # partial result. Non-finite scores are dropped and the drop is reported,
    # because silently averaging over fewer questions than were asked is the kind
    # of quiet difference this file is supposed to catch, not commit.
    aggregate: dict[str, float] = {}
    for metric in per_question[0]:
        values = [
            s[metric]
            for s in per_question
            if s.get(metric) is not None and math.isfinite(s[metric])
        ]
        dropped = len(per_question) - len(values)
        if dropped:
            print(
                f"    {metric}: {dropped} of {len(per_question)} questions could "
                "not be scored and are excluded from the mean",
                file=sys.stderr,
            )
        aggregate[metric] = statistics.fmean(values) if values else float("nan")
    return aggregate, per_question


def main() -> None:
    load_dotenv()
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="models",
        help="Model under test. Repeat to compare several against one judge.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"Model that scores the answers (default: {DEFAULT_JUDGE_MODEL}).",
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument(
        "--references",
        type=Path,
        help="JSON {question: reference answer}; enables context_recall.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run each model this many times, and report the spread as the noise "
        "floor. A difference smaller than it is not a result.",
    )
    parser.add_argument("--limit", type=int, help="Use only the first N questions.")
    parser.add_argument(
        "--embeddings-dir", type=Path, default=EmbeddingEnvironment.get_dir("reactome")
    )
    parser.add_argument("--out", type=Path, help="Write the full report as JSON.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Answer this many questions at once (default 1). Questions are "
        "independent, so this does not change what is measured -- but it does "
        "raise the rate of calls to the provider, and a rate limit now costs "
        "one question rather than the run.",
    )
    parser.add_argument(
        "--transcript-log",
        type=Path,
        help="Append every answer here as it is produced, one JSON object per "
        "line. A run that dies partway keeps everything already paid for.",
    )
    args = parser.parse_args()

    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1.")

    failed_questions: list[dict[str, Any]] = []
    models: list[str] = args.models or ["gpt-4o-mini"]
    if args.judge_model in models:
        raise SystemExit(
            f"The judge ({args.judge_model}) is also under test. A model grading "
            "its own answers is not a measurement. Pass a different --judge-model."
        )
    embeddings_dir: Path | None = args.embeddings_dir
    if embeddings_dir is None:
        raise SystemExit(
            "No reactome embeddings installed. Run "
            "./bin/embeddings_manager install <embedding-id>, or pass "
            "--embeddings-dir."
        )

    questions = read_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    references = read_references(args.references) if args.references else {}

    print(
        f"{len(questions)} questions x {len(models)} model(s) x {args.repeat} run(s), "
        f"judged by {args.judge_model}\nbundle: {embeddings_dir}\n",
        file=sys.stderr,
    )

    report: dict[str, Any] = {
        "judge_model": args.judge_model,
        "bundle": str(embeddings_dir),
        "questions": len(questions),
        "repeat": args.repeat,
        "models": {},
    }

    for model in models:
        runs: list[dict[str, float]] = []
        seconds: list[float] = []
        transcripts: list[list[dict[str, Any]]] = []
        chain, rephrase = build_chain(model, embeddings_dir)
        for run in range(1, args.repeat + 1):
            print(f"  {model}  run {run}/{args.repeat}", file=sys.stderr)
            rephrased, answers, contexts, elapsed, failures = answer_questions(
                chain,
                rephrase,
                questions,
                concurrency=args.concurrency,
                on_answered=make_transcript_writer(
                    args.transcript_log, model, run, questions
                ),
            )
            if failures:
                # Named, not counted. "3 failed" tells you nothing about
                # whether the run is still worth reading.
                print(
                    f"    {len(failures)} of {len(questions)} questions failed:",
                    file=sys.stderr,
                )
                for failure in failures:
                    print(
                        f"      [{failure.position + 1}] {failure.question[:60]} "
                        f"-> {failure.error}",
                        file=sys.stderr,
                    )
                failed_questions.extend(
                    {
                        "model": model,
                        "run": run,
                        "question": failure.question,
                        "error": failure.error,
                    }
                    for failure in failures
                )
            if not answers:
                print(
                    f"    every question failed for {model}; skipping scoring",
                    file=sys.stderr,
                )
                continue
            # Per answered question, so a run that lost some is still
            # comparable on rate rather than on total.
            seconds.append(elapsed / len(answers))
            # Scored against the rephrased question, because that is what was
            # retrieved on and answered. Judging the answer against the original
            # wording would penalise a faithful answer for a rewrite the product
            # performs deliberately.
            # The references have to follow the questions that survived. A
            # failure removes a question from the middle of the list, so
            # passing the full reference list would score every later answer
            # against the wrong reference -- silently, and plausibly.
            answered = [
                questions[i] for i in range(len(questions)) if _kept(i, failures)
            ]
            aggregate, per_question = score(
                rephrased,
                answers,
                contexts,
                resolve_references(answered, references),
                args.judge_model,
            )
            runs.append(aggregate)
            # The answers themselves, not only the scores. The version this
            # replaced wrote responses to a spreadsheet; dropping that would have
            # left a low faithfulness score with nothing to look at to find out
            # why.
            transcripts.append(
                [
                    {
                        "question": q,
                        "rephrased": r,
                        "answer": a,
                        "documents_retrieved": len(c),
                        "scores": s,
                    }
                    for q, r, a, c, s in zip(
                        answered,
                        rephrased,
                        answers,
                        contexts,
                        per_question,
                        strict=True,
                    )
                ]
            )
        report["models"][model] = {
            "runs": runs,
            "seconds_per_question": seconds,
            "transcripts": transcripts,
        }

    # In the report, not only on stderr: a scored run with three questions
    # missing is a different measurement from a complete one, and whoever reads
    # the JSON later will not have the terminal output.
    if failed_questions:
        report["failed_questions"] = failed_questions

    print_report(report)
    if failed_questions:
        print(
            f"\n  {len(failed_questions)} question-run(s) failed; scores above "
            "cover only what succeeded.",
            file=sys.stderr,
        )
    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote {args.out}", file=sys.stderr)


def print_report(report: dict[str, Any]) -> None:
    models: dict[str, Any] = report["models"]
    metric_names = sorted({m for v in models.values() for m in v["runs"][0]})

    print(
        f"\n{report['questions']} questions, judged by {report['judge_model']}, "
        f"{report['repeat']} run(s) each\n"
    )
    header = f"  {'model':<18}" + "".join(f"{m[:16]:>18}" for m in metric_names)
    print(header + f"{'s/question':>13}")
    print("  " + "-" * (len(header) + 11))
    for model, data in models.items():
        cells = ""
        for metric in metric_names:
            values = [r[metric] for r in data["runs"]]
            mean = statistics.fmean(values)
            spread = (max(values) - min(values)) if len(values) > 1 else 0.0
            cells += (
                f"{mean:>13.3f}±{spread:.3f}" if len(values) > 1 else f"{mean:>18.3f}"
            )
        secs = statistics.fmean(data["seconds_per_question"])
        print(f"  {model:<18}{cells}{secs:>13.1f}")

    if report["repeat"] > 1:
        print(
            "\n  The ± is the observed spread across runs -- the noise floor. A\n"
            "  difference between models smaller than it is not a result."
        )
    else:
        print(
            "\n  Single run, so there is no noise floor here and no way to tell a\n"
            "  real difference from run-to-run variance. Use --repeat 3 to compare."
        )


if __name__ == "__main__":
    main()
