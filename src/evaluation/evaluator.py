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
import time
from pathlib import Path
from typing import Any

import nltk
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from ragas import EvaluationDataset, SingleTurnSample, evaluate
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


def answer_questions(
    chain: Any, questions: list[str]
) -> tuple[list[str], list[list[str]], float]:
    """Ask the chain each question, keeping the answer and the retrieved context."""
    answers: list[str] = []
    contexts: list[list[str]] = []
    started = time.monotonic()
    for i, question in enumerate(questions, start=1):
        print(f"    [{i}/{len(questions)}] {question[:70]}", file=sys.stderr)
        response = chain.invoke({"input": question, "chat_history": []})
        answers.append(response["answer"])
        contexts.append([doc.page_content for doc in response["context"]])
    return answers, contexts, time.monotonic() - started


def build_chain(model: str, embeddings_dir: Path) -> Any:
    """The chain under test, built exactly as the application builds it."""
    llm: BaseChatModel = get_llm(
        "openai",
        model,
        request_timeout=360.0,
        temperature=resolve_temperature(model),
    )
    embedding = get_embedding("openai", resolve_embedding_model())
    return create_reactome_rag(llm, embedding, embeddings_dir)


def score(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    references: dict[str, str],
    judge: str,
) -> tuple[dict[str, float], list[dict[str, float]]]:
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

    samples = [
        SingleTurnSample(
            user_input=q,
            response=a,
            retrieved_contexts=c,
            reference=references.get(q),
        )
        for q, a, c in zip(questions, answers, contexts, strict=True)
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
    args = parser.parse_args()

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
        chain = build_chain(model, embeddings_dir)
        for run in range(1, args.repeat + 1):
            print(f"  {model}  run {run}/{args.repeat}", file=sys.stderr)
            answers, contexts, elapsed = answer_questions(chain, questions)
            seconds.append(elapsed / len(questions))
            aggregate, per_question = score(
                questions, answers, contexts, references, args.judge_model
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
                        "answer": a,
                        "documents_retrieved": len(c),
                        "scores": s,
                    }
                    for q, a, c, s in zip(
                        questions, answers, contexts, per_question, strict=True
                    )
                ]
            )
        report["models"][model] = {
            "runs": runs,
            "seconds_per_question": seconds,
            "transcripts": transcripts,
        }

    print_report(report)
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
