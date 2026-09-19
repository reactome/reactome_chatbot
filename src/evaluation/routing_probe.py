"""Does the classifier ever choose each collection?

`answer-sweep` cannot answer this, and the gap is structural rather than an
omission. Measured 2026-09-19 (specs/009-collection-routing/research.md):
neither `complexes` nor `reactions` can be guarded by asking a question,
because their content is duplicated in `summations` prose and in the
input/output names carried by `reactions`. Every candidate answered just as
well with the collection removed.

So a classifier that silently stops routing to a collection produces answers
that are confident, plausible and slightly worse, while every tracked question
still passes. That is the failure T005 was written for, T005a does not catch
it -- it catches a collection being unreachable, not unchosen -- and no
deterministic test can, because it is one model call's judgement.

This is the observation you have to keep making instead. It asks the
classifier directly, one call per question, and checks *what was selected*
rather than what any answer said. Two properties:

- **Coverage**: every collection in the bundle is chosen by at least one
  question. This is the T005b guard -- it is what notices `complexes` going
  unchosen.
- **No wrong narrow**: a question that narrows at all must include the
  collection its answer lives in. Leaving the selection empty is always
  allowed, because widening is the safe direction and the prompt asks for it
  whenever the model is unsure.

Empty selections are reported, not failed. A run where everything is empty is
routing doing nothing, which is worth seeing and is not incorrect.
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass, field

from agent.graph import resolve_llm_model
from agent.models import get_llm
from agent.tasks.intent_classifier import (
    SourceName,
    create_intent_classifier,
)
from reactome_mcp.session import is_configured
from retrievers.reactome.metadata_info import reactome_descriptions_info


@dataclass(frozen=True)
class Probe:
    question: str
    #: The collection this question's answer lives in. If the classifier
    #: narrows at all, this must be among what it chose.
    expect: str
    why: str


PROBES: tuple[Probe, ...] = (
    Probe(
        question="What is the UniProt accession for the TP53 protein in Reactome?",
        expect="ewas",
        why="UniProt links live only in ewas; this is the tracked question too.",
    ),
    Probe(
        question="What is the UniProt identifier for BRCA1 in Reactome?",
        expect="ewas",
        why="A second accession question, so one passing is not luck.",
    ),
    Probe(
        question="Which diseases involve variants of the PTEN gene in Reactome?",
        expect="disease_variants",
        why="The variants themselves are only in disease_variants.",
    ),
    Probe(
        question="List the ABCA1 variants in Reactome and the disease each causes.",
        expect="disease_variants",
        why="Measured: the prose survives narrowing, the variant names do not.",
    ),
    Probe(
        question="What does Reactome's summary of Selective autophagy say?",
        expect="summations",
        why="The curated prose lives only in summations.",
    ),
    Probe(
        question="How does Reactome describe the Wnt signalling pathway?",
        expect="summations",
        why="A second prose question.",
    ),
    Probe(
        question="What are the protein components of the Nup107 complex in Reactome?",
        expect="complexes",
        why="The whole reason this file exists: complexes cannot be guarded by "
        "an answer, so it is guarded by the routing decision instead.",
    ),
    Probe(
        question="Which proteins make up the PAM complex in Reactome?",
        expect="complexes",
        why="A second complexes question, because it is the unguarded one.",
    ),
    Probe(
        question="What are the inputs and outputs of the reaction where CDK1 "
        "phosphorylates MCM2?",
        expect="reactions",
        why="Inputs, outputs and catalysts are the shape reactions holds.",
    ),
    Probe(
        question="Which reaction converts cholesterol in the ABCA1 pathway, and "
        "what catalyses it?",
        expect="reactions",
        why="A second reactions question.",
    ),
)


@dataclass
class Result:
    probe: Probe
    chose: list[str] = field(default_factory=list)
    error: str = ""

    @property
    def widened(self) -> bool:
        """Chose nothing, meaning search everything. Always allowed."""
        return not self.chose and not self.error

    @property
    def wrong(self) -> bool:
        """Narrowed, but not to where the answer lives."""
        return bool(self.chose) and self.probe.expect not in self.chose


def available_sources() -> frozenset[SourceName]:
    """The prompt the deployment actually uses, so this probes the real one."""
    sources: set[SourceName] = {"reactome", "userguide"}
    if is_configured():
        sources.add("live")
    return frozenset(sources)


async def run(probes: tuple[Probe, ...] = PROBES) -> list[Result]:
    provider, model, base_url = resolve_llm_model(None)
    classifier = create_intent_classifier(
        get_llm(provider, model, base_url=base_url, request_timeout=120.0),
        available_sources(),
    )
    results = []
    for index, probe in enumerate(probes, start=1):
        print(f"  [{index}/{len(probes)}] {probe.question[:58]}", file=sys.stderr)
        result = Result(probe=probe)
        try:
            intent = await classifier.ainvoke({"rephrased_input": probe.question})
            result.chose = sorted(intent.collections)
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
        results.append(result)
    return results


def report(results: list[Result]) -> int:
    print()
    for r in results:
        if r.error:
            state = "ERROR"
        elif r.wrong:
            state = "WRONG"
        elif r.widened:
            state = "open "
        else:
            state = "ok   "
        print(
            f"  {state} {r.probe.expect:17s} {','.join(r.chose) or '(all)':32s} "
            f"{r.probe.question[:44]}"
        )
        if r.wrong:
            print(f"        narrowed away from {r.probe.expect}: {r.probe.why}")
        if r.error:
            print(f"        {r.error}")

    chosen = {c for r in results for c in r.chose}
    missing = sorted(set(reactome_descriptions_info) - chosen)
    wrong = [r for r in results if r.wrong]
    errors = [r for r in results if r.error]
    widened = [r for r in results if r.widened]

    print()
    print(
        f"  {len(results) - len(widened) - len(errors)} narrowed, "
        f"{len(widened)} left open, {len(wrong)} narrowed wrongly, "
        f"{len(errors)} errored"
    )

    if missing:
        print()
        print(f"  NEVER CHOSEN: {', '.join(missing)}")
        print("  A collection nothing routes to is served worse than before this")
        print("  feature existed, and answer-sweep cannot see it -- that is why")
        print("  this file exists. See specs/009-collection-routing (T005b).")
    else:
        print("  every collection was chosen by at least one question")

    return 1 if (missing or wrong or errors) else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(f"Asking the classifier {len(PROBES)} questions\n", file=sys.stderr)
    raise SystemExit(report(asyncio.run(run())))
