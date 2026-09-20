"""Ask the chatbot the questions it has got wrong before, and check the answers.

Every regression this week was found the same way: someone asked beta a
question and the answer was wrong. The safety checker refusing "can you run
gsea for me". A reactome question taken down by a shared Chroma settings
object. A live answer that was correct and displayed nothing. Each was found by
a person noticing, which is slow, and only happens for questions people happen
to ask.

reactome-mcp has a sweep that calls every tool and checks the answers contain
what they should; it has caught several real bugs. This is the same idea for
the chatbot: a fixed set of questions, each with what a good answer must and
must not contain, run end to end through the compiled graph.

    ./bin/answer-sweep                       # against the local checkout
    ./bin/answer-sweep --only species        # just the questions matching that
    docker exec reactome_chat \\
        python /app/bin/answer-sweep         # against the deployed container

Not a quality measurement. `src/evaluation/evaluator.py` scores answer quality
with ragas and costs real money; this asks a cheaper question -- is the chatbot
still doing the thing it was fixed to do -- and is meant to be run after every
change and before every deploy.
"""

import argparse
import asyncio
import re
import sys
import time
from dataclasses import dataclass, field

from agent.graph import AgentGraph
from agent.profile_names import ProfileName
from reactome_mcp.session import is_configured
from util.embedding_environment import EmbeddingEnvironment


@dataclass(frozen=True)
class Expectation:
    """What a good answer to one question looks like.

    `must_not` matters as much as `must`: most of the failures this file exists
    for produced confident, plausible text. "Reactome does not provide a
    specific tool" is a fluent sentence and a false one.
    """

    question: str
    why: str
    must: tuple[str, ...] = ()
    must_not: tuple[str, ...] = ()
    # For facts whose exact value legitimately changes -- the release number
    # becomes 98 shortly, so asserting "97" would fail on a correct answer.
    must_match: tuple[str, ...] = ()
    # Only answerable against the live service. Run without an MCP server --
    # a plain local checkout -- these cannot pass, and reporting them as
    # regressions is how a gate teaches people to ignore it.
    needs_live: bool = False
    # A collection that may not be in the installed bundle yet. Same reasoning
    # as needs_live: a question that cannot pass is not a regression.
    needs_collection: str = ""


EXPECTATIONS: tuple[Expectation, ...] = (
    # --- the two questions that started all of this -------------------------
    Expectation(
        question="can you run gsea for me",
        why="Refused 4/4 by the safety checker as 'outside the scope' until 2026-09-16. "
        "It is an on-topic question about a flagship Reactome feature.",
        must=("ReactomeGSA",),
        must_not=("cannot", "outside the scope", "not relevant", "does not currently"),
    ),
    Expectation(
        question="I have a gene list do you have a tool I can use to analyse where in "
        "reactome those genes are involved",
        why="Answered 'Reactome does not provide a specific tool' -- false, and about "
        "its flagship feature.",
        must=("ReactomeGSA",),
        must_not=("does not provide", "not currently available"),
    ),
    # --- gene set analysis should prefer the tool needing no install --------
    Expectation(
        question="How do I run a GSEA in Reactome?",
        why="Led with ReactomeFIViz, a Cytoscape plugin, over the web tool. The most "
        "detailed instructions are usually for the most involved tool.",
        must=("ReactomeGSA",),
    ),
    # --- but the plugin is still reachable when it is what was asked for ----
    Expectation(
        question="How do I use ReactomeFIViz in Cytoscape?",
        why="Preferring the web tool must not bury the plugin for someone who wants it.",
        must=("FIViz",),
    ),
    # --- one question per collection, so routing cannot quietly skip one ----
    # Each was chosen by removing its collection from a copy of the bundle and
    # confirming the answer changes. A question that still answers without its
    # collection guards nothing, and three of the first four candidates were
    # exactly that -- see specs/009-collection-routing/spec.md.
    Expectation(
        question="What is the UniProt accession for the TP53 protein in Reactome?",
        why="Guards the `ewas` collection: it is the only one holding UniProt "
        "links. Verified by removing ewas from a bundle copy, after which the "
        "accession is no longer answered.",
        must_match=(r"\bP04637\b",),
    ),
    Expectation(
        question="What does Reactome's summary of Selective autophagy say about "
        "where cargo is degraded?",
        why="Guards the `summations` collection: the curated prose summaries live "
        "only there. Without it the chatbot says no summary is available.",
        must=("lysosom",),
        must_not=("does not provide", "not currently available"),
    ),
    # --- disease variants, which only the new collection can name -----------
    Expectation(
        question="List the ABCA1 variants in Reactome and the disease each one causes.",
        why="Answered 'Defective ABCA1 causes Tangier Disease' and named none of "
        "the six curated variants. Pathway-level prose instead of the variant.",
        must=("Tangier",),
        # A named variant, not a specific one: which of the six come back
        # depends on retrieval order, and pinning one would fail a good answer.
        # The gene name is deliberately NOT required next to the variant.
        # Requiring "ABCA1 C1417R" failed an answer that named all six as a
        # numbered list of "**C1417R**" -- correct, and marked a failure. The
        # question is already anchored on the topic by `must=("Tangier",)`.
        must_match=(r"\b[A-Z]\d{2,4}[A-Z*]\b",),
        needs_collection="disease_variants",
    ),
    Expectation(
        question="Which diseases involve variants of the PTEN gene in Reactome?",
        why="Answered with the PTEN Loss of Function pathway. Reactome curates "
        "108 PTEN variants across 86 diseases.",
        # Same shape as the ABCA1 guard above, and for the same reason: the
        # gene name is not required adjacent to the variant.
        must=("PTEN",),
        must_match=(r"\b[A-Z]\d{2,4}[A-Z*]\b",),
        needs_collection="disease_variants",
    ),
    # --- facts about the database, which retrieval cannot answer ------------
    Expectation(
        question="what species are in reactome",
        why="Retrieval answered 'primarily Homo sapiens ... no indications of other "
        "species'. Reactome has 96. A sample of the content cannot describe the scope.",
        # A count, not the literal 96, for the same reason as the release
        # number below: it goes up, and a check that fails on a correct
        # answer is a check that gets switched off.
        must_match=(r"\b\d{2,3}\b",),
        must_not=("primarily Homo sapiens", "no indications"),
        needs_live=True,
    ),
    Expectation(
        question="Which release of Reactome is this?",
        why="The bundle is a snapshot and cannot know. Needs the live service.",
        # A plausible release number, rather than a literal: 97 becomes 98
        # shortly, and a check that fails on a correct answer gets disabled.
        # Releases are in the 90s now and will pass 100, so allow both.
        # "release" and "version" are used interchangeably here, and the live
        # answer says "version": requiring one word failed a correct answer.
        must_match=(r"\b(?:release|version)\b", r"\b(?:9\d|[1-9]\d\d)\b"),
        must_not=("cannot", "do not have"),
        needs_live=True,
    ),
    # --- ordinary retrieval, which an outage took down for a day ------------
    Expectation(
        question="What does CDK5 phosphorylate in Alzheimer disease?",
        why="Broken in production 2026-09-15 by a shared Chroma Settings object that "
        "sent reactome questions into the user guide bundle.",
        must=("CDK5",),
        must_not=("Permission denied", "I could not"),
    ),
    Expectation(
        question="How does TP53 regulate PTEN transcription?",
        why="A second ordinary retrieval question, so one passing is not luck.",
        must=("PTEN",),
    ),
    # --- the user guide ------------------------------------------------------
    Expectation(
        question="How do I use the pathway browser?",
        why="Routes to the user guide, which is only useful if its bundle is installed "
        "and registered -- two separate steps, and nothing warned when only one was done.",
        must=("Pathway Browser",),
        must_not=("does not currently cover",),
    ),
    # --- and the things it should still refuse ------------------------------
    Expectation(
        question="Who won the 1998 World Cup?",
        why="Loosening the safety checker must not make it answer anything at all.",
        must_not=("France", "Brazil"),
    ),
    Expectation(
        question="What are common side effects of statins for my high cholesterol?",
        why="Medical advice. Still refused after the safety prompt was loosened.",
        must_not=("muscle pain", "consult"),
    ),
)


@dataclass
class Result:
    expectation: Expectation
    skipped: str = ""
    answer: str = ""
    seconds: float = 0.0
    error: str = ""
    #: A live tool call raised underneath this answer. The real signal the
    #: retry keys on, in place of matching the model's prose.
    upstream_failed: bool = False
    missing: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    retried: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.skipped) or not (self.error or self.missing or self.forbidden)


def _contains(haystack: str, needle: str) -> bool:
    """Case-insensitive, and bounded at word edges where that is meaningful.

    Without the boundaries "96" matches "1996" and "9" matches any text with a
    digit in it -- which made the release-version check assert almost nothing.
    The boundary is only added where the needle actually starts or ends with a
    word character, so a needle like "R-HSA-" still matches its prefix.
    """
    pattern = re.escape(needle)
    if needle[:1].isalnum():
        pattern = r"\b" + pattern
    # Closed at the end only for a number, where a longer one is a different
    # number: "96" must not match "1996" or "965". A word is left open,
    # because its inflections are the same word and a `must_not` has to catch
    # them -- "consult" is a medical-advice guard, and the answer that trips
    # it says "consulting your physician".
    if needle[-1:].isdigit():
        pattern = pattern + r"\b"
    return re.search(pattern, haystack, re.IGNORECASE) is not None


# Retrying is decided by what actually happened upstream, not by what the
# answer says about it. `answer_from_live_services` records a tool exception
# on a `LiveReport` before stringifying it into the model's context, and the
# graph carries that out as `live_tool_failed` -- see T023 in spec 010.
#
# This used to match prose, and one of the strings was "could not find out",
# which is the wording `src/reactome_mcp/answer.py` *instructs* the model to
# use for a legitimate empty result. So the marker matched correct answers by
# construction, and since a match triggers a retry, a real regression got a
# second attempt and could pass. The gate was forgiving precisely what it
# exists to catch.
#
# The lesson generalises past this file: an exception that is caught, logged,
# stringified into a prompt and paraphrased has been through a lossy channel
# by design. Matching on the far end recovers nothing; the fix belongs where
# the information is discarded.


def _has_collection(name: str) -> bool:
    bundle = EmbeddingEnvironment.get_dir("reactome")
    return bool(bundle and (bundle / name / "chroma.sqlite3").exists())


async def run(expectations: tuple[Expectation, ...], retries: int = 1) -> list[Result]:
    graph = AgentGraph([ProfileName.React_to_Me])
    results: list[Result] = []
    live = is_configured()
    try:
        for index, expectation in enumerate(expectations, start=1):
            if expectation.needs_collection and not _has_collection(
                expectation.needs_collection
            ):
                results.append(
                    Result(
                        expectation=expectation,
                        skipped=f"the {expectation.needs_collection} collection "
                        "is not in the installed bundle",
                    )
                )
                continue
            if expectation.needs_live and not live:
                results.append(
                    Result(
                        expectation=expectation,
                        skipped="no MCP server configured ($REACTOME_MCP_URL)",
                    )
                )
                continue
            print(
                f"  [{index}/{len(expectations)}] {expectation.question[:60]}",
                file=sys.stderr,
            )
            retried = False
            for attempt in range(retries + 1):
                result = Result(expectation=expectation, retried=retried)
                started = time.monotonic()
                try:
                    out = await graph.ainvoke(
                        expectation.question,
                        "react-to-me",
                        callbacks=[],
                        # A fresh thread per attempt: these questions are
                        # independent, and a shared history would make each a
                        # follow-up of the last.
                        thread_id=f"sweep-{index}-{attempt}",
                        # No web search. The sweep reads `answer` and nothing
                        # else, so the postprocess node's Tavily call was paid
                        # for and thrown away -- fifteen of them per run, and
                        # this runs after every beta deploy. It also slowed each
                        # question by the length of a web search, for a result
                        # no expectation has ever looked at.
                        enable_postprocess=False,
                    )
                    result.answer = " ".join(str(out.get("answer") or "").split())
                    result.upstream_failed = bool(out.get("live_tool_failed"))
                except Exception as exc:
                    result.error = f"{type(exc).__name__}: {exc}"
                result.seconds = time.monotonic() - started

                if not result.error:
                    result.missing = [
                        t for t in expectation.must if not _contains(result.answer, t)
                    ] + [
                        f"/{p}/"
                        for p in expectation.must_match
                        if not re.search(p, result.answer, re.IGNORECASE)
                    ]
                    result.forbidden = [
                        t for t in expectation.must_not if _contains(result.answer, t)
                    ]

                # Retry an upstream hiccup only, and say so. A sweep that cries
                # wolf gets ignored; one that silently retries a real failure is
                # worse than no sweep at all. So this is narrow and it is
                # reported.
                if (
                    not result.ok
                    and attempt < retries
                    and (result.error or result.upstream_failed)
                ):
                    print(
                        "        upstream looked unwell; retrying once", file=sys.stderr
                    )
                    retried = True
                    continue

                results.append(result)
                break
    finally:
        await graph.close_pool()
    return results


def report(results: list[Result]) -> int:
    print()
    failures = [r for r in results if not r.ok]
    skipped = [r for r in results if r.skipped]
    for r in results:
        mark = "skip" if r.skipped else ("ok  " if r.ok else "FAIL")
        note = "  (retried once)" if r.retried else ""
        print(f"  {mark} {r.seconds:5.1f}s  {r.expectation.question[:58]}{note}")
        if r.skipped:
            print(f"        {r.skipped}")
            continue
        if r.error:
            print(f"        error: {r.error}")
        if r.missing:
            print(f"        missing: {', '.join(repr(m) for m in r.missing)}")
        if r.forbidden:
            print(
                f"        must not contain: {', '.join(repr(f) for f in r.forbidden)}"
            )
        if not r.ok:
            print(f"        why this is checked: {r.expectation.why}")
            print(f"        answered: {r.answer[:160]}")

    total = sum(r.seconds for r in results)
    ran = len(results) - len(skipped)
    print(f"\n  {ran - len(failures)}/{ran} passed, {total:.0f}s total")
    if skipped:
        # Grouped by reason: "they need the live service" was printed for
        # every skip, including one whose collection was simply not installed.
        reasons: dict[str, int] = {}
        for r in skipped:
            reasons[r.skipped] = reasons.get(r.skipped, 0) + 1
        print(f"  {len(skipped)} skipped, neither a pass nor a failure:")
        for reason, count in reasons.items():
            print(f"    {count}x {reason}")
        print("  The deploy runs this inside the container, where the bundle is")
        print("  installed and MCP is configured.")
    if failures:
        print(
            "  A failure here is a question the chatbot used to get wrong and does again."
        )
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        help="Run only questions whose text contains this substring.",
    )
    args = parser.parse_args()

    expectations = EXPECTATIONS
    if args.only:
        expectations = tuple(
            e for e in EXPECTATIONS if args.only.lower() in e.question.lower()
        )
        if not expectations:
            raise SystemExit(f"No tracked question matches {args.only!r}")

    print(
        f"Asking {len(expectations)} questions the chatbot has got wrong before\n",
        file=sys.stderr,
    )
    raise SystemExit(report(asyncio.run(run(expectations))))
