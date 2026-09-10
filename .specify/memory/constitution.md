<!--
Sync Impact Report
==================
Version: 1.0.0 -> 1.1.0  (MINOR: one principle added; none removed or redefined)

Added:    Principle VII, "Parked is not dead"
Modified: none
Removed:  none

Rationale: capability that is built, works and is deliberately undeployed had no
stated policy, so it was read as dead code and proposed for deletion. Both failure
modes are now named -- deleting it to cheapen a refactor, and investing in it while
parked.

Follow-up TODOs: none.
-->

# Reactome ChatBot Constitution

The team is three developers covering a large surface — website, pathway browser,
content service, analysis service, and this chatbot. These principles exist to
protect quality without adding ceremony that a team this size cannot afford.

## Core Principles

### I. Verify the path a user takes, not the component you changed

A component test passing is not evidence the feature works. Retrieval was changed
three times before anyone asked the pipeline a question and read the answer; a
README fix documented commands that ran individually but produced a broken app
when followed in order; the async path that production actually uses went
unexercised while the sync path was measured.

Before claiming something works, run it the way a user or the server would.

### II. Measure retrieval changes; do not argue about them

Retrieval quality has no right answer, only "did this change". Any change to what
reaches the LLM — document counts, fusion, ranking, models — must come with a
before-and-after measurement on real questions against a real bundle.
`bin/retrieval_baseline` exists for this. "This should be better" is not a
finding.

Corollary: report the measurement you actually took. `len(text) // 4` is not a
token count when `tiktoken` is installed.

### III. Characterization tests pin behaviour, including behaviour that is wrong

Where current behaviour looks like a bug, the test asserts the current behaviour
and carries a `BUG:` comment. Changing such a test and its code together is a
deliberate act; changing the code alone should fail.

This is why the suite is not a specification. It is a tripwire.

### IV. Fail loudly, never quietly differently

Configuration that cannot be honoured must stop the process, not substitute
something plausible. An invalid `config.yml` used to disable rate limiting, then
briefly fell back to defaults — which silently re-enabled a feature an operator
had turned off. Both are worse than refusing to start.

The same applies to models and bundles: a query embedded with a different model
than built the vectors returns nonsense rather than an error, so that mismatch is
reported explicitly.

### V. Derive from the source of truth; do not synchronise constants by hand

The embeddings bundle records which model built it, so the model is read from the
bundle rather than defaulted in code — that is what keeps the Reactome and Plant
Reactome deployments both correct without hardcoding either. Prefer deriving a
value over duplicating it where a durable source exists.

### VI. Bias to doing over filing

Issues are cheap to create and expensive to service. An issue is worth filing when
it records a decision, a measurement, or a defect someone else must judge — not
as a substitute for a fix that takes ten minutes. Three fixes are worth more than
five issues describing them.

### VII. Parked is not dead

Some capability here is built, works, and is deliberately not deployed: UniProt
integration, Alliance results, and the Cross-Database profile. It runs only when a
`config.yml` names it; every default is React-to-Me.

Do not delete parked work to make a refactor cheaper. That trade looks like a saving
and is not: the code cost someone real effort, "we are not working on it now" is not
"it is dead", and rebuilding is far more expensive than carrying. This principle
exists because the proposal was made -- to remove about 500 lines of UniProt and
Cross-Database serving code, on the grounds that it had been dragged through three
refactors in a week -- and it was wrong.

Equally, do not invest in parked work. It needs no new features and no new tests.
Keeping it importable and type-checking through a refactor is the whole obligation.

Parked code is unexercised, so its behaviour is unverified even where it still type
checks. Whoever un-parks it inherits that, and should be told at the point they find
it rather than after.

## Quality Gates

`main` is protected: pull request required, `enforce_admins` on, branch must be
current, and four checks must pass — `lint`, `test`, `docker-build`,
`poetry-check (ubuntu-latest)`. No approving review is required, because the
gates enforce quality and the team is too small for a mandatory second reviewer.

One standard applies to every Python file in the repository: ruff (lint, import
order, format) and mypy with `disallow_untyped_defs`. Rules that are not yet
enforced are listed in `pyproject.toml` **with the reason**, never dropped
silently. Each mypy baseline entry carries a `TODO` naming its root cause.

## Working With Contributed Code

Many open pull requests come from GSoC applicants who were not selected and will
not update their work. The mindset is to harvest what is valuable and close
courteously with credit — never "please rebase". Where a contributor's diagnosis
is right but their patch does not fit current `main`, reimplement it and credit
them in the commit message.

Two independent reports of the same defect is strong evidence it is real. Verify
it anyway.

## Deployment

Production and beta both pin an image tag; neither follows `latest`. A change is
not deployed by merging it. `deploy/beta/README.md` records how beta runs and
what is deliberately absent from it.

## Governance

This constitution supersedes convenience. Amendments are made by editing this
file in a pull request that says what changed and why.

Spec Kit is used for design work with real, unmade decisions — the retriever
rewrite, the agent API, website integration, analysis summarisation. It is not
used for bug triage or dependency bumps, where the ceremony costs more than the
fix. Retrofitting specifications onto existing code is archaeology and is not
done.

**Version**: 1.1.0 | **Ratified**: 2026-09-08 | **Last Amended**: 2026-09-10
