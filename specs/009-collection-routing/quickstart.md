# Quickstart: Validating Collection Routing

## Prerequisites

- An installed Release97 bundle (`./bin/embeddings_manager which`)
- `OPENAI_API_KEY`
- A reachable MCP server for the two live questions, or accept them as skipped

## 1. Capture the baseline BEFORE changing anything

Principle II: the comparison has to exist before the change.

```bash
./bin/retrieval_baseline capture --out before.json
```

## 2. Record the cost the feature exists to reduce

On a question that needs no variant data:

```bash
./bin/answer-sweep --only "CDK5"
```

Known baseline, 2026-09-17: 5 collections, 50 documents, 9,437 context tokens,
14.9s retrieval. Four collections was 40, 7,061 and 11.5s.

## 3. After the change

```bash
./bin/retrieval_baseline capture --out after.json
./bin/retrieval_baseline compare before.json after.json
```

The diff is **reported, not gated**: dropping documents is what this change does. It
exits non-zero when anything changed, which is information here rather than failure.

## 4. The gate

```bash
./bin/answer-sweep
```

Must stay green. Two questions fail if variant questions stop reaching
`disease_variants`; the species and release questions fail if `live` routing breaks.

**This is the pass/fail.** See spec.md "Clarifications" for why no numeric
document-loss threshold is set.

## 5. The part that does not exist yet

Four of five collections have **no** question that fails if routing stops searching
them. Adding those is task work before the routing change lands -- until then the
gate has a hole exactly where this feature could break things.

## 6. Verify through the path a user takes

Principle I. The served path is async; `retrieval_baseline` drives the sync one.

```bash
python -m pytest tests/retrievers/test_sync_async_equivalence.py
```

Both paths must filter by the same selection. A change that routes only the sync path
would pass every measurement above and serve unrouted results.
