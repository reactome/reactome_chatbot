# Contract: Intent Classifier Output

The classifier is the only interface this feature changes. One LLM call per question,
already made.

## Before routing (until 2026-09-19)

```json
{"source": "reactome"}
```

## With routing

```json
{"source": "reactome", "collections": ["disease_variants"]}
```

`collections` is optional and defaults to `[]`. Every consumer treats `[]` as "search
everything", so an older prompt, a model that omits the field, or a failed parse all
produce today's behaviour rather than a narrower search.

## Constraints

- Names must be collection directories present in the installed bundle. Validity is
  decided against `list_chroma_subdirectories()`, not a literal list
- Unknown names do not narrow the search. They are logged at WARNING and the whole
  selection falls back to all collections
- `collections` applies only when `source == "reactome"`. `userguide` has one
  collection; `live` does not use the vector store at all

## Prompt input

The per-collection descriptions in `src/retrievers/reactome/metadata_info.py`
(`reactome_descriptions_info`). They are a serving input: the classifier prompt
lists the selectable collections from them
(`src/agent/tasks/intent_classifier.py:55-67`), and `bin/routing-probe` checks
coverage against them. `bin/retrieval_baseline` also reads them. A collection with
no description is never offered to the classifier, and an inaccurate description
degrades routing.

## Compatibility

A deployment running an older prompt against newer code returns no `collections`,
gets `[]`, and searches everything. The feature degrades to the current system rather
than to a broken one.
