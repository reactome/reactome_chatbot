# Contract: Intent Classifier Output

The classifier is the only interface this feature changes. One LLM call per question,
already made.

## Before

```json
{"source": "reactome"}
```

## After

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

The per-collection descriptions already written in
`src/retrievers/reactome/metadata_info.py` (`reactome_descriptions_info`), which
exist for routing and are currently read only by `bin/retrieval_baseline`. They
become a serving input, so they must stay accurate as collections are added --
already true of `disease_variants`, described when it was added.

## Compatibility

A deployment running an older prompt against newer code returns no `collections`,
gets `[]`, and searches everything. The feature degrades to the current system rather
than to a broken one.
