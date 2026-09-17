# Phase 1 Data Model: Collection Routing

## Selection

The unit this feature adds: which collections a single question should search.

| Field | Type | Notes |
|---|---|---|
| `collections` | `list[str]` | Collection directory names, e.g. `["disease_variants", "summations"]`. Empty means "all", which is also the fallback for every failure |

Not a new class. It is a field on the classifier's existing structured output and a
key in `RunnableConfig["configurable"]`, because adding a type for it would mean
threading that type through three layers that do not otherwise know about each other.

### Validity

Decided against the **live bundle**, never a literal:

```python
valid = set(list_chroma_subdirectories(embeddings_directory))
```

Principle V: the same list that decides what is searched decides what is nameable, so
the two cannot drift. A collection added to a bundle is immediately selectable; one
removed cannot be selected.

### Resolution rules

In order. Every failure path widens the search, never narrows it.

| Input | Result | Why |
|---|---|---|
| Empty or absent | all collections | The default, and what any failure degrades to |
| All names valid | those collections | The feature |
| Some names unknown | **all** collections, WARNING naming the unknown ones | The prompt and bundle disagree; narrowing on a misunderstanding is worse than not routing. Principle IV |
| All names unknown | all collections, WARNING | As above |
| Names valid but none match the question well | those collections | Not detectable here. This is what the measurement is for, and what the sweep catches |

## QueryIntent (existing, extended)

```python
class QueryIntent(BaseModel):
    source: SourceName
    collections: list[str] = []   # new; empty means all
```

The default matters: a model that omits the field, an older prompt, or a parse
failure all produce `[]`, which searches everything. The feature cannot fail closed.

## Where it travels

```
intent_classifier  ──> QueryIntent.collections
        │
ReactToMeState["collections"]        (preprocess, beside active_sources)
        │
RunnableConfig["configurable"]["collections"]   (generate_answer)
        │
HybridRetriever.{retrieve,aretrieve}_documents  (filters collection_retrievers)
```

The retriever is constructed once at startup and reads the selection per call. See
research.md R1 for why it is not a constructor argument: `BM25Retriever` is built over
17,004 documents for `reactions` alone, which is startup work.
