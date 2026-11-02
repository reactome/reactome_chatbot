# Collection Routing: Data Model

Describes current behaviour, checked against the code 2026-10-21.

## Selection

The unit this feature adds: which collections a single question should search.

| Field | Type | Notes |
|---|---|---|
| `collections` | `list[str]` | Collection directory names, e.g. `["disease_variants", "summations"]`. Empty means "all", which is also the fallback for every failure |

Not a new class. It is a field on the classifier's structured output
(`QueryIntent.collections`), carried on the graph state and handed to the retriever
through a ContextVar (see "Where it travels"). A dedicated type would have to be
threaded through three layers that do not otherwise know about each other.

### Validity

Decided against the collections loaded from the installed bundle at startup
(`list_chroma_subdirectories` in `HybridRetriever.from_subdirectory`), never a
literal list; `resolve_collections(selected, available)` receives those names
(`src/retrievers/csv_chroma.py:320`).

Principle V: the same list that decides what is searched decides what is valid, so
the two cannot drift. However, the classifier is offered only the collections
described in `reactome_descriptions_info` (`src/agent/tasks/intent_classifier.py:55-67`),
so a new collection needs a description there before it can be chosen.

### Resolution rules

In order. Every failure path widens the search, never narrows it.

| Input | Result | Why |
|---|---|---|
| Empty or absent | all collections | The default, and what any failure degrades to |
| All names valid | those collections | The feature |
| Some names unknown | **all** collections, WARNING naming the unknown ones | The prompt and bundle disagree; narrowing on a misunderstanding is worse than not routing. Principle IV |
| All names unknown | all collections, WARNING | As above |
| Names valid but none match the question well | those collections | Not detectable here. This is what the measurement is for, and what the sweep catches |

## QueryIntent

```python
class QueryIntent(BaseModel):
    source: SourceName
    collections: list[str] = Field(default_factory=list, ...)  # empty means all
```

Ignored unless `source` is `reactome`.

The default matters: a model that omits the field, an older prompt, or a parse
failure all produce `[]`, which searches everything. The feature cannot fail closed.

## Where it travels

```
intent_classifier ──> QueryIntent.collections
        │
ReactToMeState["collections"]       (preprocess; react_to_me.py:187)
        │
selected_collections ContextVar     (set around rag.ainvoke in generate_answer,
        │                            only when source is "reactome"; reset in
        │                            finally; react_to_me.py:303-324)
HybridRetriever.{retrieve,aretrieve}_documents
        └─ resolve_collections(selected_collections.get(), collection_retrievers)
```

The retriever is constructed once at startup and reads the selection per call. See
research.md R1 and "The config route the data model describes does not exist" for
why it is not a constructor argument: `BM25Retriever` is built over
17,004 documents for `reactions` alone, which is startup work.
