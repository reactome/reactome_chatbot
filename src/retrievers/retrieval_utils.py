from collections import defaultdict
from typing import Any, Callable, Dict, List, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[Any]],
    final_k: int = 5,
    lambda_mult: float = 60.0,
    rrf_k: int | None = None,
    id_getter: Callable[[Any], str] = lambda doc: doc.metadata.get("stId")
    or doc.metadata.get("stable_id"),
) -> Tuple[List[Any], List[str], Dict[str, float]]:
    rrf_scores = defaultdict(float)
    doc_meta = {}

    for ranked in ranked_lists:
        considered = ranked[:rrf_k] if rrf_k else ranked
        for rank, doc in enumerate(considered):
            doc_id = id_getter(doc)
            if doc_id is not None:  # Skip documents without valid IDs
                rrf_scores[doc_id] += 1.0 / (lambda_mult + rank + 1)
                if doc_id not in doc_meta:
                    doc_meta[doc_id] = doc

    sorted_items = sorted(rrf_scores.items(), key=lambda x: (-x[1], x[0]))
    top_ids = [doc_id for doc_id, _ in sorted_items[:final_k]]
    top_docs = [doc_meta[doc_id] for doc_id in top_ids]
    return top_docs, top_ids, rrf_scores
