from typing import Any, Dict, List, Tuple


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    Input:
        ranked_lists:
            [
                keyword_results,
                vector_results
            ]

    Each result item should contain:
        - chunk_id
        - rank
        - prompt_text
        - metadata

    Formula:
        RRF_score(d) = Σ 1 / (k + rank(d))

    Output:
        Hybrid Ranked Documents
    """
    fused_scores: Dict[str, float] = {}
    doc_store: Dict[str, Dict[str, Any]] = {}
    rank_sources: Dict[str, List[Dict[str, Any]]] = {}

    for result_list in ranked_lists:
        for item in result_list:
            chunk_id = item.get("chunk_id")

            if not chunk_id:
                continue

            rank = item.get("rank")

            if rank is None:
                continue

            rrf_score = 1 / (k + int(rank))
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + rrf_score

            if chunk_id not in doc_store:
                doc_store[chunk_id] = item

            rank_sources.setdefault(chunk_id, []).append({
                "retriever": item.get("retriever"),
                "rank": item.get("rank"),
                "score": item.get("score"),
                "bm25_score": item.get("bm25_score"),
                "similarity": item.get("similarity"),
            })

    sorted_items: List[Tuple[str, float]] = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    fused_results: List[Dict[str, Any]] = []

    for final_rank, (chunk_id, score) in enumerate(sorted_items[:top_k], start=1):
        base_item = doc_store[chunk_id]

        fused_results.append({
            "chunk_id": chunk_id,
            "rank": final_rank,
            "rrf_score": score,
            "retriever": "hybrid_rrf",
            "prompt_text": base_item.get("prompt_text"),
            "metadata": base_item.get("metadata", {}),
            "sources": rank_sources.get(chunk_id, []),
        })

    return fused_results