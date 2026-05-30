from typing import Any, Dict, List


def build_context_label(item: Dict[str, Any]) -> str:
    """
    Build readable label for a retrieved context.
    """
    metadata = item.get("metadata", {})

    article = metadata.get("article")
    clause = metadata.get("clause")
    point = metadata.get("point")
    title = metadata.get("article_title")

    label_parts = []

    if article:
        label_parts.append(f"Điều {article}")

    if clause:
        label_parts.append(f"Khoản {clause}")

    if point:
        label_parts.append(f"Điểm {point}")

    if title:
        label_parts.append(str(title))

    if label_parts:
        return " - ".join(label_parts)

    return item.get("chunk_id", "Unknown chunk")


def build_topk_overview_rows(top_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert Top-K chunks into table rows for Streamlit dataframe.
    """
    rows = []

    for item in top_chunks:
        metadata = item.get("metadata", {})

        rows.append(
            {
                "Rank": item.get("rank"),
                "RRF Score": round(float(item.get("rrf_score", 0)), 6),
                "Article": metadata.get("article"),
                "Clause": metadata.get("clause"),
                "Point": metadata.get("point"),
                "Title": metadata.get("article_title"),
                "Chunk ID": item.get("chunk_id"),
            }
        )

    return rows


def format_retrieval_source(source: Dict[str, Any]) -> str:
    """
    Format one retrieval source into markdown text.
    """
    retriever = source.get("retriever")
    rank = source.get("rank")
    score = source.get("score")
    bm25_score = source.get("bm25_score")
    similarity = source.get("similarity")

    lines = [
        f"- Retriever: `{retriever}`",
        f"  - Rank: `{rank}`",
    ]

    if score is not None:
        lines.append(f"  - Score: `{float(score):.6f}`")

    if bm25_score is not None:
        lines.append(f"  - BM25 Score: `{float(bm25_score):.6f}`")

    if similarity is not None:
        lines.append(f"  - Similarity: `{float(similarity):.6f}`")

    return "\n".join(lines)