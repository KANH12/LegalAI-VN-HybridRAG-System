from typing import Any, Dict

from src.llm.answer_generator import LegalAnswerGenerator


class QueryEngine:
    """
    Main query processing engine.

    Flow:
        User Query
        -> Hybrid Retrieval
        -> Prompt Construction
        -> LLM Inference
        -> Structured Result
    """

    def __init__(
        self,
        keyword_index_dir: str = "data/indexes/keyword",
        vector_index_dir: str = "data/indexes/vector",
    ):
        self.answer_generator = LegalAnswerGenerator(
            keyword_index_dir=keyword_index_dir,
            vector_index_dir=vector_index_dir,
        )

    def ask(
        self,
        query: str,
        keyword_top_n: int = 20,
        vector_top_n: int = 20,
        final_top_k: int = 8,
    ) -> Dict[str, Any]:
        """
        Process one user query and return:
        - answer
        - top retrieved chunks
        - final prompt
        """
        if not query or not query.strip():
            raise ValueError("Query must not be empty.")

        result = self.answer_generator.answer(
            query=query.strip(),
            keyword_top_n=keyword_top_n,
            vector_top_n=vector_top_n,
            final_top_k=final_top_k,
        )

        return {
            "query": result.get("query", query.strip()),
            "answer": result.get("answer", ""),
            "top_chunks": result.get("top_chunks", []),
            "prompt": result.get("prompt", ""),
        }


def create_query_engine() -> QueryEngine:
    """
    Factory function used by UI or future API layer.
    """
    return QueryEngine()