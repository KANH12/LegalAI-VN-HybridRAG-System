from typing import Any, Dict, List

from src.retrieval.keyword_retriever import KeywordIndex
from src.retrieval.vector_retriever import VectorIndex
from src.retrieval.rrf_fusion import reciprocal_rank_fusion


class HybridRetriever:
    """
    Hybrid Retrieval pipeline.

    Online flow:
        User Query
        ↓
        Keyword Search / BM25
        ↓
        Vector Search / Cosine Similarity
        ↓
        RRF Fusion
        ↓
        Top-K Docs/Chunks
    """

    def __init__(
        self,
        keyword_index_dir: str = "data/indexes/keyword",
        vector_index_dir: str = "data/indexes/vector",
    ):
        print("Loading keyword index...")
        self.keyword_index = KeywordIndex.load(keyword_index_dir)

        print("Loading vector index...")
        self.vector_index = VectorIndex.load(vector_index_dir)

    def search(
        self,
        query: str,
        keyword_top_n: int = 20,
        vector_top_n: int = 20,
        final_top_k: int = 5,
        rrf_k: int = 60,
    ) -> List[Dict[str, Any]]:
        """
        Search using keyword retrieval and vector retrieval,
        then fuse both ranked lists using RRF.
        """
        keyword_results = self.keyword_index.search(
            query=query,
            top_n=keyword_top_n,
        )

        vector_results = self.vector_index.search(
            query=query,
            top_n=vector_top_n,
        )

        hybrid_results = reciprocal_rank_fusion(
            ranked_lists=[keyword_results, vector_results],
            k=rrf_k,
            top_k=final_top_k,
        )

        return hybrid_results