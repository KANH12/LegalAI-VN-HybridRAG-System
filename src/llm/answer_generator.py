from typing import Any, Dict

from src.retrieval.hybrid_retriever import HybridRetriever
from src.llm.prompt_builder import build_legal_qa_prompt
from src.llm.llm_client import GroqLLMClient


class LegalAnswerGenerator:
    """
    End-to-end QA:
        query
        -> hybrid retrieval
        -> prompt construction
        -> LLM inference
        -> final answer
    """

    def __init__(
        self,
        keyword_index_dir: str = "data/indexes/keyword",
        vector_index_dir: str = "data/indexes/vector",
    ):
        self.retriever = HybridRetriever(
            keyword_index_dir=keyword_index_dir,
            vector_index_dir=vector_index_dir,
        )
        self.llm = GroqLLMClient()

    def answer(
        self,
        query: str,
        keyword_top_n: int = 20,
        vector_top_n: int = 20,
        final_top_k: int = 8,
    ) -> Dict[str, Any]:
        top_chunks = self.retriever.search(
            query=query,
            keyword_top_n=keyword_top_n,
            vector_top_n=vector_top_n,
            final_top_k=final_top_k,
        )

        prompt = build_legal_qa_prompt(
            query=query,
            retrieved_chunks=top_chunks,
        )

        answer = self.llm.generate(prompt)

        return {
            "query": query,
            "answer": answer,
            "top_chunks": top_chunks,
            "prompt": prompt,
        }