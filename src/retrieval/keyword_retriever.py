import pickle
import re
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from rank_bm25 import BM25Okapi

from src.data_loader import load_law_chunks


def tokenize_keyword_text(text: str) -> List[str]:
    """Simple tokenizer for BM25 keyword search."""
    if not text:
        return []

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text)

    return text.strip().split()


class KeywordIndex:
    """
    Keyword Retrieval using BM25.

    Offline:
        keyword_text -> tokenize -> BM25 index -> save to disk

    Online:
        user query -> query terms -> BM25 search -> ranked keyword docs
    """

    def __init__(self, bm25: BM25Okapi, metadata: pd.DataFrame):
        self.bm25 = bm25
        self.metadata = metadata.reset_index(drop=True)

    @classmethod
    def build(
        cls,
        data_path: str = "data/processed/laws.parquet",
        index_dir: str = "data/indexes/keyword",
    ) -> "KeywordIndex":
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        df = load_law_chunks(data_path)
        keyword_texts = df["keyword_text"].fillna("").astype(str).tolist()
        tokenized_corpus = [tokenize_keyword_text(text) for text in keyword_texts]

        print("Building BM25 keyword index...")
        bm25 = BM25Okapi(tokenized_corpus)

        keyword_index = cls(bm25=bm25, metadata=df)
        keyword_index.save(index_dir)

        print(f"Keyword index saved to: {index_dir.resolve()}")
        print(f"Total documents indexed: {len(df)}")

        return keyword_index

    @classmethod
    def load(cls, index_dir: str = "data/indexes/keyword") -> "KeywordIndex":
        index_dir = Path(index_dir)

        bm25_path = index_dir / "bm25.pkl"
        metadata_path = index_dir / "metadata.parquet"

        if not bm25_path.exists():
            raise FileNotFoundError(f"Missing BM25 index file: {bm25_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing keyword metadata file: {metadata_path}")

        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)

        metadata = pd.read_parquet(metadata_path)

        print(f"Keyword index loaded from: {index_dir.resolve()}")
        print(f"Total documents loaded: {len(metadata)}")

        return cls(bm25=bm25, metadata=metadata)

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        with open(index_dir / "bm25.pkl", "wb") as f:
            pickle.dump(self.bm25, f)

        self.metadata.to_parquet(index_dir / "metadata.parquet", index=False)

    def search(self, query: str, top_n: int = 10) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        query_terms = tokenize_keyword_text(query)
        scores = self.bm25.get_scores(query_terms)

        ranked_indices = scores.argsort()[::-1][:top_n]

        results: List[Dict[str, Any]] = []

        for rank, idx in enumerate(ranked_indices, start=1):
            score = float(scores[idx])

            if score <= 0:
                continue

            row = self.metadata.iloc[idx].to_dict()

            results.append({
                "chunk_id": row["chunk_id"],
                "rank": rank,
                "score": score,
                "bm25_score": score,
                "retriever": "keyword",
                "prompt_text": row["prompt_text"],
                "metadata": {
                    "law_name": row.get("law_name"),
                    "article": row.get("article"),
                    "clause": row.get("clause"),
                    "point": row.get("point"),
                    "article_title": row.get("article_title"),
                    "source_url": row.get("source_url"),
                },
            })

        return results


if __name__ == "__main__":
    KeywordIndex.build()