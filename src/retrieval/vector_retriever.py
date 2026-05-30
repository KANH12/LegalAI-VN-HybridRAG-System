from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.data_loader import load_law_chunks


class VectorIndex:
    """
    Semantic Retrieval using dense embeddings + cosine similarity.

    Offline:
        embedding_text -> embedding model -> document vectors -> save to disk

    Online:
        user query -> query vector -> cosine similarity/kNN -> ranked vector docs
    """

    def __init__(
        self,
        doc_embeddings: np.ndarray,
        metadata: pd.DataFrame,
        model_name: str,
    ):
        self.doc_embeddings = doc_embeddings
        self.metadata = metadata.reset_index(drop=True)
        self.model_name = model_name

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)

        if len(self.metadata) != len(self.doc_embeddings):
            raise ValueError(
                f"Metadata rows ({len(self.metadata)}) != embeddings rows ({len(self.doc_embeddings)})"
            )

    @classmethod
    def build(
        cls,
        data_path: str = "data/processed/laws.parquet",
        index_dir: str = "data/indexes/vector",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        batch_size: int = 32,
    ) -> "VectorIndex":
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        df = load_law_chunks(data_path)
        texts = df["embedding_text"].fillna("").astype(str).tolist()

        print(f"Loading embedding model: {model_name}")
        model = SentenceTransformer(model_name)

        print("Encoding document chunks...")
        doc_embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        vector_index = cls(
            doc_embeddings=doc_embeddings,
            metadata=df,
            model_name=model_name,
        )

        vector_index.save(index_dir)

        print(f"Vector index saved to: {index_dir.resolve()}")
        print(f"Vector shape: {doc_embeddings.shape}")

        return vector_index

    @classmethod
    def load(cls, index_dir: str = "data/indexes/vector") -> "VectorIndex":
        index_dir = Path(index_dir)

        embeddings_path = index_dir / "doc_embeddings.npy"
        metadata_path = index_dir / "metadata.parquet"
        model_name_path = index_dir / "model_name.txt"

        if not embeddings_path.exists():
            raise FileNotFoundError(f"Missing vector embeddings: {embeddings_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing vector metadata: {metadata_path}")

        if not model_name_path.exists():
            raise FileNotFoundError(f"Missing model name file: {model_name_path}")

        doc_embeddings = np.load(embeddings_path)
        metadata = pd.read_parquet(metadata_path)
        model_name = model_name_path.read_text(encoding="utf-8").strip()

        print(f"Vector index loaded from: {index_dir.resolve()}")
        print(f"Vector shape: {doc_embeddings.shape}")

        return cls(
            doc_embeddings=doc_embeddings,
            metadata=metadata,
            model_name=model_name,
        )

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        np.save(index_dir / "doc_embeddings.npy", self.doc_embeddings)
        self.metadata.to_parquet(index_dir / "metadata.parquet", index=False)
        (index_dir / "model_name.txt").write_text(self.model_name, encoding="utf-8")

    def search(self, query: str, top_n: int = 10) -> List[Dict[str, Any]]:
        if not query or not query.strip():
            return []

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]

        # Since embeddings are normalized, dot product = cosine similarity
        similarities = np.dot(self.doc_embeddings, query_embedding)

        ranked_indices = np.argsort(similarities)[::-1][:top_n]

        results: List[Dict[str, Any]] = []

        for rank, idx in enumerate(ranked_indices, start=1):
            similarity = float(similarities[idx])
            row = self.metadata.iloc[idx].to_dict()

            results.append({
                "chunk_id": row["chunk_id"],
                "rank": rank,
                "score": similarity,
                "similarity": similarity,
                "retriever": "vector",
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
    VectorIndex.build()