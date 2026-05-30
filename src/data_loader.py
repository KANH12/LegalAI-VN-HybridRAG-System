from pathlib import Path
import pandas as pd


def load_law_chunks(path: str | Path = "data/processed/laws.parquet") -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Processed data file not found: {path}")

    df = pd.read_parquet(path)

    required_cols = [
        "chunk_id",
        "keyword_text",
        "embedding_text",
        "prompt_text",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.dropna(subset=["chunk_id", "keyword_text", "embedding_text", "prompt_text"])
    df = df.reset_index(drop=True)

    return df