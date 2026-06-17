from typing import List

import pandas as pd

from processing.chunking import (
    generate_keyword_text,
    generate_embedding_text,
    generate_prompt_text,
    safe_id_part,
    has_value,
)


def normalize_nullable(value):
    """Normalize pandas NaN/empty strings into None."""
    if pd.isna(value):
        return None

    if not has_value(value):
        return None

    return str(value).strip()


def create_law_key(row: pd.Series) -> str:
    """
    Create identity key for checking overlapping documents.

    Example:
        article_clause_point
    """
    article = normalize_nullable(row.get("article")) or "none"
    clause = normalize_nullable(row.get("clause")) or "none"
    point = normalize_nullable(row.get("point")) or "none"

    return f"{article}_{clause}_{point}".lower()


def determine_law_name(document_name: str) -> str:
    """Map raw file name to law name."""
    doc_lower = document_name.lower()

    if "vbhn_125" in doc_lower or "125_2025" in doc_lower:
        return "Văn bản hợp nhất 125/VBHN-VPCP 2025"

    if "bllđ" in doc_lower or "blld" in doc_lower or "bllaodong" in doc_lower:
        return "Bộ luật Lao động 2019"

    return f"Văn bản pháp luật {document_name}"


def determine_doc_type(document_name: str) -> str:
    """Classify document type for filtering and override logic."""
    doc_lower = document_name.lower()

    if "vbhn" in doc_lower:
        return "law_consolidated"

    if "bllđ" in doc_lower or "blld" in doc_lower:
        return "law_base"

    return "general"


def build_structured_chunk_id(row: pd.Series) -> str:
    """
    Generate clean, trace-friendly chunk ID.

    Example:
        vbhn_125_2025_art5_c1_pa
    """
    law_id = safe_id_part(row.get("law_id"))
    article = safe_id_part(row.get("article"))

    clause = normalize_nullable(row.get("clause"))
    point = normalize_nullable(row.get("point"))

    clause_part = f"c{safe_id_part(clause)}" if clause else "none"
    point_part = f"p{safe_id_part(point)}" if point else "none"

    return f"{law_id}_art{article}_{clause_part}_{point_part}"


def make_unique_chunk_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure chunk_id is unique.

    If duplicate IDs exist, add _002, _003...
    """
    seen = {}
    unique_ids = []

    for chunk_id in df["chunk_id"].tolist():
        count = seen.get(chunk_id, 0) + 1
        seen[chunk_id] = count

        if count == 1:
            unique_ids.append(chunk_id)
        else:
            unique_ids.append(f"{chunk_id}_{count:03d}")

    df["chunk_id"] = unique_ids
    return df


def remove_parent_intro_chunks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove parent chunks that only introduce points.

    Example:
        "Người lao động có các quyền sau đây:"
    These chunks are already attached to each point by parse.py.
    """
    if "text" not in df.columns:
        return df

    is_parent_intro = df["text"].fillna("").str.strip().str.endswith(":")
    has_no_point = df["point"].isna()

    return df[~(is_parent_intro & has_no_point)].copy()


def transform_and_override_data(all_data: List[dict]) -> pd.DataFrame:
    """
    Core transformation logic decoupled from I/O operations.

    Main jobs:
    - normalize null values
    - override base law chunks with consolidated law chunks when overlapping
    - generate keyword_text, embedding_text, prompt_text
    - ensure unique chunk_id
    """
    df = pd.DataFrame(all_data)

    if df.empty:
        raise ValueError("Error: input parsed data is empty.")

    if "content" not in df.columns:
        raise ValueError("Error: 'content' column is missing from parsed data.")

    # Normalize required text.
    df["content"] = df["content"].fillna("").astype(str).str.strip()
    df = df[df["content"].str.len() > 5].copy()

    # Normalize nullable structural columns.
    for col in ["clause", "point"]:
        if col in df.columns:
            df[col] = df[col].apply(normalize_nullable)
        else:
            df[col] = None

    # Fill important metadata if missing.
    for col in ["law_id", "law_name", "chapter", "article", "article_title", "source_url", "doc_type"]:
        if col not in df.columns:
            df[col] = ""

    # Create law key for override logic.
    df["law_key"] = df.apply(create_law_key, axis=1)

    # Prefer VBHN_125_2025 over base law when same article/clause/point exists.
    is_vbhn125 = df["law_id"].astype(str).str.contains("VBHN_125", case=False, na=False)

    if is_vbhn125.any():
        vbhn125_keys = set(df.loc[is_vbhn125, "law_key"].unique())
        df = df[~((~is_vbhn125) & (df["law_key"].isin(vbhn125_keys)))].copy()

    print("[RUNNING] Generating hybrid RAG text fields...")

    df["keyword_text"] = df.apply(generate_keyword_text, axis=1)
    df["embedding_text"] = df.apply(generate_embedding_text, axis=1)
    df["prompt_text"] = df.apply(generate_prompt_text, axis=1)

    # Rename content to text after generating text fields.
    df = df.rename(columns={"content": "text"})

    # Remove parent intro chunks after rename.
    df = remove_parent_intro_chunks(df)

    # Remove exact duplicate embedding content.
    df = df.drop_duplicates(subset=["embedding_text"]).copy()
    df = df.reset_index(drop=True)

    # Build and de-duplicate chunk IDs.
    df["chunk_id"] = df.apply(build_structured_chunk_id, axis=1)
    df = make_unique_chunk_ids(df)

    final_cols = [
        "chunk_id",
        "law_id",
        "law_name",
        "chapter",
        "article",
        "article_title",
        "clause",
        "point",
        "text",
        "keyword_text",
        "embedding_text",
        "prompt_text",
        "source_url",
        "doc_type",
    ]

    return df[[col for col in final_cols if col in df.columns]]