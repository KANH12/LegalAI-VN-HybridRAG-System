import os
from pathlib import Path

import pandas as pd

from processing.chunking import parse_document_to_raw_rows
from processing.transform import (
    determine_law_name,
    determine_doc_type,
    transform_and_override_data,
)


def save_pipeline_outputs(df: pd.DataFrame, output_dir: str | Path) -> None:
    """Save processed data to Parquet and JSON."""
    if df is None or df.empty:
        print("Writer Warning: DataFrame is empty. Skipping file write operation.")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "laws.parquet"
    json_path = output_dir / "laws.json"

    print("Writer running: Saving structured data to storage...")

    df.to_parquet(parquet_path, index=False)
    df.to_json(json_path, orient="records", force_ascii=False, indent=2)

    print(f"Output saved to Parquet: {parquet_path.resolve()}")
    print(f"Output saved to JSON: {json_path.resolve()}")


def process_and_write_all() -> None:
    """Main ETL orchestrator for legal text processing."""
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    output_dir = base_dir / "data" / "processed"

    all_data = []

    if not raw_dir.exists():
        print(f"Error: raw directory does not exist: {raw_dir}")
        return

    # ===== 1. EXTRACT =====
    for file_path in raw_dir.rglob("*.txt"):
        document_name = file_path.stem
        print(f"--> Processing raw file: {document_name}")

        raw_text = file_path.read_text(encoding="utf-8")
        parsed_rows = parse_document_to_raw_rows(raw_text)

        law_name = determine_law_name(document_name)
        doc_type = determine_doc_type(document_name)

        for row in parsed_rows:
            row["law_id"] = document_name
            row["law_name"] = law_name
            row["doc_type"] = doc_type
            row["source_url"] = "https://thuvienphapluat.vn/"

        all_data.extend(parsed_rows)

    if not all_data:
        print("Error: No raw text data found to process.")
        return

    # ===== 2. TRANSFORM =====
    try:
        final_df = transform_and_override_data(all_data)
    except Exception as exc:
        print(f"[ERROR] Transformation failed: {exc}")
        return

    # ===== 3. LOAD =====
    save_pipeline_outputs(final_df, output_dir)

    print(f"[SUCCESS] PIPELINE COMPLETE: Successfully saved {len(final_df)} clean chunks to disk!")

    # Quick data-quality checks.
    print("\nData quality check:")
    print(f"- Total chunks: {len(final_df)}")
    
if __name__ == "__main__":
    process_and_write_all()