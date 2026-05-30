import re
import math
from typing import Any, Dict, Optional

from processing.parse import clean_one_line


def has_value(value: Any) -> bool:
    """Return True if value is meaningful, not None/NaN/empty."""
    if value is None:
        return False

    if isinstance(value, float) and math.isnan(value):
        return False

    text = str(value).strip()

    if not text:
        return False

    if text.lower() in {"nan", "none", "null"}:
        return False

    return True


def safe_id_part(value: Optional[str]) -> str:
    """Create a safe ID component."""
    if not has_value(value):
        return "none"

    value = str(value).strip().lower()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[^\w\-]+", "", value, flags=re.UNICODE)

    return value or "none"


def get_content(row: Dict[str, Any]) -> str:
    """Safely retrieve content from row whether named 'content' or 'text'."""
    content = row.get("content")

    if has_value(content):
        return str(content)

    text = row.get("text")

    if has_value(text):
        return str(text)

    return ""


def generate_keyword_text(row: Dict[str, Any]) -> str:
    """
    Generate optimized text for BM25 / keyword retrieval.

    This text is allowed to include metadata because BM25 benefits from
    exact lexical cues such as law name, article, clause, and point.
    """
    parts = [
        row.get("law_name", ""),
        f"Điều {row.get('article', '')}" if has_value(row.get("article")) else "",
        row.get("article_title", ""),
    ]

    if has_value(row.get("clause")):
        parts.append(f"Khoản {row.get('clause')}")

    if has_value(row.get("point")):
        parts.append(f"Điểm {row.get('point')}")

    parts.append(get_content(row))

    combined = " ".join(str(p) for p in parts if has_value(p))
    combined = combined.lower()

    # Keep Vietnamese characters, letters, digits, and underscores.
    combined = re.sub(r"[^\w\s]", " ", combined, flags=re.UNICODE)
    combined = re.sub(r"\s+", " ", combined)

    return combined.strip()


def generate_embedding_text(row: Dict[str, Any]) -> str:
    """
    Generate concise text for dense embedding / semantic retrieval.

    Keep natural content and only compact citation metadata.
    """
    article = row.get("article")
    clause = row.get("clause")
    point = row.get("point")
    content = get_content(row)

    prefix_parts = []

    if has_value(article):
        prefix_parts.append(f"Điều {article}")

    if has_value(clause):
        prefix_parts.append(f"K{clause}")

    if has_value(point):
        prefix_parts.append(f"Đ{point}")

    prefix = ".".join(prefix_parts)

    if prefix:
        return clean_one_line(f"{prefix}: {content}")

    return clean_one_line(content)


def generate_prompt_text(row: Dict[str, Any]) -> str:
    """
    Create concise citation text for LLM prompt contexts.
    """
    article = row.get("article")
    title = row.get("article_title")
    clause = row.get("clause")
    point = row.get("point")
    content = get_content(row)

    ref_parts = []

    if has_value(article):
        ref_parts.append(f"Điều {article}")

    if has_value(clause):
        ref_parts.append(f"Khoản {clause}")

    if has_value(point):
        ref_parts.append(f"Điểm {point}")

    if has_value(title):
        ref_parts.append(str(title))

    ref = " - ".join(ref_parts)

    if ref:
        return clean_one_line(f"[{ref}] {content}")

    return clean_one_line(content)