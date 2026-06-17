import re
import math
import unicodedata
from typing import Any, Dict, Optional
from typing import List, Tuple, Optional

def normalize_unicode(text: str) -> str:
    """Normalize unicode to avoid errors with Vietnamese characters."""
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\ufeff", "")
    text = text.replace("\xa0", " ")
    return text


def clean_spaces(text: str) -> str:
    """Clean redundant spaces while maintaining line breaks."""
    if not text:
        return ""

    text = normalize_unicode(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


def clean_one_line(text: str) -> str:
    """Convert multi-line text into a single clean line."""
    if not text:
        return ""

    text = clean_spaces(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

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


def split_chapters(text: str) -> List[Tuple[str, str]]:
    """
    Split document by chapters.

    Returns:
        [(chapter_title, chapter_content), ...]
    """
    pattern = re.compile(
        r"(?im)^\s*(Chương\s+[IVXLC\d]+\.?\s*[^\n]*)\s*$"
    )
    matches = list(pattern.finditer(text))

    if not matches:
        return [("Không xác định", text)]

    chapters = []

    for idx, match in enumerate(matches):
        chapter_title = clean_one_line(match.group(1))
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        chapter_content = text[start:end].strip()

        if chapter_content:
            chapters.append((chapter_title, chapter_content))

    return chapters


def split_articles(chapter_content: str) -> List[Tuple[str, str, str]]:
    """
    Split chapter content into articles.

    Returns:
        [(article_number, article_title, article_content), ...]
    """
    pattern = re.compile(
        r"(?im)^\s*Điều\s+(\d+)\.\s*(.+?)\s*$"
    )
    matches = list(pattern.finditer(chapter_content))
    articles = []

    for idx, match in enumerate(matches):
        article_num = match.group(1).strip()
        article_title = clean_one_line(match.group(2))

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(chapter_content)

        article_content = chapter_content[start:end].strip()

        if article_content:
            articles.append((article_num, article_title, article_content))

    return articles


def split_clauses(article_content: str) -> List[Tuple[Optional[str], str]]:
    """
    Split article into clauses: 1., 2., 3., ...

    If article has no numbered clause, return article-level content.
    """
    content = clean_spaces(article_content)

    pattern = re.compile(r"(?m)^\s*(\d+)\.\s+")
    matches = list(pattern.finditer(content))

    if not matches:
        return [(None, content)]

    clauses = []

    for idx, match in enumerate(matches):
        clause_num = match.group(1).strip()

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(content)

        clause_text = content[start:end].strip()

        if clause_text:
            clauses.append((clause_num, clause_text))

    return clauses


def split_points(clause_text: str) -> List[Tuple[Optional[str], str]]:
    """
    Split clause into points: a), b), c), ...

    Important:
    - Do NOT create a separate chunk for intro text like:
      "Người lao động có các quyền sau đây:"
    - Instead, attach intro text to each point to keep context.
    """
    text = clean_spaces(clause_text)

    pattern = re.compile(r"(?m)^\s*([a-zA-ZđĐ])\)\s+")
    matches = list(pattern.finditer(text))

    if not matches:
        return [(None, text)]

    points = []
    intro_text = text[:matches[0].start()].strip()

    for idx, match in enumerate(matches):
        point_label = match.group(1).strip().lower()

        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)

        point_text = text[start:end].strip()

        if intro_text:
            point_text = f"{intro_text} {point_text}"

        point_text = clean_one_line(point_text)

        if point_text:
            points.append((point_label, point_text))

    return points


def parse_document_to_raw_rows(text: str) -> List[dict]:
    """
    Execute full top-down parsing pipeline from document text to dictionary rows.

    Output row:
        chapter, article, article_title, clause, point, content
    """
    text = normalize_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = clean_spaces(text)

    raw_rows = []

    for chapter_title, chapter_content in split_chapters(text):
        articles = split_articles(chapter_content)

        if not articles:
            continue

        for art_num, art_title, art_content in articles:
            for cls_num, cls_text in split_clauses(art_content):
                for p_label, p_text in split_points(cls_text):
                    content = clean_one_line(p_text)

                    if len(content) < 15:
                        continue

                    raw_rows.append({
                        "chapter": chapter_title,
                        "article": art_num,
                        "article_title": art_title,
                        "clause": cls_num,
                        "point": p_label,
                        "content": content,
                    })

    return raw_rows


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