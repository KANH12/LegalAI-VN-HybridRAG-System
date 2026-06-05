from typing import Any, Dict, List


def build_legal_qa_prompt(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> str:
    """
    Build structured prompt for legal QA.

    Input:
        query: user question
        retrieved_chunks: Top-K chunks from hybrid retriever

    Output:
        structured prompt for LLM
    """
    context_blocks = []

    for idx, item in enumerate(retrieved_chunks, start=1):
        prompt_text = item.get("prompt_text", "")
        chunk_id = item.get("chunk_id", "")
        rrf_score = item.get("rrf_score", None)

        score_text = f" | RRF score: {rrf_score:.6f}" if rrf_score is not None else ""

        context_blocks.append(
            f"[Context {idx} | chunk_id: {chunk_id}{score_text}]\n{prompt_text}"
        )

    context_text = "\n\n".join(context_blocks)

    prompt = f"""
        Bạn là trợ lý hỏi đáp pháp luật lao động Việt Nam.

        Nhiệm vụ:
        - Chỉ trả lời dựa trên Context được cung cấp.
        - Trả lời trực tiếp câu hỏi trước, sau đó bổ sung ngắn gọn nếu cần.
        - Khi có thể, hãy nhắc đến Điều/Khoản/Điểm liên quan (ưu tiên các Điều/Khoản/Điểm nằm chung với nhau)
        - Tuyệt đối không tự ý đề xuất các luật chưa được xác thực!
        - Nếu Context không đủ thông tin, hãy nói rằng thông tin được cung cấp chưa đủ để kết luận.

        Context:
        {context_text}

        Câu hỏi:
        {query}

        Câu trả lời:
    """.strip()

    return prompt