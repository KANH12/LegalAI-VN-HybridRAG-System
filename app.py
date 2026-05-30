import streamlit as st

from src.pipeline.query_engine import create_query_engine
from src.ui.formatters import (
    build_context_label,
    build_topk_overview_rows,
    format_retrieval_source,
)

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="LegalAI Labor Law Chatbot",
    page_icon="⚖️",
    layout="wide",
)


# ==========================================
# CACHE SYSTEM
# ==========================================
@st.cache_resource
def load_query_engine():
    """Load query engine once.

    Streamlit reruns the script frequently, so caching prevents repeated loading
    of BM25 index, vector index, embedding model, and LLM client.
    """
    return create_query_engine()


# ==========================================
# UI RENDERING HELPERS
# ==========================================
def render_retrieval_sources(item: dict) -> None:
    sources = item.get("sources", [])

    if not sources:
        st.write("Không có thông tin retrieval source.")
        return

    for source in sources:
        st.markdown(format_retrieval_source(source))


def render_topk_contexts(top_chunks: list[dict], show_retrieval_details: bool) -> None:
    st.subheader(f"📚 Top-{len(top_chunks)} Retrieved Contexts")

    if not top_chunks:
        st.info("Không có context nào được retrieve.")
        return

    st.markdown("### Top-K Overview")
    st.dataframe(
        build_topk_overview_rows(top_chunks),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Context Details")
    for item in top_chunks:
        label = build_context_label(item)

        with st.expander(
            f"Rank {item.get('rank')} | "
            f"RRF {float(item.get('rrf_score', 0)):.6f} | "
            f"{label}"
        ):
            st.markdown("**Chunk ID**")
            st.code(item.get("chunk_id", ""), language="text")

            st.markdown("**Retrieved Context**")
            st.write(item.get("prompt_text", ""))

            if show_retrieval_details:
                st.markdown("**Retrieval Sources**")
                render_retrieval_sources(item)


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.title("⚙️ Retrieval Settings")
st.sidebar.markdown("### Candidate Retrieval")

keyword_top_n = st.sidebar.slider(
    "Keyword Top-N",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
    help="Số lượng kết quả lấy từ BM25 keyword retrieval trước khi RRF fusion.",
)

vector_top_n = st.sidebar.slider(
    "Vector Top-N",
    min_value=5,
    max_value=50,
    value=20,
    step=5,
    help="Số lượng kết quả lấy từ semantic/vector retrieval trước khi RRF fusion.",
)

st.sidebar.markdown("### Final Context")

final_top_k = st.sidebar.slider(
    "Final Top-K Contexts",
    min_value=8,
    max_value=12,
    value=10,
    step=1,
    help="Số lượng chunks cuối cùng sau RRF được đưa vào prompt cho LLM.",
)

show_contexts = st.sidebar.checkbox(
    "Show Top-K contexts",
    value=True,
)

show_retrieval_details = st.sidebar.checkbox(
    "Show retrieval details",
    value=True,
)

show_prompt = st.sidebar.checkbox(
    "Show final prompt",
    value=False,
)


# ==========================================
# MAIN HEADER
# ==========================================
st.title("⚖️ LegalAI Labor Law Chatbot")
st.caption(
    "Hybrid Retrieval QA System: BM25 Keyword Search + Vector Search + RRF Fusion + LLM"
)

st.markdown(
    """
    Hệ thống hỏi đáp pháp luật lao động Việt Nam sử dụng pipeline:
    ```text
    User Question
    → Keyword Retrieval + Semantic Retrieval
    → RRF Fusion
    → Top-K Contexts
    → LLM Answer Generation
    ```
    """
)
st.divider()


# ==========================================
# QUERY INPUT
# ==========================================
example_questions = [
    "Người lao động được nghỉ phép năm khi nào?",
    "Người lao động làm chưa đủ 12 tháng thì nghỉ hằng năm tính như thế nào?",
    "Người sử dụng lao động có được sa thải người lao động đang nghỉ hằng năm không?",
    "Người lao động có những quyền gì?",
    "Người sử dụng lao động có những quyền gì?",
    "Cưỡng bức lao động là gì?",
    "Người lao động có được tạm ứng lương khi nghỉ hằng năm không?",
]

selected_example = st.selectbox(
    "Chọn câu hỏi mẫu hoặc tự nhập bên dưới:",
    options=[""] + example_questions,
)

query = st.text_area(
    "Nhập câu hỏi:",
    value=selected_example,
    height=110,
    placeholder="Ví dụ: Người lao động được nghỉ phép năm khi nào?",
)


# ==========================================
# EXECUTE QUERY
# ==========================================
if st.button("🔎 Trả lời", type="primary"):
    if not query.strip():
        st.warning("Vui lòng nhập câu hỏi.")
        st.stop()

    try:
        query_engine = load_query_engine()

        with st.spinner("Đang truy xuất tài liệu và sinh câu trả lời..."):
            result = query_engine.ask(
                query=query,
                keyword_top_n=keyword_top_n,
                vector_top_n=vector_top_n,
                final_top_k=final_top_k,
            )

        st.subheader("💬 Câu trả lời")
        st.write(result["answer"])
        st.divider()

        if show_contexts:
            render_topk_contexts(
                top_chunks=result.get("top_chunks", []),
                show_retrieval_details=show_retrieval_details,
            )

        if show_prompt:
            st.divider()
            st.subheader("🧾 Final Prompt Debug")
            st.code(result.get("prompt", ""), language="text")

    except Exception as exc:
        st.error(f"Lỗi khi chạy hệ thống: {exc}")