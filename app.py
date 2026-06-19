import streamlit as st

from rag_utils import (
    load_embedding_model,
    process_uploaded_pdfs,
    retrieve_chunks,
    build_context,
    group_sources_for_display,
    fingerprints,
)
from llm import answer_with_rag


DEFAULT_EMBED_MODEL = "all-MiniLM-L6-v2"
DEFAULT_OLLAMA_MODEL = "gemma4"


def reset_index_state() -> None:
    for key in [
        "file_fingerprints",
        "chunk_records",
        "faiss_index",
        "built_index",
        "warnings",
    ]:
        if key in st.session_state:
            del st.session_state[key]


st.set_page_config(page_title="Multi-PDF RAG Q&A", layout="wide")
st.title("Multi-PDF RAG Q&A")
st.write("Upload one or more PDFs, then ask questions or compare them.")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top chunks to retrieve", min_value=3, max_value=15, value=8, step=1)
    chunk_size = st.slider("Chunk size (chars)", min_value=600, max_value=2000, value=1200, step=100)
    chunk_overlap = st.slider("Chunk overlap (chars)", min_value=50, max_value=400, value=200, step=50)
    ollama_model = st.text_input("Ollama model", value=DEFAULT_OLLAMA_MODEL)
    rebuild = st.button("Rebuild index")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if rebuild:
    reset_index_state()
    st.success("Index cleared.")

if not uploaded_files:
    st.info("Upload one or more PDF files to begin.")
    st.stop()

embed_model = load_embedding_model(DEFAULT_EMBED_MODEL)

current_fingerprints = fingerprints(uploaded_files)
needs_rebuild = (
    "built_index" not in st.session_state
    or st.session_state.get("file_fingerprints") != current_fingerprints
)

if needs_rebuild:
    with st.spinner("Extracting text, chunking, and building the index..."):
        chunk_records, faiss_index, warnings = process_uploaded_pdfs(
            uploaded_files=uploaded_files,
            embed_model=embed_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    for warning in warnings:
        st.warning(warning)

    if not chunk_records:
        st.error("No readable text could be extracted from the uploaded PDFs.")
        st.stop()

    st.session_state["file_fingerprints"] = current_fingerprints
    st.session_state["chunk_records"] = chunk_records
    st.session_state["faiss_index"] = faiss_index
    st.session_state["warnings"] = warnings
    st.session_state["built_index"] = True

chunk_records = st.session_state["chunk_records"]
faiss_index = st.session_state["faiss_index"]

c1, c2 = st.columns(2)
with c1:
    st.metric("PDFs uploaded", len(uploaded_files))
with c2:
    st.metric("Chunks indexed", len(chunk_records))

st.subheader("Document previews")
for uploaded_file in uploaded_files:
    with st.expander(uploaded_file.name):
        st.write("Uploaded and indexed.")

question = st.text_area(
    "Ask a question",
    placeholder="Example: Compare the methodologies and conclusions across these PDFs.",
    height=120,
)

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    with st.spinner("Retrieving relevant passages..."):
        retrieved = retrieve_chunks(
            query=question,
            embed_model=embed_model,
            index=faiss_index,
            chunk_records=chunk_records,
            top_k=top_k,
        )
        context = build_context(retrieved, max_chars=50000)

    if not context.strip():
        st.error("No relevant context was retrieved.")
        st.stop()

    with st.spinner("Generating answer..."):
        try:
            answer = answer_with_rag(
                question=question,
                context=context,
                model_name=ollama_model,
            )
        except Exception as e:
            st.error(f"Ollama call failed: {e}")
            st.info("Check that Ollama is running and the model is installed.")
            st.stop()

    st.subheader("Answer")
    st.write(answer)

    st.subheader("Retrieved sources")
    for src in group_sources_for_display(retrieved):
        with st.expander(
            f"#{src['rank']} — {src['file']} (page {src['page']}, distance {src['distance']})"
        ):
            st.write(src["excerpt"])
