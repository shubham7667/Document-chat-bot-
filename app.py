"""
Streamlit UI for the RAG pipeline defined in main.py

Wraps the same components in the same order:
DocumentLoader -> GenerateChunks -> EmbeddingModel -> VectorStore -> Retriever -> LLMIntegration

Run with:
    streamlit run streamlit_app.py
"""

import os
import time
import traceback

import streamlit as st

from load_document import DocumentLoader
from chunking import GenerateChunks
from embedding import EmbeddingModel
from vector_db import VectorStore
from retriever import Retriever
from llm import LLMIntegration


# --------------------------------------------------------------------------------
# Page config
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="DocChat — RAG over your PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------
# Minimal styling
# --------------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .subtitle {
        color: #9aa0a6;
        margin-bottom: 1.5rem;
    }
    .pipeline-step {
        padding: 0.5rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 0.4rem;
        font-size: 0.92rem;
        border-left: 3px solid #444;
    }
    .step-done { border-left-color: #2ecc71; background: rgba(46, 204, 113, 0.08); }
    .step-active { border-left-color: #f1c40f; background: rgba(241, 196, 15, 0.08); }
    .step-pending { border-left-color: #555; color: #888; }
    .context-chunk {
        background: #1a1d24;
        border: 1px solid #2a2d36;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.6rem;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# Session state init
# --------------------------------------------------------------------------------
defaults = {
    "document": None,
    "chunks": None,
    "embedding_model": None,
    "vector_db": None,
    "pipeline_ready": False,
    "doc_path": None,
    "chat_history": [],  # list of (query, answer, context_list)
    "last_config": None,
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val


# --------------------------------------------------------------------------------
# Sidebar — document upload + pipeline settings
# --------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Pipeline Settings")

    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    use_default = st.checkbox(
        "Use default sample (Data/NNDesign.pdf)",
        value=uploaded_file is None,
        disabled=uploaded_file is not None,
    )

    st.markdown("---")
    st.markdown("### Chunking")
    chunk_size = st.slider("Chunk size", min_value=100, max_value=2000, value=500, step=50)
    chunk_overlap = st.slider("Chunk overlap", min_value=0, max_value=500, value=100, step=10)

    st.markdown("### Embedding model")
    model_name = st.text_input(
        "Sentence-transformers model",
        value="sentence-transformers/all-MiniLM-L6-v2",
    )

    st.markdown("---")
    build_clicked = st.button("🚀 Build / Rebuild Knowledge Base", use_container_width=True)

    st.markdown("---")
    if st.session_state.pipeline_ready:
        st.success("Knowledge base ready ✅")
    else:
        st.info("Knowledge base not built yet")

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# --------------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------------
st.markdown('<div class="main-title">📄 DocChat — Chat with your PDF</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Document loading → chunking → embeddings → vector store → '
    'retrieval → LLM, all in one place.</div>',
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------------
# Helper: resolve the document path to use
# --------------------------------------------------------------------------------
def resolve_doc_path() -> str:
    if uploaded_file is not None:
        os.makedirs("Data", exist_ok=True)
        save_path = os.path.join("Data", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return save_path
    return "Data/NNDesign.pdf"


# --------------------------------------------------------------------------------
# Pipeline builder (mirrors main.py, steps 1-4)
# --------------------------------------------------------------------------------
def build_pipeline(doc_path: str, chunk_size: int, chunk_overlap: int, model_name: str):
    status_box = st.empty()
    progress = st.progress(0)

    def show(step_text, pct):
        status_box.markdown(
            f'<div class="pipeline-step step-active">⏳ {step_text}</div>',
            unsafe_allow_html=True,
        )
        progress.progress(pct)

    # 1. Load document
    show("Loading document...", 10)
    loader = DocumentLoader(doc_path)
    document = loader.load_docs()

    # 2. Generate chunks
    show("Splitting into chunks...", 35)
    get_chunk = GenerateChunks(document, chunk_size, chunk_overlap)
    chunks = get_chunk.chunking()

    # 3. Load embedding model
    show("Loading embedding model...", 60)
    embedding = EmbeddingModel(model_name)
    embedding_model = embedding.load_model()

    # 4. Build vector store
    show("Building vector database...", 85)
    database_manager = VectorStore(chunks, embedding_model)
    vector_db = database_manager.create_vector_db()

    show("Done!", 100)
    time.sleep(0.3)
    status_box.empty()
    progress.empty()

    return document, chunks, embedding_model, vector_db


# --------------------------------------------------------------------------------
# Trigger build
# --------------------------------------------------------------------------------
if build_clicked:
    try:
        doc_path = resolve_doc_path()
        if not os.path.exists(doc_path):
            st.error(f"File not found: `{doc_path}`. Upload a PDF or add it to the Data/ folder.")
        else:
            with st.spinner("Running pipeline — this can take a minute on first run..."):
                document, chunks, embedding_model, vector_db = build_pipeline(
                    doc_path, chunk_size, chunk_overlap, model_name
                )
            st.session_state.document = document
            st.session_state.chunks = chunks
            st.session_state.embedding_model = embedding_model
            st.session_state.vector_db = vector_db
            st.session_state.doc_path = doc_path
            st.session_state.pipeline_ready = True
            st.session_state.last_config = (doc_path, chunk_size, chunk_overlap, model_name)
            st.success(f"Knowledge base built from `{doc_path}` — {len(chunks)} chunks indexed.")
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        with st.expander("Show traceback"):
            st.code(traceback.format_exc())


# --------------------------------------------------------------------------------
# Pipeline status panel
# --------------------------------------------------------------------------------
with st.expander("📊 Pipeline status", expanded=not st.session_state.pipeline_ready):
    cols = st.columns(5)
    steps = [
        ("Load", st.session_state.document is not None),
        ("Chunk", st.session_state.chunks is not None),
        ("Embed", st.session_state.embedding_model is not None),
        ("Index", st.session_state.vector_db is not None),
        ("Ready", st.session_state.pipeline_ready),
    ]
    for col, (label, done) in zip(cols, steps):
        icon = "✅" if done else "⬜"
        col.markdown(f"**{icon} {label}**")

    if st.session_state.chunks is not None:
        st.caption(f"{len(st.session_state.chunks)} chunks indexed from `{st.session_state.doc_path}`")


# --------------------------------------------------------------------------------
# Chat / Query interface
# --------------------------------------------------------------------------------
st.markdown("## 💬 Ask a question")

if not st.session_state.pipeline_ready:
    st.warning("Build the knowledge base from the sidebar first.")
else:
    # Replay history
    for q, a, ctx in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(q)
        with st.chat_message("assistant"):
            st.write(a)
            if ctx:
                with st.expander("Retrieved context"):
                    for i, c in enumerate(ctx, 1):
                        st.markdown(
                            f'<div class="context-chunk"><b>Chunk {i}</b><br>{c}</div>',
                            unsafe_allow_html=True,
                        )

    query_input = st.chat_input("Enter your query...")

    if query_input:
        with st.chat_message("user"):
            st.write(query_input)

        with st.chat_message("assistant"):
            answer_placeholder = st.empty()
            context_chunks = []
            try:
                with st.spinner("Retrieving relevant context..."):
                    retriever = Retriever(st.session_state.embedding_model, query_input)
                    context = retriever.retrieve_document()

                # Normalize context for display (could be a string or a list of strings)
                if isinstance(context, (list, tuple)):
                    context_chunks = list(context)
                else:
                    context_chunks = [str(context)]

                with st.spinner("Generating answer..."):
                    llm = LLMIntegration(context, query_input)
                    response = llm.generate_response()

                answer_placeholder.write(response)

                if context_chunks:
                    with st.expander("Retrieved context"):
                        for i, c in enumerate(context_chunks, 1):
                            st.markdown(
                                f'<div class="context-chunk"><b>Chunk {i}</b><br>{c}</div>',
                                unsafe_allow_html=True,
                            )

                st.session_state.chat_history.append((query_input, response, context_chunks))

            except Exception as e:
                answer_placeholder.error(f"Failed to generate a response: {e}")
                with st.expander("Show traceback"):
                    st.code(traceback.format_exc())


# --------------------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------------------
st.markdown("---")
st.caption("RAG pipeline: DocumentLoader → GenerateChunks → EmbeddingModel → VectorStore → Retriever → LLMIntegration")