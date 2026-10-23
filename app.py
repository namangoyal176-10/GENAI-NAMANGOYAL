"""Streamlit Web Application: Gas Turbine Maintenance Assistant.

A Retrieval-Augmented Generation (RAG) system answering questions strictly
from GE GER-3620 Rev P — Heavy-Duty Gas Turbine Operating and Maintenance Considerations.
"""

import sys
import time
from pathlib import Path
import streamlit as st

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import DEFAULT_PDF_NAME, DOCUMENT_TITLE, PDF_PATH
from src.pdf_loader import PDFLoader, compute_pdf_fingerprint
from src.rag_pipeline import RAGPipeline, RAGResponse
from src.text_splitter import DocumentSplitter
from src.vector_store import VectorStore

# Configure page settings
st.set_page_config(
    page_title="Gas Turbine Maintenance Assistant",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling (Theme-Adaptive for Light & Dark Modes)
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        opacity: 0.85;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.35rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-success {
        background-color: #065F46;
        color: #ECFDF5;
    }
    .badge-warning {
        background-color: #854D0E;
        color: #FEFCE8;
    }
    .not-found-banner {
        background-color: rgba(220, 38, 38, 0.15);
        border: 2px solid #EF4444;
        padding: 1.25rem;
        border-radius: 0.5rem;
        font-size: 1.3rem;
        font-weight: 800;
        color: #F87171;
        margin: 1rem 0;
    }
    .answer-container {
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
        background: rgba(59, 130, 246, 0.08);
        border-radius: 0.375rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    """Initializes and caches the RAG pipeline."""
    return RAGPipeline()


pipeline = get_pipeline()
vector_store = pipeline.vector_store
index_meta = vector_store.get_metadata()
is_indexed = vector_store.is_indexed()

# ------------------------------------------------------------------------------
# Sidebar - Index Status & Controls
# ------------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ System Status")

    if is_indexed:
        st.markdown('<span class="status-badge badge-success">● Vector Store Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge badge-warning">⚠️ Index Not Built</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(f"**Knowledge Source:** `{DEFAULT_PDF_NAME}`")
    st.markdown(f"**Document Exists:** `{'Yes' if PDF_PATH.exists() else 'No'}`")

    if is_indexed and index_meta:
        st.markdown(f"**Pages Indexed:** `{index_meta.get('total_pages', 60)}`")
        st.markdown(f"**Chunks Stored:** `{index_meta.get('chunks_count', 'N/A')}`")
        st.markdown(f"**Fingerprint:** `{index_meta.get('fingerprint', '')[:12]}...`")

    st.markdown("---")
    st.subheader("Configuration")
    st.caption("Locally running embeddings (all-MiniLM-L6-v2) with strict documentary evidence verification.")

    if st.button("🔄 Reindex PDF Document", use_container_width=True):
        if not PDF_PATH.exists():
            st.error(f"Cannot reindex: {DEFAULT_PDF_NAME} is missing from data/.")
        else:
            with st.spinner("Extracting PDF, generating embeddings, and rebuilding index..."):
                loader = PDFLoader(PDF_PATH)
                pages = loader.load()
                splitter = DocumentSplitter()
                chunks = splitter.split_pages(pages, source_name=DEFAULT_PDF_NAME)
                fp = compute_pdf_fingerprint(PDF_PATH)
                vector_store.build_index(chunks, fingerprint=fp, source_name=DEFAULT_PDF_NAME, total_pages=len(pages), force=True)
                st.success("Indexing completed successfully!")
                st.rerun()

# ------------------------------------------------------------------------------
# Main Content
# ------------------------------------------------------------------------------
st.markdown('<div class="main-title">Gas Turbine Maintenance Assistant</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="sub-title">{DOCUMENT_TITLE}</div>',
    unsafe_allow_html=True,
)

# Display error banner if prerequisites are missing
setup_error = pipeline.check_setup()
if setup_error:
    st.error(setup_error)

st.markdown("### 📋 Sample Assignment Questions")
col1, col2, col3 = st.columns(3)
col4, col5, _ = st.columns(3)

# Handle session state for query input so button clicks update text
if "current_query" not in st.session_state:
    st.session_state.current_query = ""

with col1:
    if st.button("Q1: Three Inspection Types", use_container_width=True):
        st.session_state.current_query = "What are the three main types of inspection?"
with col2:
    if st.button("Q2: Maintenance Factor of Two", use_container_width=True):
        st.session_state.current_query = "What does a maintenance factor of two mean?"
with col3:
    if st.button("Q3: Purchase Price", use_container_width=True):
        st.session_state.current_query = "What is the purchase price of this gas turbine?"
with col4:
    if st.button("Q4: Titanium Melting Point", use_container_width=True):
        st.session_state.current_query = "What is the melting point of titanium?"
with col5:
    if st.button("Q5: Combustion Interval", use_container_width=True):
        st.session_state.current_query = "What is the recommended combustion-inspection interval in hours?"

user_query = st.text_input(
    "Enter your question about GER-3620-P:",
    value=st.session_state.current_query,
    placeholder="Ask anything grounded in GER-3620-P.pdf...",
    key="query_input",
)

ask_btn = st.button("🔍 Ask Question", type="primary")

active_query = user_query.strip() if user_query.strip() else st.session_state.current_query.strip()

if (ask_btn or active_query) and active_query:
    with st.spinner("Retrieving document passages and verifying evidence..."):
        t0 = time.time()
        response: RAGResponse = pipeline.query(active_query)
        elapsed = time.time() - t0

    st.markdown("---")
    st.subheader("Query Result")

    if not response.is_supported:
        st.markdown(
            f'<div class="not-found-banner">Not found</div>',
            unsafe_allow_html=True,
        )
        st.info(f"**Verification Reason:** {response.validation_reason}")
    else:
        st.markdown(
            f"""
            <div class="answer-container">
                <h3 style="margin-top:0;">Answer</h3>
                <p style="font-size:1.1rem; line-height:1.7; white-space: pre-line;">{response.answer}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.success(f"**Reference:** {response.reference}")

        with st.expander("📄 Supporting Evidence & Metadata", expanded=True):
            st.markdown(f"**Source Document:** `{DEFAULT_PDF_NAME}`")
            st.markdown(f"**PDF Page:** `{response.pdf_page}` | **Printed Page:** `{response.printed_page}`")
            st.markdown(f"**Section Heading:** `{response.section}`")
            st.markdown(f"**Retrieval Distance:** `{response.best_distance:.4f}` (Cosine Similarity: `{response.best_similarity:.4f}`)")
            st.markdown("**Evidence Excerpt:**")
            st.code(response.evidence, language="text")

    st.caption(f"⏱️ Retrieval and verification completed in {elapsed:.2f} seconds.")
