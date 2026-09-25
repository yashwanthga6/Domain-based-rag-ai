import os
import tempfile

import streamlit as st

from document_loader import load_pdf, create_chunks
from vector_store import create_vector_store
from rag_pipeline import answer_question


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Domain RAG AI",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

/* ============================================================
   MAIN APPLICATION
   ============================================================ */

.stApp {
    background:
        radial-gradient(
            circle at 15% 10%,
            rgba(0, 229, 255, 0.08),
            transparent 30%
        ),
        radial-gradient(
            circle at 90% 20%,
            rgba(0, 120, 255, 0.07),
            transparent 30%
        ),
        #050911;
}


/* ============================================================
   MAIN CONTENT
   ============================================================ */

.main .block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* ============================================================
   SIDEBAR
   ============================================================ */

[data-testid="stSidebar"] {
    background: #070d16;
    border-right: 1px solid rgba(0, 229, 255, 0.12);
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #eafaff;
}

[data-testid="stSidebar"] p {
    color: #8199a6;
}


/* ============================================================
   HEADINGS
   ============================================================ */

h1 {
    color: #effcff !important;
    font-weight: 800 !important;
    letter-spacing: -1px;
}

h2 {
    color: #dffaff !important;
    font-weight: 750 !important;
}

h3 {
    color: #cceff7 !important;
}


/* ============================================================
   METRIC CARDS
   ============================================================ */

[data-testid="stMetric"] {
    background:
        linear-gradient(
            145deg,
            #0c1723,
            #08101a
        );

    border: 1px solid rgba(0, 229, 255, 0.13);

    border-radius: 14px;

    padding: 18px;

    transition: 0.2s ease;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(0, 229, 255, 0.35);
    transform: translateY(-2px);
}

[data-testid="stMetricLabel"] {
    color: #708996 !important;
    font-size: 11px !important;
}

[data-testid="stMetricValue"] {
    color: #eafcff !important;
}


/* ============================================================
   FILE UPLOADER
   ============================================================ */

[data-testid="stFileUploader"] {
    background:
        linear-gradient(
            145deg,
            rgba(9, 23, 34, 0.95),
            rgba(6, 13, 22, 0.95)
        );

    border: 1px dashed rgba(0, 229, 255, 0.35);

    border-radius: 15px;

    padding: 18px;

    margin-top: 10px;
}

[data-testid="stFileUploader"]:hover {
    border-color: #00e5ff;
}


/* ============================================================
   CHAT MESSAGES
   ============================================================ */

[data-testid="stChatMessage"] {
    background: rgba(9, 17, 27, 0.75);

    border: 1px solid rgba(0, 229, 255, 0.08);

    border-radius: 14px;

    margin-bottom: 12px;

    padding: 10px;
}


/* ============================================================
   CHAT INPUT
   ============================================================ */

[data-testid="stChatInput"] {
    border-color: rgba(0, 229, 255, 0.25);
}

[data-testid="stChatInput"] textarea {
    color: #eafaff;
}


/* ============================================================
   BUTTONS
   ============================================================ */

.stButton > button {
    border-radius: 9px;

    border: 1px solid rgba(0, 229, 255, 0.20);

    background: #0b1722;

    color: #dffaff;

    transition: 0.2s ease;
}

.stButton > button:hover {
    border-color: #00e5ff;

    background: rgba(0, 229, 255, 0.08);

    color: white;
}


/* ============================================================
   EXPANDERS
   ============================================================ */

[data-testid="stExpander"] {
    background: rgba(8, 16, 26, 0.7);

    border: 1px solid rgba(0, 229, 255, 0.10);

    border-radius: 12px;
}


/* ============================================================
   INFO / SUCCESS / ERROR
   ============================================================ */

[data-testid="stAlert"] {
    border-radius: 11px;
}


/* ============================================================
   DIVIDERS
   ============================================================ */

hr {
    border-color: rgba(0, 229, 255, 0.08);
}


/* ============================================================
   SIDEBAR STATUS
   ============================================================ */

.status-online {
    color: #00e5ff;
    font-weight: 700;
}

.status-text {
    color: #728995;
    font-size: 12px;
}


/* ============================================================
   FOOTER
   ============================================================ */

.footer-text {
    text-align: center;
    color: #425965;
    font-size: 10px;
    padding-top: 30px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_file" not in st.session_state:
    st.session_state.processed_file = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("◈ DOMAIN RAG AI")

    st.caption(
        "Retrieval-Augmented Document Intelligence"
    )

    st.divider()

    st.markdown("### SYSTEM STATUS")

    st.markdown(
        "● **RAG Pipeline**",
    )

    st.caption(
        "Retrieval and generation engine ready"
    )

    st.markdown(
        "● **Vector Database**",
    )

    st.caption(
        "FAISS similarity search"
    )

    st.markdown(
        "● **Embedding Model**",
    )

    st.caption(
        "Sentence Transformer"
    )

    st.divider()

    st.markdown("### CURRENT DOCUMENT")

    if st.session_state.document_name:

        st.success(
            f"Loaded: {st.session_state.document_name}"
        )

        st.caption(
            f"{st.session_state.page_count} pages · "
            f"{st.session_state.chunk_count} chunks"
        )

    else:

        st.info(
            "No document loaded"
        )

        st.caption(
            "Upload a PDF to begin"
        )

    st.divider()

    st.markdown("### PIPELINE")

    st.markdown(
        """
        **PDF**

        ↓

        **Text Extraction**

        ↓

        **Chunking**

        ↓

        **MiniLM Embeddings**

        ↓

        **FAISS Retrieval**

        ↓

        **Groq Generation**
        """
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.caption(
    "DOCUMENT INTELLIGENCE SYSTEM"
)

st.title(
    "Domain RAG AI"
)

st.write(
    "Upload a domain-specific document and ask questions "
    "using retrieval-augmented generation."
)

st.divider()


# ============================================================
# SYSTEM OVERVIEW
# ============================================================

st.subheader(
    "System Overview"
)

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        label="Knowledge Source",
        value="PDF",
        help="Domain-specific PDF documents",
    )

with col2:

    st.metric(
        label="Retrieval",
        value="FAISS",
        help="Vector similarity search",
    )

with col3:

    st.metric(
        label="Embeddings",
        value="MiniLM",
        help="Sentence Transformer embeddings",
    )

with col4:

    st.metric(
        label="Generation",
        value="Groq",
        help="Large language model generation",
    )


st.divider()


# ============================================================
# DOCUMENT CENTER
# ============================================================

st.subheader(
    "Document Center"
)

st.caption(
    "Upload a PDF to create a searchable knowledge base."
)

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    help="Upload a domain-specific PDF document.",
)


# ============================================================
# PROCESS PDF
# ============================================================

if uploaded_file is not None:

    file_identifier = (
        uploaded_file.name,
        uploaded_file.size,
    )

    if st.session_state.processed_file != file_identifier:

        with st.status(
            "Processing document...",
            expanded=True,
        ) as status:

            temp_pdf_path = None

            try:

                st.write(
                    "Saving uploaded document..."
                )

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    temp_pdf_path = temp_file.name

                st.write(
                    "Extracting PDF text..."
                )

                pages = load_pdf(
                    temp_pdf_path,
                    document_name=uploaded_file.name,
                )

                if not pages:

                    status.update(
                        label="Document processing failed",
                        state="error",
                    )

                    st.error(
                        "No readable text was found "
                        "in the uploaded PDF."
                    )

                else:

                    st.write(
                        f"Extracted {len(pages)} pages."
                    )

                    st.write(
                        "Creating document chunks..."
                    )

                    chunks = create_chunks(
                        pages
                    )

                    if not chunks:

                        status.update(
                            label="Chunk creation failed",
                            state="error",
                        )

                        st.error(
                            "No text chunks could be created."
                        )

                    else:

                        st.write(
                            f"Created {len(chunks)} chunks."
                        )

                        st.write(
                            "Building FAISS vector index..."
                        )

                        create_vector_store(
                            chunks
                        )

                        st.session_state.processed_file = (
                            file_identifier
                        )

                        st.session_state.document_name = (
                            uploaded_file.name
                        )

                        st.session_state.page_count = (
                            len(pages)
                        )

                        st.session_state.chunk_count = (
                            len(chunks)
                        )

                        st.session_state.messages = []

                        status.update(
                            label="Document ready",
                            state="complete",
                        )

                        st.success(
                            "Knowledge base created successfully."
                        )

            except Exception as e:

                status.update(
                    label="Processing failed",
                    state="error",
                )

                st.error(
                    f"Document processing failed: {e}"
                )

            finally:

                if (
                    temp_pdf_path
                    and os.path.exists(temp_pdf_path)
                ):

                    os.remove(
                        temp_pdf_path
                    )


# ============================================================
# ACTIVE DOCUMENT
# ============================================================

if st.session_state.document_name:

    st.divider()

    st.subheader(
        "Active Knowledge Base"
    )

    doc1, doc2, doc3 = st.columns(3)

    with doc1:

        st.metric(
            "Document",
            st.session_state.document_name,
        )

    with doc2:

        st.metric(
            "Pages",
            st.session_state.page_count,
        )

    with doc3:

        st.metric(
            "Chunks",
            st.session_state.chunk_count,
        )


# ============================================================
# CHAT AREA
# ============================================================

st.divider()

st.subheader(
    "Ask Your Document"
)

st.caption(
    "Ask questions about the information contained in "
    "the uploaded document."
)


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            with st.expander(
                "View source references"
            ):

                displayed_sources = set()

                for source in message["sources"]:

                    document = source.get(
                        "document",
                        "Unknown document",
                    )

                    page = source.get(
                        "page",
                        "?",
                    )

                    source_key = (
                        document,
                        page,
                    )

                    if source_key not in displayed_sources:

                        st.markdown(
                            f"**{document}** — "
                            f"Page **{page}**"
                        )

                        displayed_sources.add(
                            source_key
                        )


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.document_name:

    st.info(
        "Upload a PDF above to activate the document "
        "knowledge base."
    )


# ============================================================
# CHAT INPUT
# ============================================================

if st.session_state.document_name:

    question = st.chat_input(
        "Ask something about your document..."
    )

    if question:

        # --------------------------------------------
        # USER MESSAGE
        # --------------------------------------------

        with st.chat_message("user"):

            st.markdown(
                question
            )

        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        # --------------------------------------------
        # AI RESPONSE
        # --------------------------------------------

        with st.chat_message("assistant"):

            with st.spinner(
                "Searching the knowledge base..."
            ):

                try:

                    result = answer_question(
                        question
                    )

                    answer = result.get(
                        "answer",
                        "No answer was generated.",
                    )

                    sources = result.get(
                        "sources",
                        [],
                    )

                    st.markdown(
                        answer
                    )

                    if sources:

                        with st.expander(
                            "View source references"
                        ):

                            displayed_sources = set()

                            for source in sources:

                                document = source.get(
                                    "document",
                                    "Unknown document",
                                )

                                page = source.get(
                                    "page",
                                    "?",
                                )

                                source_key = (
                                    document,
                                    page,
                                )

                                if (
                                    source_key
                                    not in displayed_sources
                                ):

                                    st.markdown(
                                        f"**{document}** — "
                                        f"Page **{page}**"
                                    )

                                    displayed_sources.add(
                                        source_key
                                    )

                    else:

                        st.caption(
                            "No source references were returned."
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        }
                    )

                except Exception as e:

                    error_message = (
                        f"An error occurred: {e}"
                    )

                    st.error(
                        error_message
                    )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_message,
                            "sources": [],
                        }
                    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Domain RAG AI · Retrieval-Augmented Document Intelligence"
)