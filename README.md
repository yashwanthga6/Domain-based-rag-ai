# Domain-Based RAG AI

A document question-answering system that uses Retrieval-Augmented Generation (RAG) to provide answers grounded in information contained in uploaded PDF documents.

The system extracts text from a PDF, divides it into meaningful chunks, converts the chunks into semantic embeddings, stores them in a FAISS vector database, retrieves relevant information for a user query, and generates a contextual answer using a Groq-hosted language model.

---

## Features

- Upload domain-specific PDF documents
- Extract text from PDF files
- Split documents into searchable chunks
- Generate semantic embeddings using Sentence Transformers
- Store and search document embeddings using FAISS
- Retrieve relevant document content for user questions
- Generate context-aware answers using Groq
- Display source document and page references
- Interactive Streamlit web interface
- Document processing status and knowledge-base information
- Evaluation script for testing RAG responses

---

## System Architecture

```text
                  ┌──────────────────┐
                  │    PDF Upload    │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Text Extraction │
                  │      PyPDF       │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │    Chunking      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │   Embeddings     │
                  │    MiniLM        │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  FAISS Vector   │
                  │     Store       │
                  └────────┬─────────┘
                           │
                    User Question
                           │
                           ▼
                  ┌──────────────────┐
                  │ Similarity Search│
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │   Groq LLM       │
                  │   Generation      │
                  └────────┬─────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │ Answer + Sources │
                  └──────────────────┘
