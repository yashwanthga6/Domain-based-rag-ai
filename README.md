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
How It Works
1. PDF Upload

The user uploads a domain-specific PDF document through the Streamlit interface.

2. Text Extraction

The PDF content is extracted using pypdf.

3. Document Chunking

The extracted content is divided into smaller chunks so that relevant sections can be efficiently retrieved.

4. Embedding Generation

Each chunk is converted into a numerical vector using the Sentence Transformer model:

all-MiniLM-L6-v2
5. Vector Storage

The generated embeddings are stored in a FAISS vector index.

FAISS allows the system to efficiently perform similarity searches against the document embeddings.

6. Question Processing

When the user asks a question, the system searches the vector database for document chunks that are semantically related to the question.

7. Response Generation

The retrieved document context is provided to the Groq language model.

The model generates an answer based on the retrieved information.

8. Source References

The application displays the relevant document and page references associated with the retrieved information.

Technologies Used
Technology	Purpose
Python	Core programming language
Streamlit	Web application interface
PyPDF	PDF text extraction
Sentence Transformers	Semantic embeddings
FAISS	Vector similarity search
Groq	Large language model generation
LangChain Core	Prompt and LLM integration
LangChain Groq	Groq model integration
python-dotenv	Environment variable management
Project Structure
Domain-based-rag-ai/
│
├── app.py
│
├── document_loader.py
├── vector_store.py
├── rag_pipeline.py
├── prompt.py
├── check_models.py
│
├── requirements.txt
├── .env
├── .gitignore
│
├── documents/
│   └── sample.pdf
│
└── tests/
    ├── evaluate.py
    └── test_questions.csv
Installation
1. Clone the repository
git clone https://github.com/yashwanthga6/Domain-based-rag-ai.git
2. Open the project
cd Domain-based-rag-ai
3. Create a virtual environment

Windows:

python -m venv venv
4. Activate the virtual environment

Windows PowerShell:

.\venv\Scripts\Activate.ps1

If activation is blocked by PowerShell execution policy, use:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Then:

.\venv\Scripts\Activate.ps1
5. Install dependencies
python -m pip install -r requirements.txt
Environment Variables

Create a .env file in the project root:

GROQ_API_KEY=your_groq_api_key

Replace:

your_groq_api_key

with your actual Groq API key.

Do not upload .env to GitHub.

The .gitignore file should contain:

.env
venv/
.venv/
__pycache__/
*.pyc
.pytest_cache/
.streamlit/secrets.toml
Running the Application

Start the Streamlit application with:

python -m streamlit run app.py

The application will open in the browser.

Upload a PDF and wait for the knowledge base to be created.

You can then ask questions about the uploaded document.

Example Questions

For a document containing machine learning concepts, example questions include:

What is backpropagation?

What is a multilayer perceptron?

What are the two phases of training a multilayer perceptron?

What happens during the forward phase?

What happens during the backward phase?

What is gradient descent?

Why is backpropagation used in multilayer networks?

What are the basic features of a multilayer perceptron?

The system retrieves relevant document content before generating the answer.

RAG Evaluation

The project includes an evaluation script:

python tests/evaluate.py

The evaluation contains questions designed to test the retrieval and answer-generation pipeline.

Example:

========== RAG EVALUATION ==========

Total questions: 9

The evaluation also includes an out-of-domain question to verify that the system does not generate unsupported information when the answer cannot be found in the uploaded document.

RAG Pipeline

The core RAG workflow is:

Document
   ↓
PDF Text Extraction
   ↓
Text Chunking
   ↓
Semantic Embeddings
   ↓
FAISS Index
   ↓
User Query
   ↓
Query Embedding
   ↓
Similarity Search
   ↓
Relevant Context
   ↓
Groq LLM
   ↓
Grounded Answer
   ↓
Source Pages
Why RAG?

A traditional language model may generate an answer based on its previously learned information.

A RAG system first retrieves relevant information from a specific knowledge source and then uses that retrieved context to generate the response.

This makes the application useful for domain-specific document question answering.

Source Grounding

The system is designed to ground responses in the uploaded document.

If relevant information cannot be retrieved from the document, the system can indicate that the information was not found instead of relying on unrelated external knowledge.

Limitations
The quality of answers depends on the quality and structure of the uploaded PDF.
Scanned PDFs without extractable text may require OCR.
Very large documents may require additional optimization.
Embedding-model downloads may require an internet connection on the first run.
Groq API access requires a valid API key.
The system only has access to information available in the indexed documents.
Future Enhancements

Possible future improvements include:

Support for multiple documents
Document management interface
OCR support for scanned PDFs
Conversational memory
Improved chunking strategies
Hybrid keyword and vector search
Re-ranking retrieved documents
Document deletion and replacement
Authentication and user accounts
Advanced evaluation metrics
Support for additional document formats
License

This project is intended for educational and research purposes.

Repository

GitHub:

https://github.com/yashwanthga6/Domain-based-rag-ai


### Then save and push it

After saving `README.md`:

```powershell
git add README.md
git commit -m "Add project documentation"
git push
