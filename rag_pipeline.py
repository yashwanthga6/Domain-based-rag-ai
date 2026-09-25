import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from vector_store import search_documents
from prompt import SYSTEM_PROMPT


# Load environment variables
load_dotenv()


def create_llm():
    """
    Create the Groq language model.
    """

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is not configured. "
            "Please add it to the .env file."
        )

    return ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=api_key
    )


def build_context(results):
    """
    Convert retrieved chunks into context for the LLM.
    """

    context_parts = []

    for result in results:

        context_parts.append(
            f"Document: {result['document']}\n"
            f"Page: {result['page']}\n"
            f"Content:\n{result['text']}"
        )

    return "\n\n".join(context_parts)


def get_unique_sources(results):
    """
    Return unique document-page sources.
    """

    sources = []
    seen = set()

    for result in results:

        source_key = (
            result["document"],
            result["page"]
        )

        if source_key not in seen:

            sources.append(
                {
                    "document": result["document"],
                    "page": result["page"],
                    "score": result["score"]
                }
            )

            seen.add(source_key)

    return sources


def answer_question(question):
    """
    Retrieve relevant chunks and generate a grounded answer.
    """

    if not question or not question.strip():

        return {
            "answer": "Please enter a question.",
            "sources": []
        }

    # Retrieve relevant chunks
    results = search_documents(
        question,
        top_k=5
    )

    if not results:

        return {
            "answer": (
                "I could not find this information "
                "in the uploaded documents."
            ),
            "sources": []
        }

    # --------------------------------------------------
    # Filter weak retrieval results
    # --------------------------------------------------

    # FAISS uses cosine similarity because the embeddings
    # are normalized and IndexFlatIP is being used.
    MIN_SIMILARITY = 0.45

    filtered_results = [
        result
        for result in results
        if result["score"] >= MIN_SIMILARITY
    ]

    # If no result passes the threshold, treat the
    # information as unavailable in the document.
    if not filtered_results:

        return {
            "answer": (
                "I could not find this information "
                "in the uploaded documents."
            ),
            "sources": []
        }

    # --------------------------------------------------
    # Build context
    # --------------------------------------------------

    context = build_context(filtered_results)

    # --------------------------------------------------
    # Build prompt
    # --------------------------------------------------

    prompt = SYSTEM_PROMPT.format(
        context=context
    )

    prompt += f"""

User Question:
{question}

Answer:
"""

    # --------------------------------------------------
    # Generate answer
    # --------------------------------------------------

    llm = create_llm()

    response = llm.invoke(prompt)

    # --------------------------------------------------
    # Prepare unique sources
    # --------------------------------------------------

    sources = get_unique_sources(
        filtered_results
    )

    return {
        "answer": response.content,
        "sources": sources
    }