import os

from pypdf import PdfReader


def load_pdf(file_path, document_name=None):
    """
    Extract text from a PDF while preserving page information.

    document_name is the original uploaded filename.
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"PDF file not found: {file_path}"
        )

    reader = PdfReader(file_path)

    # Use original filename if provided
    if document_name is None:
        document_name = os.path.basename(file_path)

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text and text.strip():

            pages.append(
                {
                    "document": document_name,
                    "page": page_number,
                    "text": text.strip()
                }
            )

    print(f"Pages with text: {len(pages)}")

    return pages


def create_chunks(pages, chunk_size=800, chunk_overlap=150):
    """
    Split page text into overlapping chunks while
    preserving document and page metadata.
    """

    chunks = []

    for page in pages:

        text = page["text"]

        start = 0

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[start:end].strip()

            if chunk_text:

                chunks.append(
                    {
                        "document": page["document"],
                        "page": page["page"],
                        "text": chunk_text
                    }
                )

            if end >= len(text):
                break

            start = end - chunk_overlap

    print(f"Total chunks created: {len(chunks)}")

    return chunks