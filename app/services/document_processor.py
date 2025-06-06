import logging
from typing import List, Tuple
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def load_and_split_pdf(
        file_path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
) -> Tuple[List[str], int]:
    """
    Loads a PDF, extracts text, and splits it into chunks.

    Args:
        file_path (str): The path to the PDF file.
        chunk_size (int): The desired size of text chunks.
        chunk_overlap (int): The overlap between consecutive chunks.

    Returns:
        Tuple[List[str], int]: A list of text chunks and the number of pages.
                               Returns ([], 0) if an error occurs.
    """
    try:
        logger.info(f"Loading PDF from: {file_path}")
        reader = PdfReader(file_path)
        num_pages = len(reader.pages)
        logger.info(f"PDF has {num_pages} pages.")

        if num_pages == 0:
            logger.warning(f"No pages found in PDF: {file_path}")
            return [], 0

        # Limit to 1000 pages as per requirement
        if num_pages > 1000:
            logger.warning(f"PDF has {num_pages} pages, processing only the first 1000 pages.")

        raw_text = ""
        for i, page in enumerate(reader.pages):
            if i >= 1000:  # Enforce 1000 page limit
                break
            text = page.extract_text()
            if text:
                raw_text += text + "\n"  # Add newline between pages for better context splitting

        if not raw_text.strip():
            logger.warning(f"No text extracted from PDF: {file_path}")
            return [], num_pages  # Return num_pages even if no text

        logger.info(f"Successfully extracted text from {min(num_pages, 1000)} pages.")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            add_start_index=False,
        )
        chunks = text_splitter.split_text(raw_text)

        # Filter out empty or very short chunks that might result from splitting
        chunks = [chunk for chunk in chunks if len(chunk.strip()) > 10]  # Min 10 chars

        logger.info(f"Split text into {len(chunks)} chunks.")
        return chunks, min(num_pages, 1000)

    except FileNotFoundError:
        logger.error(f"PDF file not found: {file_path}")
        return [], 0
    except Exception as e:
        logger.error(f"Error processing PDF {file_path}: {e}", exc_info=True)
        return [], 0


# Example Usage (for testing this module directly)
if __name__ == "__main__":
    # Create a dummy PDF file for testing
    from reportlab.pdfgen import canvas

    dummy_pdf_path = "dummy_test.pdf"
    c = canvas.Canvas(dummy_pdf_path)
    for i in range(5):  # 5 pages
        c.drawString(100, 750 - i * 20, f"This is page {i + 1}. " + "Lorem ipsum dolor sit amet. " * 50)
        c.showPage()
    c.save()

    chunks, page_count = load_and_split_pdf(dummy_pdf_path)
    if chunks:
        print(f"Successfully processed {dummy_pdf_path}: Pages={page_count}, Chunks={len(chunks)}")
        print("First chunk example:", chunks[0][:200])
    else:
        print(f"Failed to process {dummy_pdf_path}")

    # Clean up dummy file
    import os

    os.remove(dummy_pdf_path)