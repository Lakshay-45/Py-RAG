import pytest
import os
from app.services import document_processor
from reportlab.pdfgen import canvas  # For creating a dummy PDF

# Helper function to create a dummy PDF for testing
def create_dummy_pdf(path: str, num_pages: int, text_per_page: str):
    c = canvas.Canvas(path)
    for i in range(num_pages):
        c.drawString(72, 800 - (i * 20) % 700, f"{text_per_page} Page {i + 1}")  # Basic text
        c.showPage()
    c.save()


@pytest.fixture(scope="module")
def dummy_pdf_path():
    pdf_path = "test_dummy_doc.pdf"
    create_dummy_pdf(pdf_path, num_pages=3, text_per_page="This is test content.")
    yield pdf_path  # provide the path to the test
    os.remove(pdf_path)  # cleanup after tests in this module are done


@pytest.fixture(scope="module")
def empty_pdf_path():
    pdf_path = "empty_test_doc.pdf"
    c = canvas.Canvas(pdf_path)  # Create a canvas
    c.save()  # Saving it, it will be a valid PDF but with no drawable content / pages
    yield pdf_path
    os.remove(pdf_path)


@pytest.fixture(scope="module")
def long_pdf_path():
    pdf_path = "long_test_doc.pdf"
    create_dummy_pdf(pdf_path, num_pages=5, text_per_page="L" * 300)  # ~300 chars per page
    yield pdf_path
    os.remove(pdf_path)


def test_load_and_split_pdf_successful(dummy_pdf_path):
    chunks, page_count = document_processor.load_and_split_pdf(
        dummy_pdf_path,
        chunk_size=100,  # Small chunk size for testing
        chunk_overlap=10
    )
    assert page_count == 3
    assert len(chunks) > 0
    assert isinstance(chunks, list)
    assert all(isinstance(chunk, str) for chunk in chunks)
    assert "This is test content. Page 1" in chunks[0]  # Check content


def test_load_and_split_pdf_file_not_found():
    chunks, page_count = document_processor.load_and_split_pdf("non_existent_file.pdf")
    assert page_count == 0
    assert len(chunks) == 0


def test_load_and_split_pdf_empty_or_unreadable(empty_pdf_path):
    # This test depends on how pypdf handles truly empty PDFs or those it can't parse text from.
    chunks, page_count = document_processor.load_and_split_pdf(empty_pdf_path)

    # If PdfReader can open it and finds 0 pages, page_count will be 0.
    # If it opens and finds pages but no text, page_count is >0, chunks is [].
    # document_processor logs warnings in these cases.
    assert len(chunks) == 0
    # page_count could be 0 or more depending on the PDF structure. The key is no chunks.


def test_chunking_logic(long_pdf_path):
    # Test if chunking works as expected with overlaps
    # For this, we set smaller chunk sizes to force multiple chunks
    chunk_size = 350  # Slightly more than one page's content
    chunk_overlap = 50
    chunks, page_count = document_processor.load_and_split_pdf(
        long_pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    assert page_count == 5
    assert len(chunks) > 1  # Expecting multiple chunks

    for chunk in chunks:
        assert len(chunk) <= chunk_size + 100  # Allow some leeway for splitter behavior
