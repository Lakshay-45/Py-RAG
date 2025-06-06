import pytest
from chromadb.config import Settings
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app, logger
from app.db.database import Base, get_db
from app.core import config
import os
import chromadb
import shutil
import logging
from unittest.mock import patch

from app.services import vector_store_manager

logger = logging.getLogger(__name__)

TEST_METADATA_DB_URL = "sqlite:///./test_metadata.db"
TEST_CHROMA_DB_PATH = "./test_chroma_db_data"

engine_integration_test = create_engine(
    TEST_METADATA_DB_URL, connect_args={"check_same_thread": False}
)
TestingSessionLocalIntegration = sessionmaker(autocommit=False, autoflush=False, bind=engine_integration_test)


def override_get_db():
    try:
        db = TestingSessionLocalIntegration()
        yield db
    finally:
        db.close()


original_vector_db_path = config.VECTOR_DB_PATH
original_chroma_client_instance = vector_store_manager.persistent_client
original_chroma_ef = vector_store_manager.default_embedding_function


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    global original_chroma_client_instance, original_chroma_ef

    logger.info("Setting up test environment (module scope)...")
    config.VECTOR_DB_PATH = TEST_CHROMA_DB_PATH

    # Clean up old test directories if they exist from a previous run
    if os.path.exists(TEST_CHROMA_DB_PATH):
        try:
            shutil.rmtree(TEST_CHROMA_DB_PATH)
            logger.info(f"Force removed existing {TEST_CHROMA_DB_PATH} at module setup.")
        except Exception as e:
            logger.warning(f"Could not pre-remove {TEST_CHROMA_DB_PATH}: {e}")
    os.makedirs(TEST_CHROMA_DB_PATH, exist_ok=True)

    if os.path.exists("test_metadata.db"):
        try:
            os.remove("test_metadata.db")
            logger.info("Force removed existing test_metadata.db at module setup.")
        except Exception as e:
            logger.warning(f"Could not pre-remove test_metadata.db: {e}")

    # Initialize a new Chroma client for tests with allow_reset=True
    # This client will be used by vector_store_manager for the duration of these tests.
    try:
        test_chroma_settings = Settings(allow_reset=True, anonymized_telemetry=False)
        vector_store_manager.persistent_client = chromadb.PersistentClient(
            path=TEST_CHROMA_DB_PATH,
            settings=test_chroma_settings
        )
        # Re-initialize embedding function as it might be tied to client or settings implicitly
        if hasattr(vector_store_manager.embedding_functions, 'SentenceTransformerEmbeddingFunction'):
            vector_store_manager.default_embedding_function = vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=vector_store_manager.SENTENCE_TRANSFORMER_MODEL_NAME
            )
        logger.info(f"Test ChromaDB client initialized at {TEST_CHROMA_DB_PATH} with allow_reset=True.")
    except Exception as e:
        logger.error(f"Fatal error setting up test ChromaDB client: {e}", exc_info=True)
        pytest.fail(f"Could not set up test ChromaDB client: {e}")

    Base.metadata.create_all(bind=engine_integration_test)
    app.dependency_overrides[get_db] = override_get_db

    yield  # Tests run here

    logger.info("Tearing down test environment (module scope)...")
    app.dependency_overrides.clear()
    Base.metadata.drop_all(bind=engine_integration_test)
    if engine_integration_test:  # Check if engine exists
        engine_integration_test.dispose()
        logger.info("Test SQLAlchemy engine disposed.")

    # Reset and attempt to delete ChromaDB data
    if vector_store_manager.persistent_client:
        try:
            logger.info(f"Resetting ChromaDB at {TEST_CHROMA_DB_PATH} before teardown...")
            vector_store_manager.persistent_client.reset()  # Clears all data
            logger.info(f"ChromaDB at {TEST_CHROMA_DB_PATH} reset successfully.")
        except Exception as e_reset:
            logger.error(f"Error resetting test ChromaDB: {e_reset}", exc_info=True)

    # Try to release/delete the client instance from the module
    # This is to encourage Python's garbage collector to release file handles
    if hasattr(vector_store_manager, 'persistent_client'):
        del vector_store_manager.persistent_client
        vector_store_manager.persistent_client = None
    if hasattr(vector_store_manager, 'default_embedding_function'):
        del vector_store_manager.default_embedding_function
        vector_store_manager.default_embedding_function = None

    import gc
    gc.collect()
    logger.info("Forced garbage collection.")

    # Final attempt to remove physical files/directories
    if os.path.exists("test_metadata.db"):
        try:
            os.remove("test_metadata.db")
            logger.info("test_metadata.db removed.")
        except PermissionError as e:
            logger.warning(f"Could not remove test_metadata.db during teardown: {e}")

    if os.path.exists(TEST_CHROMA_DB_PATH):
        try:
            shutil.rmtree(TEST_CHROMA_DB_PATH)
            logger.info(f"{TEST_CHROMA_DB_PATH} removed.")
        except PermissionError as e:
            logger.warning(f"Could not remove {TEST_CHROMA_DB_PATH} during teardown: {e}")

    # Restore original config and client if they were set
    config.VECTOR_DB_PATH = original_vector_db_path
    vector_store_manager.persistent_client = original_chroma_client_instance
    vector_store_manager.default_embedding_function = original_chroma_ef
    logger.info("Original configurations restored.")


@pytest.fixture(scope="function")
def client(setup_test_environment):
    with TestClient(app) as c:
        yield c


# Helper to create a dummy PDF file for uploads
def create_test_pdf_file(tmp_path, filename="test_upload.pdf",
                         content="This is test PDF content for integration test."):
    from reportlab.pdfgen import canvas
    file_path = tmp_path / filename
    c = canvas.Canvas(str(file_path))
    c.drawString(72, 800, content)
    c.showPage()
    c.save()
    return file_path


def test_root_endpoint(client: TestClient):
    response = client.get("/")
    assert response.status_code == 200
    assert "Welcome to the RAG Specialist API" in response.json()["message"]


def test_upload_single_document(client: TestClient, tmp_path):
    pdf_file_path = create_test_pdf_file(tmp_path)
    with open(pdf_file_path, "rb") as f:
        response = client.post("/api/v1/documents/upload", files={"files": (pdf_file_path.name, f, "application/pdf")})

    assert response.status_code == 201
    data = response.json()
    assert data["message"] == "Document processing complete. Check status for each file."
    assert len(data["processed_files"]) == 1
    processed_file = data["processed_files"][0]
    assert processed_file["filename"] == pdf_file_path.name
    assert processed_file["status"] == "completed"
    assert processed_file["num_pages"] == 1
    assert processed_file["num_chunks"] > 0


def test_get_all_documents_after_upload(client: TestClient, tmp_path):
    # Upload a document to ensure there's data
    target_filename="doc_for_get_all.pdf"
    pdf_file_path = create_test_pdf_file(tmp_path, filename=target_filename)
    with open(pdf_file_path, "rb") as f:
        upload_response = client.post("/api/v1/documents/upload", files={"files": (pdf_file_path.name, f, "application/pdf")})
        assert upload_response.status_code == 201  # Ensure upload was successful

    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    # Search for the uploaded document in the list
    found_doc = None
    for doc in data:
        if doc["filename"] == target_filename:
            found_doc = doc
            break

    assert found_doc is not None, f"{target_filename} not found in the list of documents"
    assert found_doc["filename"] == target_filename

def test_get_specific_document(client: TestClient, tmp_path):
    # Upload a doc first
    pdf_file_path = create_test_pdf_file(tmp_path, filename="specific_doc.pdf")
    doc_id = None
    with open(pdf_file_path, "rb") as f:
        upload_response = client.post("/api/v1/documents/upload",
                                      files={"files": (pdf_file_path.name, f, "application/pdf")})
        doc_id = upload_response.json()["processed_files"][0]["doc_id"]

    assert doc_id is not None
    response = client.get(f"/api/v1/documents/{doc_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["doc_id"] == doc_id
    assert data["filename"] == "specific_doc.pdf"


def test_get_non_existent_document(client: TestClient):
    response = client.get("/api/v1/documents/fake-doc-id-123")
    assert response.status_code == 404


# Patch the RAG pipeline's LLM call to avoid actual API calls during integration tests of query endpoint
@patch('app.services.rag_pipeline.get_llm_response')
def test_query_system_successful_retrieval(mock_get_llm_response, client: TestClient, tmp_path):
    mock_get_llm_response.return_value = "Mocked LLM answer based on context."

    # Upload a document with known content
    content_to_query = "The secret ingredient is love."
    pdf_file_path = create_test_pdf_file(tmp_path, filename="query_doc.pdf", content=content_to_query)
    doc_id = None
    with open(pdf_file_path, "rb") as f:
        upload_resp = client.post("/api/v1/documents/upload",
                                  files={"files": (pdf_file_path.name, f, "application/pdf")})
        assert upload_resp.status_code == 201
        assert upload_resp.json()["processed_files"][0]["status"] == "completed"
        doc_id = upload_resp.json()["processed_files"][0]["doc_id"]

    assert doc_id is not None

    query_payload = {"query": "What is the secret ingredient?", "doc_id": doc_id, "top_k": 1}
    response = client.post("/api/v1/query", json=query_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Mocked LLM answer based on context."
    assert data["retrieved_chunks_found"] is True
    assert len(data["source_chunks"]) > 0

    # Check if LLM mock was called with a prompt containing context
    mock_get_llm_response.assert_called_once()
    args, _ = mock_get_llm_response.call_args
    prompt_arg = args[0]
    assert "Context:" in prompt_arg
    assert content_to_query in prompt_arg


def test_query_system_no_chunks_found(client: TestClient):
    logger.info("Running test_query_system_no_chunks_found...")
    if vector_store_manager.persistent_client:
        try:
            logger.info(f"Resetting ChromaDB via client for test_query_system_no_chunks_found...")
            vector_store_manager.persistent_client.reset()
            logger.info(f"ChromaDB reset for test_query_system_no_chunks_found successful.")
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB for test_query_system_no_chunks_found: {e}", exc_info=True)
    else:
        logger.warning(
            "vector_store_manager.persistent_client is None in test_query_system_no_chunks_found, cannot reset.")

    query_payload = {"query": "AbsolutelyUniqueStringThatDoesNotExistInAnyDocumentEver123XYZ", "top_k": 1}

    with patch('app.services.rag_pipeline.get_llm_response') as mock_llm_no_context:
        mock_llm_no_context.return_value = "Mocked: I don't know, no context found."
        response = client.post("/api/v1/query", json=query_payload)

    assert response.status_code == 200
    data = response.json()
    logger.info(f"Response for no_chunks_found test: {data}")
    assert data["retrieved_chunks_found"] is False, f"Expected no chunks, but found: {data['source_chunks']}"
    assert len(data["source_chunks"]) == 0
    assert data["answer"] == "I could not find any relevant information in the documents for your query."
