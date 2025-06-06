import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.main import app
from app.db.database import Base, get_db, engine as main_app_engine  # Import main_app_engine
from app.core import config
import os
import shutil
from unittest.mock import patch
import logging

from app.services import vector_store_manager
import chromadb
from chromadb.config import Settings  # Import Settings

logger = logging.getLogger(__name__)

TEST_METADATA_DB_URL = "sqlite:///./test_metadata.db"  # Test DB in project root
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


# Store original values to restore after tests
original_vector_db_path = config.VECTOR_DB_PATH
original_chroma_client_instance = vector_store_manager.persistent_client
original_chroma_ef = vector_store_manager.default_embedding_function


@pytest.fixture(scope="module", autouse=True)
def setup_test_environment():
    global original_chroma_client_instance, original_chroma_ef, original_vector_db_path

    logger.info("Setting up test environment (module scope)...")

    # The app's lifespan event uses main_app_engine.
    app_db_url_from_config = config.METADATA_DB_URL  # e.g., "sqlite:///persistent_metadata/metadata.db"
    if app_db_url_from_config.startswith("sqlite:///"):
        # Extract the file path part
        db_file_path = app_db_url_from_config[len("sqlite:///"):]
        # If it's a relative path like "persistent_metadata/metadata.db", make it absolute from CWD
        if not os.path.isabs(db_file_path):
            db_file_path = os.path.join(os.getcwd(), db_file_path)

        app_db_dir = os.path.dirname(db_file_path)
        if app_db_dir and not os.path.exists(app_db_dir):
            try:
                os.makedirs(app_db_dir, exist_ok=True)
                logger.info(f"Created directory {app_db_dir} for main app's default engine during tests.")
            except Exception as e:
                logger.error(f"Could not create directory {app_db_dir} for main app engine: {e}")
    else:
        logger.warning(
            f"Main app METADATA_DB_URL '{app_db_url_from_config}' is not a local SQLite path; directory creation skipped.")

    # --- Test-Specific Database Setup ---
    if os.path.exists("test_metadata.db"):  # This is TEST_METADATA_DB_URL's file
        try:
            os.remove("test_metadata.db")
            logger.info("Force removed existing test_metadata.db at module setup.")
        except Exception as e:
            logger.warning(f"Could not pre-remove test_metadata.db: {e}")

    Base.metadata.create_all(bind=engine_integration_test)  # Create tables in test_metadata.db
    logger.info("Tables created for test_metadata.db using engine_integration_test.")
    app.dependency_overrides[get_db] = override_get_db  # Ensure endpoints use test_metadata.db

    # --- ChromaDB Test Setup ---
    config.VECTOR_DB_PATH = TEST_CHROMA_DB_PATH  # Override path for vector_store_manager
    if os.path.exists(TEST_CHROMA_DB_PATH):
        try:
            shutil.rmtree(TEST_CHROMA_DB_PATH)
            logger.info(f"Force removed existing {TEST_CHROMA_DB_PATH} at module setup.")
        except Exception as e:
            logger.warning(f"Could not pre-remove {TEST_CHROMA_DB_PATH}: {e}")
    os.makedirs(TEST_CHROMA_DB_PATH, exist_ok=True)

    try:
        test_chroma_settings = Settings(allow_reset=True, anonymized_telemetry=False)
        # Re-initialize the global client in vector_store_manager
        if hasattr(vector_store_manager, 'persistent_client') and vector_store_manager.persistent_client is not None:
            pass

        vector_store_manager.persistent_client = chromadb.PersistentClient(
            path=TEST_CHROMA_DB_PATH,
            settings=test_chroma_settings
        )
        if hasattr(vector_store_manager.embedding_functions, 'SentenceTransformerEmbeddingFunction'):
            vector_store_manager.default_embedding_function = vector_store_manager.embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=vector_store_manager.SENTENCE_TRANSFORMER_MODEL_NAME
            )
        logger.info(f"Test ChromaDB client initialized at {TEST_CHROMA_DB_PATH} with allow_reset=True.")
    except Exception as e:
        logger.error(f"Fatal error setting up test ChromaDB client: {e}", exc_info=True)
        pytest.fail(f"Could not set up test ChromaDB client: {e}")

    yield  # Tests run here

    # --- Teardown ---
    logger.info("Tearing down test environment (module scope)...")
    app.dependency_overrides.clear()  # Clear FastAPI dependency overrides

    # Teardown for test_metadata.db
    if engine_integration_test:
        Base.metadata.drop_all(bind=engine_integration_test)  # Drop tables from test_metadata.db
        engine_integration_test.dispose()
        logger.info("Test SQLAlchemy engine (engine_integration_test) disposed.")
    if os.path.exists("test_metadata.db"):
        try:
            os.remove("test_metadata.db")
            logger.info("test_metadata.db removed.")
        except PermissionError as e:
            logger.warning(f"Could not remove test_metadata.db during teardown: {e}")

    # Teardown for ChromaDB
    if hasattr(vector_store_manager, 'persistent_client') and vector_store_manager.persistent_client:
        try:
            logger.info(f"Resetting ChromaDB at {TEST_CHROMA_DB_PATH} during teardown...")
            vector_store_manager.persistent_client.reset()
            logger.info(f"ChromaDB at {TEST_CHROMA_DB_PATH} reset successfully.")
        except Exception as e_reset:
            logger.error(f"Error resetting test ChromaDB during teardown: {e_reset}", exc_info=True)
        # Help garbage collection
        del vector_store_manager.persistent_client
        vector_store_manager.persistent_client = None
        if hasattr(vector_store_manager, 'default_embedding_function'):
            del vector_store_manager.default_embedding_function
            vector_store_manager.default_embedding_function = None
        import gc
        gc.collect()
        logger.info("Forced garbage collection after ChromaDB client deletion.")

    if os.path.exists(TEST_CHROMA_DB_PATH):
        try:
            shutil.rmtree(TEST_CHROMA_DB_PATH)
            logger.info(f"{TEST_CHROMA_DB_PATH} removed.")
        except Exception as e:  # Catch broader exceptions for rmtree if PermissionError is too specific
            logger.warning(f"Could not remove {TEST_CHROMA_DB_PATH} during teardown: {e}")

    # Teardown for the main app's DB path if its directory was created by tests
    if 'app_db_dir' in locals() and app_db_dir and os.path.exists(app_db_dir):  # Check if app_db_dir was defined
        db_file_to_remove = os.path.join(app_db_dir, os.path.basename(config.METADATA_DB_URL.split("///")[-1]))
        if os.path.exists(db_file_to_remove):
            try:
                os.remove(db_file_to_remove)
                logger.info(f"Removed main app DB file at {db_file_to_remove} if it was created by app startup.")
            except Exception as e:
                logger.warning(f"Could not remove main app DB file {db_file_to_remove} during teardown: {e}")
        # Try to remove the directory if it's empty
        try:
            if not os.listdir(app_db_dir):
                os.rmdir(app_db_dir)
                logger.info(f"Removed empty main app DB directory {app_db_dir}.")
        except Exception as e:
            logger.warning(f"Could not remove main app DB directory {app_db_dir}: {e}")

    # Restore original configurations
    config.VECTOR_DB_PATH = original_vector_db_path
    vector_store_manager.persistent_client = original_chroma_client_instance
    vector_store_manager.default_embedding_function = original_chroma_ef
    logger.info("Original configurations restored.")


@pytest.fixture(scope="function")
def client(setup_test_environment):  # This fixture just ensures setup_test_environment has run
    with TestClient(app) as c:
        yield c


# Helper to create a dummy PDF file for uploads
def create_test_pdf_file(tmp_path, filename="test_upload.pdf",
                         content="This is test PDF content for integration test."):
    from reportlab.pdfgen import canvas
    file_path = tmp_path / filename  # tmp_path is a Path object
    c = canvas.Canvas(str(file_path))
    c.drawString(72, 800, content)
    c.showPage()
    c.save()
    return file_path


# --- Test Functions ---

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
    target_filename = "doc_for_get_all.pdf"
    pdf_file_path = create_test_pdf_file(tmp_path, filename=target_filename)
    with open(pdf_file_path, "rb") as f:
        upload_response = client.post("/api/v1/documents/upload",
                                      files={"files": (pdf_file_path.name, f, "application/pdf")})
        assert upload_response.status_code == 201

    response = client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)

    found_doc = None
    for doc in data:
        if doc["filename"] == target_filename:
            found_doc = doc
            break
    assert found_doc is not None, f"{target_filename} not found in the list of documents. Found: {[d['filename'] for d in data]}"
    assert found_doc["filename"] == target_filename


def test_get_specific_document(client: TestClient, tmp_path):
    pdf_file_path = create_test_pdf_file(tmp_path, filename="specific_doc.pdf")
    doc_id = None
    with open(pdf_file_path, "rb") as f:
        upload_response = client.post("/api/v1/documents/upload",
                                      files={"files": (pdf_file_path.name, f, "application/pdf")})
        assert upload_response.status_code == 201
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


@patch('app.services.rag_pipeline.get_llm_response')
def test_query_system_successful_retrieval(mock_get_llm_response, client: TestClient, tmp_path):
    mock_get_llm_response.return_value = "Mocked LLM answer based on context."

    content_to_query = "The secret ingredient for this test is specific."
    pdf_file_path = create_test_pdf_file(tmp_path, filename="query_doc.pdf", content=content_to_query)
    doc_id = None
    with open(pdf_file_path, "rb") as f:
        upload_resp = client.post("/api/v1/documents/upload",
                                  files={"files": (pdf_file_path.name, f, "application/pdf")})
        assert upload_resp.status_code == 201
        processed_file_data = upload_resp.json()["processed_files"][0]
        assert processed_file_data[
                   "status"] == "completed", f"Upload failed: {processed_file_data.get('error_message')}"
        doc_id = processed_file_data["doc_id"]

    assert doc_id is not None

    query_payload = {"query": "What is the secret ingredient for this test?", "doc_id": doc_id, "top_k": 1}
    response = client.post("/api/v1/query", json=query_payload)

    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Mocked LLM answer based on context."
    assert data["retrieved_chunks_found"] is True
    assert len(data["source_chunks"]) > 0
    assert content_to_query in data["source_chunks"][0]["text"], "Queried content not found in retrieved chunks"

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
        mock_llm_no_context.return_value = "This mock should not be hit if hardcoded response is used."

        response = client.post("/api/v1/query", json=query_payload)

    assert response.status_code == 200
    data = response.json()
    logger.info(f"Response for no_chunks_found test: {data}")
    assert data["retrieved_chunks_found"] is False, f"Expected no chunks, but found: {data['source_chunks']}"
    assert len(data["source_chunks"]) == 0
    # This assertion depends on the hardcoded message in rag_pipeline.py when no chunks are found
    assert data["answer"] == "I could not find any relevant information in the documents for your query."