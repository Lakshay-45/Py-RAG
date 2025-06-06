import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from app.db import crud, models as db_models
from app.db.database import Base

# Setup an in-memory SQLite database for testing
SQLALCHEMY_DATABASE_URL_TEST = "sqlite:///:memory:"
engine_test = create_engine(SQLALCHEMY_DATABASE_URL_TEST, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine_test)


@pytest.fixture(scope="function")  # Use "function" scope to get a fresh DB for each test
def db_session() -> Session:
    Base.metadata.create_all(bind=engine_test)  # Create tables
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine_test)  # Drop tables after test


def test_create_document_metadata(db_session: Session):
    filename = "test_doc.pdf"
    doc_meta = crud.create_document_metadata(db_session, filename=filename)

    assert doc_meta.id is not None
    assert doc_meta.filename == filename
    assert doc_meta.doc_id is not None
    assert doc_meta.status == db_models.ProcessingStatus.PENDING


def test_get_document_metadata_by_doc_id(db_session: Session):
    filename = "another_doc.pdf"
    created_doc = crud.create_document_metadata(db_session, filename=filename)

    retrieved_doc = crud.get_document_metadata_by_doc_id(db_session, doc_id=created_doc.doc_id)
    assert retrieved_doc is not None
    assert retrieved_doc.id == created_doc.id
    assert retrieved_doc.filename == filename

    non_existent_doc = crud.get_document_metadata_by_doc_id(db_session, "fake-id")
    assert non_existent_doc is None


def test_get_all_document_metadata(db_session: Session):
    crud.create_document_metadata(db_session, filename="doc1.pdf")
    crud.create_document_metadata(db_session, filename="doc2.pdf")

    all_docs = crud.get_all_document_metadata(db_session, limit=10)
    assert len(all_docs) == 2

    limited_docs = crud.get_all_document_metadata(db_session, limit=1)
    assert len(limited_docs) == 1


def test_update_document_status_and_details(db_session: Session):
    doc = crud.create_document_metadata(db_session, filename="update_me.pdf")

    updated_doc = crud.update_document_status_and_details(
        db_session,
        doc_id=doc.doc_id,
        status=db_models.ProcessingStatus.COMPLETED,
        num_pages=10,
        num_chunks=50
    )
    assert updated_doc is not None
    assert updated_doc.status == db_models.ProcessingStatus.COMPLETED
    assert updated_doc.num_pages == 10
    assert updated_doc.num_chunks == 50
    assert updated_doc.processed_at is not None  # Due to onupdate


def test_set_document_failed(db_session: Session):
    doc = crud.create_document_metadata(db_session, filename="fail_me.pdf")
    error_msg = "It failed."
    failed_doc = crud.set_document_failed(db_session, doc_id=doc.doc_id, error_message=error_msg)

    assert failed_doc is not None
    assert failed_doc.status == db_models.ProcessingStatus.FAILED
    assert failed_doc.error_message == error_msg