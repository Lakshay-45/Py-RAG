from sqlalchemy.orm import Session
from . import models
from typing import List, Optional
import uuid

def create_document_metadata(db: Session, filename: str) -> models.DocumentMetadata:
    """
    Creates an initial metadata record for a document.
    Generates a unique doc_id.
    """
    db_doc = models.DocumentMetadata(
        doc_id=str(uuid.uuid4()), # Generate a unique ID for the document
        filename=filename,
        status=models.ProcessingStatus.PENDING
    )
    db.add(db_doc)
    db.commit()
    db.refresh(db_doc)
    return db_doc

def get_document_metadata_by_doc_id(db: Session, doc_id: str) -> Optional[models.DocumentMetadata]:
    """
    Retrieves document metadata by its unique doc_id.
    """
    return db.query(models.DocumentMetadata).filter(models.DocumentMetadata.doc_id == doc_id).first()

def get_all_document_metadata(db: Session, skip: int = 0, limit: int = 100) -> List[models.DocumentMetadata]:
    """
    Retrieves all document metadata entries with pagination.
    """
    return db.query(models.DocumentMetadata).offset(skip).limit(limit).all()

def update_document_status_and_details(
    db: Session,
    doc_id: str,
    status: models.ProcessingStatus,
    num_pages: Optional[int] = None,
    num_chunks: Optional[int] = None,
    error_message: Optional[str] = None
) -> Optional[models.DocumentMetadata]:
    """
    Updates the status and other details of a document.
    """
    db_doc = get_document_metadata_by_doc_id(db, doc_id)
    if db_doc:
        db_doc.status = status
        if num_pages is not None:
            db_doc.num_pages = num_pages
        if num_chunks is not None:
            db_doc.num_chunks = num_chunks
        db_doc.error_message = error_message # Will clear previous error if new one is None
        if status in [models.ProcessingStatus.COMPLETED, models.ProcessingStatus.FAILED]:
            pass
        db.commit()
        db.refresh(db_doc)
    return db_doc

def set_document_processing(db: Session, doc_id: str) -> Optional[models.DocumentMetadata]:
    return update_document_status_and_details(db, doc_id, models.ProcessingStatus.PROCESSING)

def set_document_completed(db: Session, doc_id: str, num_pages: int, num_chunks: int) -> Optional[models.DocumentMetadata]:
    return update_document_status_and_details(db, doc_id, models.ProcessingStatus.COMPLETED, num_pages, num_chunks)

def set_document_failed(db: Session, doc_id: str, error_message: str) -> Optional[models.DocumentMetadata]:
    return update_document_status_and_details(db, doc_id, models.ProcessingStatus.FAILED, error_message=error_message)