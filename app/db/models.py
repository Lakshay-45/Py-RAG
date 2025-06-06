from sqlalchemy import Column, Integer, String, DateTime, Enum as SAEnum
from sqlalchemy.sql import func
from .database import Base
import enum

class ProcessingStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    doc_id = Column(String, unique=True, index=True, nullable=False) # Internal unique ID
    num_pages = Column(Integer, nullable=True)
    num_chunks = Column(Integer, nullable=True)
    status = Column(SAEnum(ProcessingStatus), default=ProcessingStatus.PENDING)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
    processed_at = Column(DateTime(timezone=True), nullable=True, onupdate=func.now())
    error_message = Column(String, nullable=True) # To store any processing errors

    def __repr__(self):
        return f"<DocumentMetadata(doc_id='{self.doc_id}', filename='{self.filename}', status='{self.status}')>"