from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict
from datetime import datetime
from app.db.models import ProcessingStatus # Import the enum

class DocumentMetadataBase(BaseModel):
    filename: str

class DocumentMetadataResponse(DocumentMetadataBase):
    id: int # From the database model
    status: ProcessingStatus
    uploaded_at: datetime
    processed_at: Optional[datetime] = None
    num_pages: Optional[int] = None
    num_chunks: Optional[int] = None
    error_message: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)

class DocumentUploadResponse(BaseModel):
    message: str
    # doc_ids: List[str] # Return full metadata for each uploaded doc
    processed_files: List[DocumentMetadataResponse]
    errors: List[Dict[str, str]] # To report errors for specific files if any

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The user's query.")
    doc_id: Optional[str] = Field(None, description="Optional: ID of a specific document to query against.")
    top_k: Optional[int] = Field(3, ge=1, le=10, description="Number of relevant chunks to retrieve.")

class SourceChunk(BaseModel):
    text: str
    metadata: dict
    distance: Optional[float] = None # Distance from query embedding

class QueryResponse(BaseModel):
    query: str
    answer: str
    retrieved_chunks_found: bool
    source_chunks: List[SourceChunk] # To show what context was used