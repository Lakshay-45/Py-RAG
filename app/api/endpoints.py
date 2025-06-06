from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query as FastAPIQuery
from sqlalchemy.orm import Session
from typing import List, Dict
import shutil  # For saving uploaded files temporarily
import os
import logging

from app.db.database import get_db
from app.db import crud, models as db_models
from app.models import schemas  # Pydantic schemas
from app.services import document_processor, vector_store_manager, rag_pipeline
from app.core import config

router = APIRouter()
logger = logging.getLogger(__name__)

# Ensure upload directory exists
UPLOAD_DIR = "./data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/documents/upload", response_model=schemas.DocumentUploadResponse, status_code=201)
async def upload_documents(
        files: List[UploadFile] = File(..., description="Up to 20 PDF documents, each max 1000 pages."),
        db: Session = Depends(get_db)
):
    """
    Upload one or more PDF documents for processing and ingestion.
    - Supports up to 20 documents per request.
    - Each document should be a PDF and ideally under 1000 pages (processing is capped at 1000).
    """
    if len(files) > 20:
        raise HTTPException(status_code=400, detail="Cannot upload more than 20 documents at a time.")

    processed_files_metadata: List[schemas.DocumentMetadataResponse] = []
    upload_errors: List[Dict[str, str]] = []

    for file in files:
        if not file.filename:
            upload_errors.append({"file": "Unknown", "error": "File has no name."})
            continue
        if not file.filename.lower().endswith(".pdf"):
            upload_errors.append({"file": file.filename, "error": "Invalid file type. Only PDF is supported."})
            logger.warning(f"Skipping non-PDF file: {file.filename}")
            continue

        temp_file_path = os.path.join(UPLOAD_DIR, file.filename)

        try:
            # Save uploaded file temporarily
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            logger.info(f"Temporarily saved uploaded file: {temp_file_path}")

            # 1. Create initial metadata entry
            doc_meta_db = crud.create_document_metadata(db, filename=file.filename)
            doc_id = doc_meta_db.doc_id
            logger.info(f"Created metadata for {file.filename} with doc_id: {doc_id}")

            # 2. Update status to processing
            crud.set_document_processing(db, doc_id=doc_id)
            db.refresh(doc_meta_db)  # Refresh to get the updated status

            # 3. Load PDF and chunk text
            logger.info(f"Processing and chunking {file.filename} (doc_id: {doc_id})...")
            chunks, num_pages = document_processor.load_and_split_pdf(temp_file_path)

            if not chunks and num_pages == 0 and not os.path.exists(
                    temp_file_path):  # Check if file even exists after pypdf
                # Handling if pypdf failed to open file
                error_msg = "PDF file could not be read or is corrupted."
                logger.error(error_msg + f" File: {file.filename}")
                crud.set_document_failed(db, doc_id=doc_id, error_message=error_msg)
                upload_errors.append({"file": file.filename, "error": error_msg})
                db.refresh(doc_meta_db)
                processed_files_metadata.append(schemas.DocumentMetadataResponse.model_validate(doc_meta_db))
                continue

            if not chunks:  # If chunks is empty but num_pages might be > 0 (e.g. image PDF)
                error_msg = f"No text content extracted from PDF or file is empty. Pages found: {num_pages}."
                logger.warning(error_msg + f" File: {file.filename}")
                crud.set_document_failed(db, doc_id=doc_id, error_message=error_msg)
                # Still add to processed_files_metadata with failed status
                db.refresh(doc_meta_db)  # Refresh to get updated status and processed_at
                processed_files_metadata.append(schemas.DocumentMetadataResponse.model_validate(doc_meta_db))
                continue  # Skip to next file if no text to embed

            logger.info(f"Extracted {len(chunks)} chunks and {num_pages} pages from {file.filename}.")

            # 4. Add chunks to vector database
            success_vector_db = vector_store_manager.add_chunks_to_vector_db(
                doc_id=doc_id,
                chunks=chunks
            )

            if success_vector_db:
                crud.set_document_completed(db, doc_id=doc_id, num_pages=num_pages, num_chunks=len(chunks))
                logger.info(f"Successfully processed and ingested {file.filename}")
            else:
                error_msg = "Failed to add document chunks to vector database."
                crud.set_document_failed(db, doc_id=doc_id, error_message=error_msg)
                upload_errors.append({"file": file.filename, "error": error_msg})
                logger.error(f"{error_msg} for {file.filename}")

            db.refresh(doc_meta_db)  # Refresh to get the latest state
            processed_files_metadata.append(schemas.DocumentMetadataResponse.model_validate(doc_meta_db))

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}", exc_info=True)
            # If doc_meta_db was created, mark as failed
            if 'doc_meta_db' in locals() and doc_meta_db:
                crud.set_document_failed(db, doc_id=doc_meta_db.doc_id,
                                         error_message=f"Unexpected processing error: {str(e)}")
                db.refresh(doc_meta_db)
                processed_files_metadata.append(schemas.DocumentMetadataResponse.model_validate(doc_meta_db))
            upload_errors.append({"file": file.filename, "error": f"Unexpected processing error: {str(e)}"})
        finally:
            # Clean up the temporarily saved file
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Removed temporary file: {temp_file_path}")
                except OSError as e_os:
                    logger.error(f"Error removing temporary file {temp_file_path}: {e_os}")
            if 'file' in locals() and hasattr(file, 'file') and not file.file.closed:
                file.file.close()

    return schemas.DocumentUploadResponse(
        message="Document processing complete. Check status for each file.",
        processed_files=processed_files_metadata,
        errors=upload_errors
    )


@router.post("/query", response_model=schemas.QueryResponse)
async def query_system(
        request: schemas.QueryRequest,
        # db: Session = Depends(get_db) # Not directly needed unless logging queries or something
):
    """
    Query the RAG system with a user question.
    Optionally, scope the query to a specific `doc_id`.
    """
    if not rag_pipeline.llm_client:  # Or check if rag_pipeline.openai_client is None
        raise HTTPException(status_code=503, detail="OpenAI API key not configured. Querying is disabled.")

    logger.info(f"Received query: '{request.query}', doc_id_filter: {request.doc_id}, top_k: {request.top_k}")

    rag_result = rag_pipeline.answer_query(
        user_query=request.query,
        top_k_chunks=request.top_k,
        doc_id_filter=request.doc_id
    )

    # Convert source_chunks from RAG pipeline's format to Pydantic schema format
    source_chunks_response = [
        schemas.SourceChunk(
            text=chunk.get("text", ""),
            metadata=chunk.get("metadata", {}),
            distance=chunk.get("distance")
        ) for chunk in rag_result.get("source_chunks", [])
    ]

    return schemas.QueryResponse(
        query=rag_result["query"],
        answer=rag_result["answer"],
        retrieved_chunks_found=rag_result["retrieved_chunks_found"],
        source_chunks=source_chunks_response
    )


@router.get("/documents", response_model=List[schemas.DocumentMetadataResponse])
async def list_all_documents_metadata(
        skip: int = FastAPIQuery(0, ge=0),
        limit: int = FastAPIQuery(10, ge=1, le=100),
        db: Session = Depends(get_db)
):
    """
    Retrieve metadata for all processed documents with pagination.
    """
    logger.info(f"Fetching all document metadata: skip={skip}, limit={limit}")
    metadata_db = crud.get_all_document_metadata(db, skip=skip, limit=limit)
    return [schemas.DocumentMetadataResponse.model_validate(meta) for meta in metadata_db]


@router.get("/documents/{doc_id}", response_model=schemas.DocumentMetadataResponse)
async def get_single_document_metadata(doc_id: str, db: Session = Depends(get_db)):
    """
    Retrieve metadata for a specific document by its `doc_id`.
    """
    logger.info(f"Fetching metadata for doc_id: {doc_id}")
    metadata_db = crud.get_document_metadata_by_doc_id(db, doc_id=doc_id)
    if metadata_db is None:
        raise HTTPException(status_code=404, detail=f"Document with doc_id '{doc_id}' not found.")
    return schemas.DocumentMetadataResponse.model_validate(metadata_db)