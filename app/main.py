from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from app.api import endpoints as api_endpoints
from app.db.database import engine, Base, SessionLocal
from app.core import config
from app.services import vector_store_manager
from app.services import rag_pipeline
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)

# --- Database Table Creation ---
# This function will be called by the startup event handler
def create_db_and_tables_sync():
    logger.info("Creating database tables if they don't exist...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables checked/created successfully.")
    except Exception as e:
        logger.error(f"Error creating database tables: {e}", exc_info=True)
        raise


@asynccontextmanager
async def lifespan(app_instance: FastAPI):  # FastAPI instance is passed by FastAPI itself
    # Code to run on startup
    logger.info("Application startup via lifespan...")
    create_db_and_tables_sync()  # Call the synchronous function

    if not vector_store_manager.persistent_client:
        logger.error("ChromaDB client failed to initialize during startup. Vector operations may fail.")
    if not vector_store_manager.default_embedding_function:
        logger.error(
            "Default Embedding Function (Sentence Transformer) in vector_store_manager failed to initialize. Vector operations may fail.")
    else:
        logger.info("Default Embedding Function (Sentence Transformer) initialized successfully.")

    if not rag_pipeline.llm_client:
        provider_name = rag_pipeline.LLM_PROVIDER.capitalize() if hasattr(rag_pipeline, 'LLM_PROVIDER') else "LLM"
        api_key_missing = False
        if hasattr(rag_pipeline, 'LLM_PROVIDER'):
            if rag_pipeline.LLM_PROVIDER == "groq" and not config.GROQ_API_KEY:
                api_key_missing = True
            elif rag_pipeline.LLM_PROVIDER == "openai" and not config.OPENAI_API_KEY:
                api_key_missing = True
        if api_key_missing:
            logger.warning(f"{provider_name} API key not found. Client not initialized.")
        else:
            logger.warning(f"{provider_name} client for RAG (rag_pipeline) not initialized for other reasons.")
    elif hasattr(rag_pipeline, 'LLM_PROVIDER'):
        logger.info(
            f"{rag_pipeline.LLM_PROVIDER.capitalize()} client for RAG initialized (model: {rag_pipeline.LLM_MODEL_NAME}).")

    yield  # Application runs here

    # Code to run on shutdown
    logger.info("Application shutdown via lifespan...")
    if engine:
        engine.dispose()
        logger.info("SQLAlchemy engine disposed on shutdown.")


# --- FastAPI App Initialization ---
app = FastAPI(
    title="RAG Specialist API",
    description="API for Retrieval-Augmented Generation with document upload and querying.",
    version="0.1.0",
    lifespan=lifespan # Assign the lifespan context manager
)


# --- Custom Exception Handler for Validation Errors ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    error_messages = []
    for error in exc.errors():
        field = ".".join(str(loc) for loc in error['loc']) if error['loc'] else "general"
        error_messages.append({field: error['msg']})
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Validation Error", "errors": error_messages},
    )

# --- API Routers ---
app.include_router(api_endpoints.router, prefix="/api/v1", tags=["RAG System"])

# --- Root Endpoint ---
@app.get("/", tags=["General"])
async def root():
    """
    Root endpoint providing a welcome message.
    """
    return {"message": "Welcome to the RAG Specialist API. Visit /docs for API documentation."}
