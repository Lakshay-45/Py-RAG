import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
VECTOR_DB_PATH = "./chroma_db_data" # Path for ChromaDB persistence
METADATA_DB_URL = "sqlite:///persistent_metadata/metadata.db" # SQLite for metadata
