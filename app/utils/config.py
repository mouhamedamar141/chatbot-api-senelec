import os

# Répertoire de données
DATA_DIR = os.getenv("DATA_DIR", "data")

# configuration de ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b") # ou mistral ou autres 

# configuration du model d'embedding 
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-large-en-v1.5")

# configuration Vector DB 
VECTOR_DB_TYPE = os.getenv("VECTOR_DB_TYPE", "lancedb")  # qdrant or lancedb
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
LANCEDB_PATH = os.getenv("LANCEDB_PATH", "./lancedb")

# configurationSQLite BM25 
SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "knowledge_base.db")

# Chunking configuration - AJUSTÉ POUR ÉVITER LES ERREURS
# Réduit de 700 à 450 pour éviter les dépassements de tokens
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 450))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

# Limites de sécurité
MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", 500000))  # 500k caractères max
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 50))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Créer le répertoire data si nécessaire
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)