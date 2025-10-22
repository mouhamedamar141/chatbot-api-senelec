from fastapi import APIRouter
from datetime import datetime
import os
import logging

from ..services.vector_store import VectorStore
from ..services.sqlite_bm25_store import SQLiteBM25Store
from ..utils.config import (
    DATA_DIR, VECTOR_DB_TYPE, SQLITE_DB_PATH
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", tags=["Surveillance"])
async def health_check():
    """Vérifie l’état général du système : bases vectorielles, BM25 et fichiers locaux."""
    
    statut = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "bases_de_donnees": {},
        "erreurs": []
    }

    #  Vérification de la base vectorielle (Qdrant ou LanceDB)
    try:
        vector_store = VectorStore()
        nombre_chunks = vector_store.count_documents()  # Nombre total de morceaux vectorisés
        
        statut["bases_de_donnees"]["vector_db"] = {
            "type": VECTOR_DB_TYPE,
            "chunks": nombre_chunks,
            "status": "healthy" if nombre_chunks > 0 else "empty"
        }
    except Exception as e:
        statut["bases_de_donnees"]["vector_db"] = {"status": "error", "erreur": str(e)}
        statut["erreurs"].append(f"VectorDB : {e}")

    #  Vérification de la base SQLite (BM25)
    try:
        sqlite_store = SQLiteBM25Store()
        nb_documents = sqlite_store.count_documents()
        statut["bases_de_donnees"]["sqlite_bm25"] = {
            "documents": nb_documents,
            "path": SQLITE_DB_PATH,
            "status": "healthy" if os.path.exists(SQLITE_DB_PATH) else "missing"
        }
    except Exception as e:
        statut["bases_de_donnees"]["sqlite_bm25"] = {"status": "error", "erreur": str(e)}
        statut["erreurs"].append(f"SQLite : {e}")

    #  Vérification du dossier de données (PDF / DOCX)
    try:
        if os.path.exists(DATA_DIR):
            fichiers = [
                f for f in os.listdir(DATA_DIR)
                if os.path.isfile(os.path.join(DATA_DIR, f))
            ]
            pdfs = [f for f in fichiers if f.endswith('.pdf')]
            docxs = [f for f in fichiers if f.endswith('.docx')]

            taille_totale = round(
                sum(os.path.getsize(os.path.join(DATA_DIR, f)) for f in fichiers) / (1024 * 1024), 2
            )

            statut["bases_de_donnees"]["data_dir"] = {
                "path": DATA_DIR,
                "total_fichiers": len(fichiers),
                "pdf_files": len(pdfs),
                "docx_files": len(docxs),
                "taille_totale_Mo": taille_totale,
                "status": "healthy"
            }
        else:
            statut["bases_de_donnees"]["data_dir"] = {"path": DATA_DIR, "status": "missing"}
    except Exception as e:
        statut["bases_de_donnees"]["data_dir"] = {"status": "error", "erreur": str(e)}
        statut["erreurs"].append(f"DataDir : {e}")

    # 🟡 Déterminer le statut global
    if statut["erreurs"]:
        statut["status"] = "degraded" if len(statut["erreurs"]) < 3 else "unhealthy"

    return statut
