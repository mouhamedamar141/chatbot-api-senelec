from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import json

from ..services.embedding_service import EmbeddingService
from ..services.vector_store import VectorStore
from ..services.sqlite_bm25_store import SQLiteBM25Store
from ..services.ollama_service import OllamaService
from ..utils.config import SQLITE_DB_PATH

router = APIRouter()

embedding_service = EmbeddingService()
vector_store = VectorStore()
sqlite_store = SQLiteBM25Store(db_path=SQLITE_DB_PATH)
ollama_service = OllamaService()

class ChatRequest(BaseModel):
    query: str
    chat_history: List[Dict] = []

def clean_text(text: str) -> str:
    """Nettoie le texte des caractères de contrôle invalides pour JSON"""
    if not text:
        return text
    # Remplacer les caractères de contrôle problématiques
    return text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').strip()

@router.post("/chat")
async def chat_with_document(request: ChatRequest):
    try:
        #  generation de l'embedding de la requête
        query_embedding = embedding_service.get_embeddings([request.query])[0]

        #  RECHERCHE HYBRIDE : Vector + BM25 en parallèle
        vector_results = vector_store.search(query_embedding, limit=5)
        bm25_results = sqlite_store.search_bm25(request.query, limit=5)
        
        # Combiner les résultats (fusion)
        all_results = []
        seen_texts = set()  # Pour éviter les doublons
        
        # Ajouter les résultats vector (scores normalisés)
        for res in vector_results:
            text = res["text"]
            if text not in seen_texts:
                all_results.append({
                    "text": text,
                    "metadata": res["metadata"],
                    "score": res.get("score", 0),
                    "source": "vector"
                })
                seen_texts.add(text)
        
        # Ajouter les résultats BM25
        for res in bm25_results:
            text = res["text"]
            if text not in seen_texts:
                all_results.append({
                    "text": text,
                    "metadata": res["metadata"],
                    "score": res.get("score", 0),
                    "source": "bm25"
                })
                seen_texts.add(text)
        
        # Trier par score
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Prendre les top K résultats
        top_results = all_results[:5]
        
        # Fallback FTS5 si aucun résultat
        if not top_results:
            print("Aucun résultat trouvé, fallback vers FTS5.")
            fts5_results = sqlite_store.search_fts5(request.query, limit=5)
            if fts5_results:
                top_results = [{"text": res["text"], "metadata": res["metadata"]} for res in fts5_results]

        if not top_results:
            return {"response": "Je ne trouve pas cette information dans ma base de connaissance.", "sources": []}

        # 5. Préparer le contexte et les sources (NETTOYER LE TEXTE)
        context_chunks = [clean_text(res["text"]) for res in top_results]
        sources_metadata = [res["metadata"] for res in top_results]

        response_text = ollama_service.generate_response(request.query, context_chunks, request.chat_history)
        
        response_text = clean_text(response_text)

        sources = []
        for metadata in sources_metadata:
            sources.append({
                "page_label": clean_text(str(metadata.get("page_label", "N/A"))),
                "file_name": clean_text(str(metadata.get("file_name", "N/A")))
            })

        return {"response": response_text, "sources": sources}
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Erreur de format JSON: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur durant le chat process: {str(e)}")