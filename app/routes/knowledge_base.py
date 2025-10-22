from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import os
import shutil
import logging

from ..services.document_parser import DocumentParser
from ..utils.chunker import Chunker
from ..services.embedding_service import EmbeddingService
from ..services.vector_store import VectorStore
from ..services.sqlite_bm25_store import SQLiteBM25Store
from ..utils.config import DATA_DIR, SQLITE_DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

chunker = Chunker()
embedding_service = EmbeddingService()
vector_store = VectorStore()
sqlite_store = SQLiteBM25Store(db_path=SQLITE_DB_PATH)

SUPPORTED_EXTENSIONS = ["pdf", "docx", "doc"]
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

@router.post("/kb/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload et indexation avec gestion robuste des erreurs"""
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    file_location = os.path.join(DATA_DIR, file.filename)
    file_extension = file.filename.split(".")[-1].lower()
    
    # Validation
    if file_extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Type non supporté. Extensions autorisées : {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    
    try:
        # Sauvegarder le fichier
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Vérifier que le fichier n'est pas vide
        if os.path.getsize(file_location) == 0:
            os.remove(file_location)
            raise HTTPException(status_code=400, detail="Le fichier est vide")
        
        logger.info(f"Fichier sauvegardé : {file.filename} ({os.path.getsize(file_location)} bytes)")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur de sauvegarde : {e}")
        raise HTTPException(status_code=500, detail=f"Échec de sauvegarde : {e}")

    try:
        # Parser le document (retourne les chunks avec métadonnées de page)
        try:
            raw_chunks_with_meta = DocumentParser.parse_document_chunks(file_location, file_extension)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Document invalide : {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Erreur de parsing : {str(e)}")
        
        if not raw_chunks_with_meta:
            raise HTTPException(status_code=400, detail="Aucun texte extrait du document")
        
        logger.info(f"Document parsé : {len(raw_chunks_with_meta)} chunks initiaux")
        
        # Chunker davantage (chunks sémantiques)
        all_chunks = []
        all_page_nums = []
        
        for chunk_meta in raw_chunks_with_meta:
            try:
                # Vérifier si c'est un dict ou une string
                if isinstance(chunk_meta, dict):
                    text = chunk_meta["text"]
                    page_num = chunk_meta["page_num"]
                else:
                    # Ancien format (string simple)
                    text = chunk_meta
                    page_num = "N/A"
                
                semantic_chunks = chunker.chunk_text(text)
                if semantic_chunks:
                    all_chunks.extend(semantic_chunks)
                    # Répéter le numéro de page pour chaque chunk sémantique
                    all_page_nums.extend([page_num] * len(semantic_chunks))
            except Exception as e:
                logger.error(f"Erreur chunking : {e}", exc_info=True)
                # En cas d'erreur, utiliser le texte brut
                text = chunk_meta["text"] if isinstance(chunk_meta, dict) else chunk_meta
                if text.strip():
                    all_chunks.append(text)
                    all_page_nums.append("N/A")
        
        if not all_chunks:
            raise HTTPException(status_code=500, detail="Échec du découpage du texte")
        
        logger.info(f"Chunks sémantiques : {len(all_chunks)}")
        
        # Générer embeddings par batch
        BATCH_SIZE = 50
        all_embeddings = []
        
        try:
            for i in range(0, len(all_chunks), BATCH_SIZE):
                batch = all_chunks[i:i + BATCH_SIZE]
                batch_embeddings = embedding_service.get_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
                logger.info(f"Batch {i//BATCH_SIZE + 1}/{(len(all_chunks)-1)//BATCH_SIZE + 1} traité")
        except Exception as e:
            logger.error(f"Erreur embeddings : {e}")
            raise HTTPException(status_code=500, detail=f"Erreur génération embeddings : {e}")
        
        # Métadonnées avec numéros de page
        metadatas = [
            {
                "file_name": file.filename,
                "chunk_id": i,
                "chunk_size": len(chunk),
                "file_type": file_extension,
                "page_label": str(all_page_nums[i]) if i < len(all_page_nums) else "N/A"
            }
            for i, chunk in enumerate(all_chunks)
        ]
        
        # Stocker
        try:
            vector_store.add_vectors(all_chunks, all_embeddings, metadatas)
            sqlite_store.add_documents(all_chunks, metadatas)
            logger.info("✓ Stockage réussi")
        except Exception as e:
            logger.error(f"Erreur stockage : {e}")
            raise HTTPException(status_code=500, detail=f"Erreur de stockage : {e}")
        
        return {
            "message": "Document indexé avec succès",
            "filename": file.filename,
            "chunks_indexed": len(all_chunks),
            "file_size_bytes": os.path.getsize(file_location)
        }
        
    except HTTPException:
        # Nettoyer en cas d'erreur
        if os.path.exists(file_location):
            try:
                os.remove(file_location)
            except:
                pass
        raise
    except Exception as e:
        logger.error(f"Erreur inattendue : {e}", exc_info=True)
        if os.path.exists(file_location):
            try:
                os.remove(file_location)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"Erreur : {e}")


@router.post("/kb/reindex")
async def reindex_documents():
    """Réindexation avec gestion d'erreurs complète"""
    
    if not os.path.exists(DATA_DIR):
        raise HTTPException(status_code=404, detail="Répertoire introuvable")
    
    indexed_count = 0
    failed_files = []
    total_chunks = 0
    
    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    
    if not files:
        return {"message": "Aucun document à réindexer", "documents_reindexed": 0}
    
    logger.info(f"Réindexation de {len(files)} fichiers")
    
    for filename in files:
        file_path = os.path.join(DATA_DIR, filename)
        file_extension = filename.split(".")[-1].lower()

        if file_extension not in SUPPORTED_EXTENSIONS:
            continue
        
        try:
            if os.path.getsize(file_path) == 0:
                failed_files.append({"file": filename, "error": "Fichier vide"})
                continue
            
            # Parser
            raw_chunks_with_meta = DocumentParser.parse_document_chunks(file_path, file_extension)
            
            if not raw_chunks_with_meta:
                failed_files.append({"file": filename, "error": "Aucun texte"})
                continue
            
            # Chunker
            all_chunks = []
            all_page_nums = []
            
            for chunk_meta in raw_chunks_with_meta:
                # Vérifier le format
                if isinstance(chunk_meta, dict):
                    text = chunk_meta["text"]
                    page_num = chunk_meta["page_num"]
                else:
                    text = chunk_meta
                    page_num = "N/A"
                
                semantic_chunks = chunker.chunk_text(text)
                all_chunks.extend(semantic_chunks)
                all_page_nums.extend([page_num] * len(semantic_chunks))
            
            if not all_chunks:
                failed_files.append({"file": filename, "error": "Échec chunking"})
                continue
            
            # Embeddings
            BATCH_SIZE = 50
            all_embeddings = []
            for i in range(0, len(all_chunks), BATCH_SIZE):
                batch = all_chunks[i:i + BATCH_SIZE]
                batch_embeddings = embedding_service.get_embeddings(batch)
                all_embeddings.extend(batch_embeddings)
            
            # Métadonnées
            metadatas = [
                {"file_name": filename, "chunk_id": i, "file_type": file_extension}
                for i in range(len(all_chunks))
            ]
            
            # Stocker
            vector_store.add_vectors(all_chunks, all_embeddings, metadatas)
            sqlite_store.add_documents(all_chunks, metadatas)
            
            indexed_count += 1
            total_chunks += len(all_chunks)
            logger.info(f"✓ {filename} : {len(all_chunks)} chunks")
            
        except Exception as e:
            logger.error(f"✗ {filename} : {e}")
            failed_files.append({"file": filename, "error": str(e)})

    return {
        "message": "Réindexation terminée",
        "documents_reindexed": indexed_count,
        "total_chunks": total_chunks,
        "failed_files": failed_files
    }


@router.get("/kb/status")
async def get_kb_status():
    """Statut de la base de connaissances"""
    try:
        if not os.path.exists(DATA_DIR):
            return {"status": "empty", "documents": 0}
        
        files = [
            f for f in os.listdir(DATA_DIR) 
            if os.path.isfile(os.path.join(DATA_DIR, f)) 
            and f.split(".")[-1].lower() in SUPPORTED_EXTENSIONS
        ]
        
        return {
            "status": "active" if files else "empty",
            "documents": len(files),
            "files": files
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))