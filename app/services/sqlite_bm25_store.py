import sqlite_utils
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any
import logging
from ..utils.config import SQLITE_DB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SQLiteBM25Store:
    def __init__(self, db_path: str = SQLITE_DB_PATH, table_name: str = "documents"):
        self.db = sqlite_utils.Database(db_path)
        self.table_name = table_name
        self._initialize_db()
        self.bm25 = None
        self.corpus = []
        self.document_ids = []

    def _initialize_db(self):
        """Initialise la base de données avec le bon schéma"""
        try:
            if self.table_name not in self.db.table_names():
                # Créer la table avec TOUS les champs
                self.db[self.table_name].create({
                    "id": int,
                    "text": str,
                    "file_name": str,
                    "chunk_id": int,
                    "chunk_size": int,
                    "file_type": str,
                    "page_label": str,
                }, pk="id")
                
                # Activer FTS5 pour la recherche full-text
                self.db[self.table_name].enable_fts(
                    ["text"],
                    fts_version="FTS5"
                )
                logger.info(f"✓ Table SQLite créée : {self.table_name}")
            else:
                logger.info(f"✓ Table SQLite existante : {self.table_name}")
                
        except Exception as e:
            logger.error(f"Erreur d'initialisation SQLite : {e}", exc_info=True)
            raise

    def add_documents(self, chunks: List[str], metadatas: List[Dict[str, Any]]):
        """Ajoute des documents avec métadonnées complètes"""
        try:
            records = []
            
            # Obtenir le dernier ID
            last_id = 0
            try:
                result = list(self.db.query(f"SELECT MAX(id) as max_id FROM {self.table_name}"))
                if result and result[0]["max_id"] is not None:
                    last_id = result[0]["max_id"]
            except:
                pass
            
            for i, (chunk, metadata) in enumerate(zip(chunks, metadatas)):
                record = {
                    "id": last_id + i + 1,
                    "text": chunk,
                    "file_name": metadata.get("file_name", "unknown"),
                    "chunk_id": metadata.get("chunk_id", i),
                    "chunk_size": metadata.get("chunk_size", len(chunk)),
                    "file_type": metadata.get("file_type", "unknown"),
                    "page_label": metadata.get("page_label", "N/A"),
                }
                records.append(record)
                
                # Ajouter au corpus BM25
                self.corpus.append(chunk.lower().split())
                self.document_ids.append(record["id"])
            
            # Insérer dans SQLite
            self.db[self.table_name].insert_all(records, alter=True, replace=True)
            
            # Reconstruire l'index BM25
            self.bm25 = BM25Okapi(self.corpus)
            
            logger.info(f"✓ {len(records)} documents ajoutés à SQLite")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de documents : {e}", exc_info=True)
            raise

    def search_bm25(self, query: str, limit: int = 5) -> List[Dict]:
        """Recherche BM25 avec gestion d'erreurs"""
        try:
            if not self.bm25:
                # Reconstruire BM25 si nécessaire
                logger.info("Reconstruction de l'index BM25...")
                self.corpus = []
                self.document_ids = []
                
                for row in self.db.query(f"SELECT id, text FROM {self.table_name}"):
                    self.corpus.append(row["text"].lower().split())
                    self.document_ids.append(row["id"])
                
                if not self.corpus:
                    logger.warning("Aucun document dans la base")
                    return []
                
                self.bm25 = BM25Okapi(self.corpus)

            tokenized_query = query.lower().split()
            doc_scores = self.bm25.get_scores(tokenized_query)
            
            # Top N indices
            top_n_indices = sorted(
                range(len(doc_scores)), 
                key=lambda i: doc_scores[i], 
                reverse=True
            )[:limit]
            
            results = []
            for idx in top_n_indices:
                doc_id = self.document_ids[idx]
                doc_data = self.db[self.table_name].get(doc_id)
                
                if doc_data:
                    results.append({
                        "text": doc_data["text"],
                        "score": float(doc_scores[idx]),
                        "metadata": {
                            "file_name": doc_data.get("file_name", "N/A"),
                            "chunk_id": doc_data.get("chunk_id", 0),
                            "chunk_size": doc_data.get("chunk_size", 0),
                            "file_type": doc_data.get("file_type", "N/A"),
                            "page_label": doc_data.get("page_label", "N/A"),
                        }
                    })
            
            logger.info(f"✓ BM25 : {len(results)} résultats trouvés")
            return results
            
        except Exception as e:
            logger.error(f"Erreur BM25 : {e}", exc_info=True)
            return []

    def search_fts5(self, query: str, limit: int = 5) -> List[Dict]:
        """Recherche FTS5 avec gestion d'erreurs"""
        try:
            results = []
            
            sql = f"""
                SELECT id, text, file_name, chunk_id, chunk_size, file_type, page_label 
                FROM {self.table_name} 
                WHERE {self.table_name} MATCH :query 
                ORDER BY rank 
                LIMIT :limit
            """
            
            for row in self.db.query(sql, {"query": query, "limit": limit}):
                results.append({
                    "text": row["text"],
                    "score": 1.0,  # FTS5 ne fournit pas de score direct
                    "metadata": {
                        "file_name": row.get("file_name", "N/A"),
                        "chunk_id": row.get("chunk_id", 0),
                        "chunk_size": row.get("chunk_size", 0),
                        "file_type": row.get("file_type", "N/A"),
                        "page_label": row.get("page_label", "N/A"),
                    }
                })
            
            logger.info(f"✓ FTS5 : {len(results)} résultats trouvés")
            return results
            
        except Exception as e:
            logger.error(f"Erreur FTS5 : {e}", exc_info=True)
            return []

    def get_all_documents(self) -> List[Dict]:
        """Retourne tous les documents"""
        try:
            return list(self.db[self.table_name].rows)
        except Exception as e:
            logger.error(f"Erreur get_all_documents : {e}")
            return []

    def count_documents(self) -> int:
        """Compte le nombre de documents"""
        try:
            result = list(self.db.query(f"SELECT COUNT(*) as count FROM {self.table_name}"))
            return result[0]["count"] if result else 0
        except Exception as e:
            logger.error(f"Erreur count : {e}")
            return 0

    def clear_all(self):
        """Supprime tous les documents"""
        try:
            self.db[self.table_name].drop()
            self._initialize_db()
            self.corpus = []
            self.document_ids = []
            self.bm25 = None
            logger.info("✓ Base de données vidée")
        except Exception as e:
            logger.error(f"Erreur clear_all : {e}")
            raise