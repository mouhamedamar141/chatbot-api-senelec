from typing import List, Dict
from qdrant_client import QdrantClient, models
import lancedb
import pyarrow as pa
import logging
from ..utils.config import VECTOR_DB_TYPE, QDRANT_HOST, QDRANT_PORT, LANCEDB_PATH

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, collection_name: str = "knowledge_base"):
        self.collection_name = collection_name
        self.db_type = VECTOR_DB_TYPE
        self.client = None
        self.table = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialise le client avec le bon schéma"""
        try:
            if self.db_type == "qdrant":
                self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
                
                # Vérifier si la collection existe
                collections = self.client.get_collections().collections
                collection_exists = any(c.name == self.collection_name for c in collections)
                
                if not collection_exists:
                    self.client.create_collection(
                        collection_name=self.collection_name,
                        vectors_config=models.VectorParams(
                            size=1024, 
                            distance=models.Distance.COSINE
                        ),
                    )
                    logger.info(f"✓ Collection Qdrant créée : {self.collection_name}")
                else:
                    logger.info(f"✓ Collection Qdrant existante : {self.collection_name}")
                    
            elif self.db_type == "lancedb":
                db = lancedb.connect(LANCEDB_PATH)
                
                # Définir le schéma avec TOUS les champs nécessaires
                schema = pa.schema([
                    pa.field("vector", pa.list_(pa.float32(), 1024)),
                    pa.field("text", pa.string()),
                    pa.field("file_name", pa.string()),
                    pa.field("chunk_id", pa.int64()),
                    pa.field("chunk_size", pa.int64()),
                    pa.field("file_type", pa.string()),
                ])
                
                # Vérifier si la table existe
                table_names = db.table_names()
                
                if self.collection_name in table_names:
                    logger.info(f"✓ Table LanceDB existante : {self.collection_name}")
                    self.table = db.open_table(self.collection_name)
                else:
                    # Créer la table avec le bon schéma
                    self.table = db.create_table(
                        self.collection_name,
                        schema=schema,
                        mode="create"
                    )
                    logger.info(f"✓ Table LanceDB créée : {self.collection_name}")
            else:
                raise ValueError(f"Type de DB vectorielle non supporté : {self.db_type}")
                
        except Exception as e:
            logger.error(f"Erreur d'initialisation du VectorStore : {e}", exc_info=True)
            raise

    def add_vectors(self, texts: List[str], embeddings: List[List[float]], metadatas: List[Dict]):
        """Ajoute des vecteurs avec métadonnées complètes"""
        try:
            if self.db_type == "qdrant":
                # Obtenir le dernier ID pour éviter les collisions
                try:
                    collection_info = self.client.get_collection(self.collection_name)
                    last_id = collection_info.points_count
                except:
                    last_id = 0
                
                points = []
                for i, (text, embedding, metadata) in enumerate(zip(texts, embeddings, metadatas)):
                    # Créer le payload avec toutes les métadonnées
                    payload = {
                        "text": text,
                        "file_name": metadata.get("file_name", "unknown"),
                        "chunk_id": metadata.get("chunk_id", i),
                        "chunk_size": metadata.get("chunk_size", len(text)),
                        "file_type": metadata.get("file_type", "unknown"),
                    }
                    
                    # Ajouter les métadonnées supplémentaires
                    for key, value in metadata.items():
                        if key not in payload:
                            payload[key] = value
                    
                    points.append(
                        models.PointStruct(
                            id=last_id + i,
                            vector=embedding,
                            payload=payload
                        )
                    )
                
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                logger.info(f"✓ {len(points)} vecteurs ajoutés à Qdrant")
                
            elif self.db_type == "lancedb":
                data = []
                for text, embedding, metadata in zip(texts, embeddings, metadatas):
                    # Préparer les données avec TOUS les champs du schéma
                    record = {
                        "vector": embedding,
                        "text": text,
                        "file_name": metadata.get("file_name", "unknown"),
                        "chunk_id": metadata.get("chunk_id", 0),
                        "chunk_size": metadata.get("chunk_size", len(text)),
                        "file_type": metadata.get("file_type", "unknown"),
                    }
                    data.append(record)
                
                # Ajouter à la table
                self.table.add(data)
                logger.info(f"✓ {len(data)} vecteurs ajoutés à LanceDB")
                
        except Exception as e:
            logger.error(f"Erreur lors de l'ajout de vecteurs : {e}", exc_info=True)
            raise

    def search(self, query_embedding: List[float], limit: int = 5) -> List[Dict]:
        """Recherche avec gestion d'erreurs"""
        results = []
        
        try:
            if self.db_type == "qdrant":
                search_result = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_embedding,
                    limit=limit,
                    with_payload=True
                )
                
                for hit in search_result:
                    results.append({
                        "text": hit.payload.get("text", ""),
                        "score": hit.score,
                        "metadata": {
                            "file_name": hit.payload.get("file_name", "N/A"),
                            "chunk_id": hit.payload.get("chunk_id", 0),
                            "chunk_size": hit.payload.get("chunk_size", 0),
                            "file_type": hit.payload.get("file_type", "N/A"),
                            "page_label": hit.payload.get("page_label", "N/A"),
                        }
                    })
                    
            elif self.db_type == "lancedb":
                search_result = self.table.search(query_embedding).limit(limit).to_list()
                
                for hit in search_result:
                    results.append({
                        "text": hit.get("text", ""),
                        "score": 1.0 / (1.0 + hit.get("_distance", 0)),  # Convertir distance en score
                        "metadata": {
                            "file_name": hit.get("file_name", "N/A"),
                            "chunk_id": hit.get("chunk_id", 0),
                            "chunk_size": hit.get("chunk_size", 0),
                            "file_type": hit.get("file_type", "N/A"),
                            "page_label": hit.get("page_label", "N/A"),
                        }
                    })
                    
            logger.info(f"✓ Recherche effectuée : {len(results)} résultats")
            
        except Exception as e:
            logger.error(f"Erreur lors de la recherche : {e}", exc_info=True)
            return []
        
        return results

    def delete_collection(self):
        """Supprime la collection/table"""
        try:
            if self.db_type == "qdrant":
                self.client.delete_collection(collection_name=self.collection_name)
                logger.info(f"✓ Collection Qdrant supprimée : {self.collection_name}")
            elif self.db_type == "lancedb":
                db = lancedb.connect(LANCEDB_PATH)
                db.drop_table(self.collection_name)
                logger.info(f"✓ Table LanceDB supprimée : {self.collection_name}")
        except Exception as e:
            logger.error(f"Erreur lors de la suppression : {e}")
            raise

    def count_documents(self) -> int:
        """Compte le nombre de documents stockés"""
        try:
            if self.db_type == "qdrant":
                collection_info = self.client.get_collection(self.collection_name)
                return collection_info.points_count
            elif self.db_type == "lancedb":
                return len(self.table.to_pandas())
        except Exception as e:
            logger.error(f"Erreur lors du comptage : {e}")
            return 0