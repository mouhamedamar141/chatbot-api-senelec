from sentence_transformers import SentenceTransformer
from typing import List
from ..utils.config import EMBEDDING_MODEL_NAME

class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

