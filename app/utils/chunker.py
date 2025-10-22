from typing import List
from transformers import AutoTokenizer
import logging
from ..utils.config import CHUNK_SIZE, CHUNK_OVERLAP

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Chunker:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.max_chunk_size = CHUNK_SIZE
            self.chunk_overlap = CHUNK_OVERLAP
            logger.info(f"Chunker initialisé : max={self.max_chunk_size}, overlap={self.chunk_overlap}")
        except Exception as e:
            logger.error(f"Erreur d'initialisation du tokenizer : {e}")
            raise

    def chunk_text(self, text: str) -> List[str]:
        """
        Découpe le texte en chunks avec gestion robuste des erreurs
        """
        if not text or not text.strip():
            logger.warning("Texte vide fourni au chunker")
            return []
        
        try:
            # Tokenizer le texte
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if not tokens:
                logger.warning("Aucun token généré")
                return [text]
            
            # Si le texte est plus court que la taille du chunk, le retourner tel quel
            if len(tokens) <= self.max_chunk_size:
                return [text]
            
            chunks = []
            num_chunks = 0
            
            # Découper en chunks avec overlap
            for i in range(0, len(tokens), self.max_chunk_size - self.chunk_overlap):
                chunk_tokens = tokens[i : i + self.max_chunk_size]
                
                # Vérifier que le chunk n'est pas vide
                if not chunk_tokens:
                    continue
                
                try:
                    # Décoder le chunk
                    chunk_text = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
                    
                    # Vérifier que le chunk décodé n'est pas vide
                    if chunk_text.strip():
                        chunks.append(chunk_text)
                        num_chunks += 1
                    else:
                        logger.warning(f"Chunk {num_chunks} vide après décodage, ignoré")
                        
                except Exception as e:
                    logger.error(f"Erreur lors du décodage du chunk {num_chunks}: {e}")
                    continue
            
            if not chunks:
                logger.warning("Aucun chunk valide généré, retour du texte original")
                return [text]
            
            logger.info(f"Texte découpé en {len(chunks)} chunks (tokens: {len(tokens)})")
            return chunks
            
        except Exception as e:
            logger.error(f"Erreur critique dans chunk_text : {e}", exc_info=True)
            # En cas d'erreur, retourner le texte par morceaux de caractères
            return self._fallback_chunking(text)
    
    def _fallback_chunking(self, text: str) -> List[str]:
        """
        Méthode de secours : découpage basé sur les caractères
        Utilisée quand le tokenizer échoue
        """
        logger.warning("Utilisation du chunking de secours (par caractères)")
        
        # Estimer ~4 caractères par token
        chars_per_chunk = self.max_chunk_size * 4
        overlap_chars = self.chunk_overlap * 4
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 pour l'espace
            
            if current_length + word_length > chars_per_chunk:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    
                    # Overlap : garder les derniers mots
                    overlap_words = []
                    overlap_length = 0
                    for w in reversed(current_chunk):
                        if overlap_length + len(w) < overlap_chars:
                            overlap_words.insert(0, w)
                            overlap_length += len(w) + 1
                        else:
                            break
                    
                    current_chunk = overlap_words + [word]
                    current_length = overlap_length + word_length
                else:
                    # Mot trop long, le forcer dans un chunk
                    chunks.append(word)
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk.append(word)
                current_length += word_length
        
        # Ajouter le dernier chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Fallback chunking : {len(chunks)} chunks générés")
        return chunks if chunks else [text]
    
    def get_chunk_info(self, text: str) -> dict:
        """
        Retourne des informations sur le chunking sans découper
        Utile pour debugging
        """
        try:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            num_tokens = len(tokens)
            estimated_chunks = max(1, (num_tokens - self.chunk_overlap) // (self.max_chunk_size - self.chunk_overlap))
            
            return {
                "text_length": len(text),
                "num_tokens": num_tokens,
                "estimated_chunks": estimated_chunks,
                "max_chunk_size": self.max_chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "will_be_chunked": num_tokens > self.max_chunk_size
            }
        except Exception as e:
            logger.error(f"Erreur dans get_chunk_info : {e}")
            return {
                "text_length": len(text),
                "error": str(e)
            }