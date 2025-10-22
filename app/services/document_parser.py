
from typing import List, Optional ,Dict
from pypdf import PdfReader
from docx import Document
import logging
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentParser:
    MAX_CHARS = 2000 * 4  # Limite pour éviter les dépassements (2000 tokens * 4 chars)
    
    @staticmethod
    def validate_pdf_file(file_content: bytes) -> tuple[bool, str]:
        """Valide la structure d'un fichier PDF"""
        try:
            if not file_content:
                return False, "Le fichier est vide"
            
            if not file_content.startswith(b'%PDF-'):
                return False, "Le fichier ne commence pas par la signature PDF"
            
            if b'startxref' not in file_content:
                return False, "Structure PDF invalide : startxref manquant"
            
            return True, ""
        except Exception as e:
            return False, f"Erreur de validation : {str(e)}"
    
    @staticmethod
    def chunk_large_text(text: str, max_chars: int = None) -> List[str]:
        """Découpe le texte en morceaux pour éviter les dépassements de tokens"""
        if max_chars is None:
            max_chars = DocumentParser.MAX_CHARS
        
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1
            
            if current_length + word_length > max_chars:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = word_length
                else:
                    chunks.append(word[:max_chars])
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    @staticmethod
    def parse_pdf(file_path: str, return_pages: bool = False) -> str:
        """Parse un PDF avec validation et gestion d'erreurs robuste"""
        text = ""
        pages_info = []  # Pour stocker les infos de page
        
        try:
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Valider le PDF
            is_valid, error_msg = DocumentParser.validate_pdf_file(file_content)
            if not is_valid:
                logger.error(f"PDF invalide ({file_path}): {error_msg}")
                raise ValueError(f"PDF invalide : {error_msg}")
            
            # Méthode 1 : pypdf
            try:
                reader = PdfReader(io.BytesIO(file_content))
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        if return_pages:
                            pages_info.append({
                                "page_num": i + 1,
                                "text": page_text,
                                "start_pos": len(text) - len(page_text) - 1
                            })
                    else:
                        logger.warning(f"Page {i+1} vide dans {file_path}")
                
                if text.strip():
                    logger.info(f"✓ PDF parsé avec pypdf : {len(text)} caractères, {len(pages_info)} pages")
                    return (text, pages_info) if return_pages else text
            except Exception as e:
                logger.warning(f"pypdf a échoué : {e}")
            
            # Méthode 2 : PyPDF2 (fallback)
            try:
                from PyPDF2 import PdfReader as PyPDF2Reader
                reader = PyPDF2Reader(io.BytesIO(file_content))
                text = ""
                pages_info = []
                
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                        if return_pages:
                            pages_info.append({
                                "page_num": i + 1,
                                "text": page_text,
                                "start_pos": len(text) - len(page_text) - 1
                            })
                
                if text.strip():
                    logger.info(f"✓ PDF parsé avec PyPDF2 : {len(text)} caractères")
                    return (text, pages_info) if return_pages else text
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"PyPDF2 a échoué : {e}")
            
            # Méthode 3 : pdfplumber (fallback)
            try:
                import pdfplumber
                pages_info = []
                with pdfplumber.open(io.BytesIO(file_content)) as pdf:
                    text = ""
                    for i, page in enumerate(pdf.pages):
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                            if return_pages:
                                pages_info.append({
                                    "page_num": i + 1,
                                    "text": page_text,
                                    "start_pos": len(text) - len(page_text) - 1
                                })
                
                if text.strip():
                    logger.info(f"✓ PDF parsé avec pdfplumber : {len(text)} caractères")
                    return (text, pages_info) if return_pages else text
            except ImportError:
                pass
            except Exception as e:
                logger.warning(f"pdfplumber a échoué : {e}")
            
            raise ValueError("Impossible d'extraire le texte avec toutes les méthodes disponibles")
            
        except FileNotFoundError:
            logger.error(f"Fichier non trouvé : {file_path}")
            raise
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Erreur inattendue : {e}", exc_info=True)
            raise ValueError(f"Erreur lors du parsing : {str(e)}")

    @staticmethod
    def parse_docx(file_path: str) -> str:
        """Parse un DOCX avec gestion d'erreurs"""
        text = ""
        try:
            doc = Document(file_path)
            
            # Extraire paragraphes
            for para in doc.paragraphs:
                if para.text.strip():
                    text += para.text + "\n"
            
            # Extraire tableaux
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text.strip():
                            text += cell.text + " "
                text += "\n"
            
            if not text.strip():
                logger.warning(f"Aucun texte dans {file_path}")
            else:
                logger.info(f"✓ DOCX parsé : {len(text)} caractères")
                
        except FileNotFoundError:
            logger.error(f"Fichier non trouvé : {file_path}")
            raise
        except Exception as e:
            logger.error(f"Erreur DOCX : {e}", exc_info=True)
            raise ValueError(f"Erreur lors du parsing DOCX : {str(e)}")
        
        return text

    @staticmethod
    def parse_document(file_path: str, file_type: str, chunk: bool = False) -> str:
        """
        Parse un document et retourne le texte
        
        Args:
            file_path: Chemin du fichier
            file_type: Type ('pdf' ou 'docx')
            chunk: Si True, découpe automatiquement les textes longs
        
        Returns:
            Texte extrait (ou premier chunk si chunk=True et texte trop long)
        """
        # Parser selon le type
        if file_type.lower() == "pdf":
            text = DocumentParser.parse_pdf(file_path)
        elif file_type.lower() in ["docx", "doc"]:
            text = DocumentParser.parse_docx(file_path)
        else:
            raise ValueError(f"Type non supporté : {file_type}")
        
        if not text.strip():
            raise ValueError("Aucun texte extrait du document")
        
        # Découper si nécessaire (pour compatibilité avec l'ancien code)
        if chunk and len(text) > DocumentParser.MAX_CHARS:
            logger.warning(f"Texte trop long ({len(text)} chars), retour du premier chunk")
            chunks = DocumentParser.chunk_large_text(text)
            return chunks[0]
        
        return text
    
    @staticmethod
    def parse_document_chunks(file_path: str, file_type: str) -> List[Dict]:
        """
        Parse un document et retourne des chunks avec métadonnées de page
        Version recommandée pour éviter les problèmes de tokens
        """
        if file_type.lower() == "pdf":
            result = DocumentParser.parse_pdf(file_path, return_pages=True)
            if isinstance(result, tuple):
                text, pages_info = result
            else:
                text = result
                pages_info = []
        elif file_type.lower() in ["docx", "doc"]:
            text = DocumentParser.parse_docx(file_path)
            pages_info = []
        else:
            raise ValueError(f"Type non supporté : {file_type}")
        
        if not text.strip():
            raise ValueError("Aucun texte extrait du document")
        
        # Retourner les chunks avec info de page
        chunks_with_metadata = []
        
        if len(text) > DocumentParser.MAX_CHARS:
            logger.info(f"Découpage en chunks : {len(text)} caractères")
            text_chunks = DocumentParser.chunk_large_text(text)
            
            for chunk in text_chunks:
                # Trouver la page du chunk
                page_num = "N/A"
                if pages_info:
                    chunk_start = text.find(chunk[:50])  # Chercher le début du chunk
                    for page_info in pages_info:
                        if chunk_start >= page_info["start_pos"]:
                            page_num = page_info["page_num"]
                
                chunks_with_metadata.append({
                    "text": chunk,
                    "page_num": page_num
                })
        else:
            page_num = 1 if pages_info else "N/A"
            chunks_with_metadata.append({
                "text": text,
                "page_num": page_num
            })
        
        return chunks_with_metadata