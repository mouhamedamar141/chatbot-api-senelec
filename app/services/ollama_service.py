from ollama import Client
from typing import List, Dict
from ..utils.config import OLLAMA_BASE_URL, OLLAMA_MODEL

class OllamaService:
    def __init__(self):
        self.client = Client(host=OLLAMA_BASE_URL)
        self.model = OLLAMA_MODEL

    def generate_response(self, prompt: str, context: List[str], chat_history: List[Dict] = None) -> str:
        messages = []
        if chat_history:
            messages.extend(chat_history)
        
        context_str = "\n\n".join(context)
        full_prompt = f"""Utilisez le contexte suivant pour répondre à la question. Si vous ne trouvez pas la réponse dans le contexte, indiquez que vous ne trouvez pas l'information dans votre base de connaissance. Ne générez pas de réponses en dehors du contexte fourni.

Contexte:
{context_str}

Question: {prompt}
Réponse en français:"""
        
        messages.append({"role": "user", "content": full_prompt})
        
        try:
            response = self.client.chat(model=self.model, messages=messages)
            return response["message"]["content"]
        except Exception as e:
            print(f"Erreur de génération de réponse avec ollama: {e}")
            return "Je ne trouve pas cette information dans ma base de connaissance."

