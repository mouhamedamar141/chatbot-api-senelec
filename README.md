# Chatbot SENELEC API

Cette API FastAPI implémente un chatbot basé sur la génération augmentée par récupération (RAG), capable de répondre à des questions en se basant sur des documents PDF ou DOCX uploadés. Il utilise une combinaison de bases de données vectorielles (Qdrant ou LanceDB) et textuelles (SQLite avec BM25/FTS5) pour la recherche, et un modèle de langage local (LLM) via Ollama pour la génération de réponses.

## Structure du projet

```
chatbot-api/
├── app/
│   ├── main.py
│   ├── routes/
│   │   ├── health.py
│   │   ├── knowledge_base.py
│   │   └── chat.py
│   ├── services/
│   │   ├── document_parser.py
│   │   ├── embedding_service.py
│   │   ├── vector_store.py
│   │   ├── sqlite_bm25_store.py
│   │   └── ollama_service.py
│   ├── utils/
│   │   ├── chunker.py
│   │   └── config.py
│   └── __init__.py
├── data/
│   └── (stocke les documents PDF/DOCX et la base de données SQLite)
├── requirements.txt
└── README.md
```

## Technologies utilisées

*   **Backend:** FastAPI
*   **LLM:** Ollama (modèle local, ex: `llama3`, `mistral`, `deepseek-r1`)
*   **RAG Framework:** LlamaIndex (pour l'orchestration)
*   **Vector DB:** Qdrant ou LanceDB
*   **Recherche classique:** SQLite + BM25 (via `sqlite-utils` et `rank-bm25`) + FTS5
*   **Embeddings:** HuggingFace (`BAAI/bge-large-en-v1.5`)
*   **Langage:** Python  3.11.4

## Fonctionnalités

*   **Upload de documents:** Téléchargez des fichiers PDF ou DOCX dans le dossier `/data`.
*   **Indexation:** Le contenu des documents est extrait, découpé en `chunks`, et indexé dans:
    *   Une base vectorielle (Qdrant ou LanceDB) pour les embeddings.
    *   Une base textuelle (SQLite avec BM25 et FTS5) pour la recherche classique.
*   **Interrogation:** Posez des questions au chatbot, qui utilise les documents indexés pour générer des réponses pertinentes en français.
*   **Historique de conversation:** Gestion d'une mémoire courte pour les conversations.
*   **Fallback automatique:** Si la recherche vectorielle ne donne pas de résultats, le système bascule automatiquement sur la recherche BM25, puis FTS5.
*   **Endpoint `/health`:** Pour vérifier l'état de l'API.

## Installation

1.  **Cloner le dépôt (ou dézipper le projet):**

    ```bash
    git clone <URL_DU_DEPOT>
    cd chatbot-api
    ```

2.  **Créer un environnement virtuel et activer:**

   **Linux/Mac:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

    **Windows:**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Installer les dépendances Python:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Installer Ollama et télécharger un modèle:**

    Suivez les instructions sur le site officiel d'Ollama pour l'installation:
    [https://ollama.com/](https://ollama.com/)

    Après l'installation, téléchargez un modèle. Par exemple, pour `llama3`:

    ```bash
    ollama pull mistral # ou llama ....
    ```

    Vous pouvez configurer le modèle utilisé dans `app/utils/config.py` via la variable d'environnement `OLLAMA_MODEL`.

## Modèles Ollama Recommandés pour RAG

Voici les modèles les plus performants pour un chatbot en production avec réponses rapides (< 3 secondes) :

| Rang | Modèle | Taille | RAM Min | Temps Réponse | Qualité RAG | Recommandation |
|------|--------|--------|---------|---------------|-------------|----------------|
| 1 | **phi3:3.8b** | 3.8B | 4-6 GB | **< 2 sec** | ⭐⭐⭐⭐⭐ | ✅ **Meilleur choix** - Excellent compromis |
| 2 | **llama3.2:3b** | 3B | 4-6 GB | **< 2 sec** | ⭐⭐⭐⭐ | ✅ Très bon français, rapide |
| 3 | **llama3.2:1b** | 1B | 2-3 GB | **< 1 sec** | ⭐⭐⭐ | ✅ Ultra-rapide pour FAQ simple |
| 4 | **qwen2.5:3b** | 3B | 4-6 GB | **2-3 sec** | ⭐⭐⭐⭐ | ✅ Excellent pour docs techniques |
| 5 | **mistral:7b-q4** | 7B | 5-7 GB | **2-3 sec** | ⭐⭐⭐⭐⭐ | ⚠️ Version quantifiée plus rapide |
| 6 | **gemma2:2b** | 2B | 3-4 GB | **< 2 sec** | ⭐⭐⭐ | ✅ Très léger et rapide |

### 🎯 Recommandation selon votre serveur :

- **Serveur standard (8 vCPU, 16GB RAM)** : `phi3:3.8b` ou `llama3.2:3b`
- **Serveur limité (4 vCPU, 8GB RAM)** : `llama3.2:1b` ou `gemma2:2b`
- **Serveur avec GPU** : `mistral:7b` (devient très rapide avec GPU)

### ⚠️ Modèles à éviter en production (trop lents) :
- ❌ `mistral:7b` (sans GPU) : 4-5 secondes
- ❌ `llama3.1:8b` : 5-6 secondes
- ❌ Tout modèle > 12B sans GPU

5.  **Configuration des bases de données (optionnel):**

    *   **Qdrant:** Si vous utilisez Qdrant, assurez-vous qu'un serveur Qdrant est en cours d'exécution. Vous pouvez le lancer via Docker:

        ```bash
        docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
        ```

        Configurez `QDRANT_HOST` et `QDRANT_PORT` dans `app/utils/config.py` ou via des variables d'environnement.

    *   **LanceDB:** LanceDB est basé sur des fichiers et ne nécessite pas de serveur séparé. Le chemin est configuré dans `app/utils/config.py` via `LANCEDB_PATH`.

    *   **SQLite:** La base de données SQLite sera créée automatiquement dans le dossier racine du projet.

    Vous pouvez choisir entre Qdrant et LanceDB en définissant la variable d'environnement `VECTOR_DB_TYPE` à `qdrant` ou `lancedb` (par défaut `qdrant`).

## Utilisation

1.  **Lancer l'API:**

    ```bash
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

    L'API sera accessible à `http://localhost:8000`.

2.  **Accéder à la documentation interactive (Swagger UI):**

    Ouvrez votre navigateur et allez à `http://localhost:8000/docs`.

3.  **Endpoints de l'API:**

    *   **`/health` (GET):** Vérifie si l'API est active.

        ```bash
        curl http://localhost:8000/health
        ```

    *   **`/kb/upload` (POST):** Uploade un document (PDF ou DOCX) et l'indexe.

        ```bash
        curl -X POST "http://localhost:8000/kb/upload" \
             -H "accept: application/json" \
             -H "Content-Type: multipart/form-data" \
             -F "file=@/path/to/your/document.pdf;type=application/pdf"
        ```

    *   **`/kb/reindex` (POST):** Réindexe tous les documents présents dans le dossier `/data`.

        ```bash
        curl -X POST "http://localhost:8000/kb/reindex" \
             -H "accept: application/json"
        ```

    *   **`/chat` (POST):** Pose une question au chatbot.

        ```bash
        curl -X POST "http://localhost:8000/chat" \
             -H "accept: application/json" \
             -H "Content-Type: application/json" \
             -d '{"query": "Quels sont les moyens de paiement acceptés ", "chat_history": []}'
        ```

        Exemple avec historique de conversation:

        ```bash
        curl -X POST "http://localhost:8000/chat" \
             -H "accept: application/json" \
             -H "Content-Type: application/json" \
             -d '{"query": "Comment lire mon compteur électrique  ?", "chat_history": []}'
        ```

## Fichier d'évaluation (jeu de Q/A tests)

Un fichier `SENELEC_FAQ_Abonnement.pdf` est fourni, contenant un ensemble de questions et de réponses attendues pour tester la performance du chatbot.
---

**Auteur:** Mouhamed Amar
**Date:** Octobre 2025

