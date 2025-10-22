from fastapi import FastAPI
from .routes import health, knowledge_base, chat

app = FastAPI(title="Chatbot RAG API")

app.include_router(health.router)
app.include_router(knowledge_base.router)
app.include_router(chat.router)

@app.on_event("startup")
async def startup_event():
    pass

@app.on_event("shutdown")
async def shutdown_event():
    pass

