"""
FastAPI backend. Run with:
    uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

from app.config import (
    TRAINED_MODEL_PATH, H5AD_PATH, MOESM3_PATH, MOESM5_PATH, PAPER_PDF_PATH,
)
from app.structured_retrieval import StructuredStore
from app.vector_store import PaperVectorStore
from app.model_service import ModelService
from app.rag_report import generate_explanation
from app.chatbot import ChatSession

app = FastAPI(title="BiSpec Pairwise AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this before any real deployment
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Load everything ONCE at startup, not per-request ----
store = StructuredStore(MOESM3_PATH, MOESM5_PATH)
vectors = PaperVectorStore(PAPER_PDF_PATH)
model = ModelService(TRAINED_MODEL_PATH, H5AD_PATH)  # this loads the .h5ad into memory -- slow startup, expected

# session_id -> ChatSession
_sessions: dict[str, ChatSession] = {}


class PairRequest(BaseModel):
    gene_a: str
    gene_b: str


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PairRequest):
    ctx = store.lookup(req.gene_a, req.gene_b)
    model_out = model.predict(req.gene_a, req.gene_b, ctx)
    return {**ctx, **model_out}


@app.post("/report")
def report(req: PairRequest):
    ctx = store.lookup(req.gene_a, req.gene_b)
    model_out = model.predict(req.gene_a, req.gene_b, ctx)
    chunks = vectors.retrieve(f"{req.gene_a} {req.gene_b} target combination")
    explanation = generate_explanation(ctx, model_out, chunks)
    return {**ctx, **model_out, "explanation": explanation, "n_chunks_used": len(chunks)}


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    if session_id not in _sessions:
        _sessions[session_id] = ChatSession(store, vectors, model)
    reply = _sessions[session_id].ask(req.message)
    return {"session_id": session_id, "reply": reply}
