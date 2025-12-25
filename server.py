from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

from rag_answer import answer_question


app = FastAPI(
    title="VAC Policy Assistant API",
    description="RAG-powered API that explains Veterans Affairs Canada policies.",
    version="0.1.0",
)

# CORS so web UI & (later) mobile apps can call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str
    mode: Optional[str] = "explain"  # "explain" or "recommend"
    profile: Optional[dict[str, Any]]= None



class Source(BaseModel):
    title: str | None = None
    url: str | None = None
    section: str | None = None
    effective_date: str | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.get("/")
def root():
    return {"message": "VAC Policy Assistant API is running"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    print("PROFILE RECEIVED:", req.profile)  # <-- TEMP DEBUG
    mode = req.mode or "explain"
    profile = req.profile or {}
    result = answer_question(req.question, mode=mode, profile=profile)
    return AskResponse(answer=result["answer"], sources=result["sources"])

