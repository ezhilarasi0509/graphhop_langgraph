"""uvicorn app.api:app --reload   ->   http://localhost:8000/docs"""
from typing import Literal
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
from . import answer
from .diagram import pipeline_mermaid

app = FastAPI(title="GraphHop: Knowledge Graph RAG (LangGraph)")


class AskRequest(BaseModel):
    question: str
    mode: Literal["auto", "vector", "graph", "both"] = "auto"


@app.post("/ask")
def ask(req: AskRequest):
    return answer.ask(req.question, req.mode)


@app.get("/pipeline", response_class=PlainTextResponse)
def pipeline_diagram():
    """The real pipeline as a Mermaid diagram, generated from the compiled LangGraph."""
    return pipeline_mermaid()


@app.get("/health")
def health():
    return {"status": "ok"}
