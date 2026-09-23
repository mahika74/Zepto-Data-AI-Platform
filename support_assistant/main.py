from fastapi import FastAPI
from pydantic import BaseModel, Field

from .graph import ask_question, SupportResponse


app = FastAPI(
    title="Zepto Support Assistant",
    description="LangGraph + ChromaDB + FastAPI support assistant",
    version="1.0.0"
)


class AskRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Customer question"
    )


@app.get("/")
def root():
    return {
        "message": "Zepto Support Assistant is running."
    }


@app.post(
    "/ask",
    response_model=SupportResponse
)
def ask(request: AskRequest):
    return ask_question(request.query)
