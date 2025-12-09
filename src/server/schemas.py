"""
Pydantic schemas shared by REST endpoints and UI helpers.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    doc_type: Optional[str] = Field(
        default=None,
        description="Optional metadata filter such as 'product_terms'",
    )
    history: List[ChatMessage] = Field(
        default_factory=list,
        description="Previous chat history in OpenAI format",
    )


class SourceDocument(BaseModel):
    source: Optional[str]
    doc_type: Optional[str]
    preview: str


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]


# ---------- Flashcard schemas for /flashcards ----------

class Flashcard(BaseModel):
    question: str = Field(..., description="Front of the flashcard")
    answer: str = Field(..., description="Back of the flashcard")


class FlashcardResponse(BaseModel):
    flashcards: List[Flashcard] = Field(
        default_factory=list,
        description="Generated flashcards",
    )
