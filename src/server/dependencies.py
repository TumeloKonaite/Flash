"""
Shared dependency helpers for server-facing modules.
"""

from functools import lru_cache

from src.pipeline.rag_pipeline import RAGPipeline
from src.flashcards.flashcard_pipeline import FlashcardPipeline


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    """
    Lazily instantiate a single RAGPipeline instance for reuse across
    FastAPI routes and the Gradio UI.
    """
    return RAGPipeline()


@lru_cache(maxsize=1)
def get_flashcard_pipeline() -> FlashcardPipeline:
    """
    Lazily instantiate a single FlashcardPipeline instance for reuse across
    FastAPI routes that generate flashcards from uploaded PDFs.
    """
    return FlashcardPipeline()
