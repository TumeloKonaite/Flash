# src/flashcards/flashcard_pipeline.py

from __future__ import annotations

import json
from typing import List, Dict

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI  # adjust if you use a different LLM client

from src.chunking.document_chunking import DocumentChunking


class FlashcardPipeline:
    """
    Simple pipeline to generate flashcards from Documents.

    Flow:
    - Take LangChain Documents (e.g. from an uploaded PDF)
    - Chunk them using the existing DocumentChunking logic
    - Concatenate chunk text into a single context
    - Ask the LLM to produce flashcards as JSON
    """

    def __init__(
        self,
        model_name: str = "gpt-4.1-mini",
        max_cards: int = 20,
    ) -> None:
        self.llm = ChatOpenAI(model=model_name)  # you can change this model name
        self.chunker = DocumentChunking()
        self.max_cards = max_cards

    def _build_prompt(self, text: str, num_cards: int) -> str:
        """
        Prompt the LLM to produce flashcards in a structured JSON format.
        """
        return f"""
You are a helpful study assistant.

Given the following document text, create at most {num_cards} high-quality flashcards.
Each flashcard must contain:
- a question (front of the card)
- a short, precise answer (back of the card)

Return ONLY valid JSON in exactly this format:

{{
  "cards": [
    {{"question": "Question 1?", "answer": "Answer 1."}},
    {{"question": "Question 2?", "answer": "Answer 2."}}
  ]
}}

Do not include explanations, comments, or any extra keys.

Document text:
\"\"\"{text}\"\"\"
"""

    def _parse_flashcards(self, raw: str) -> List[Dict[str, str]]:
        """
        Parse the JSON returned by the LLM into a list of {question, answer} dicts.
        """
        # Try to extract JSON between first '{' and last '}'
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = raw[start : end + 1]
        else:
            json_str = raw

        data = json.loads(json_str)
        cards_raw = data.get("cards", [])
        cards: List[Dict[str, str]] = []

        for item in cards_raw:
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            if q and a:
                cards.append({"question": q, "answer": a})

        return cards

    def generate_from_docs(
        self,
        documents: List[Document],
        max_cards: int | None = None,
    ) -> List[Dict[str, str]]:
        """
        High-level entry point:
        - chunk docs in-memory
        - merge chunk content
        - call LLM
        - parse flashcards

        Returns:
            List of dicts: [{"question": "...", "answer": "..."}, ...]
        """
        if not documents:
            return []

        # Use the in-memory chunking path so we don't write CSVs for each request
        chunks = self.chunker.create_chunks_in_memory(documents)
        if not chunks:
            return []

        combined_text = "\n\n".join(c.page_content for c in chunks)
        num_cards = max_cards or self.max_cards

        prompt = self._build_prompt(combined_text, num_cards)
        response = self.llm.invoke(prompt)

        # For ChatOpenAI, response is usually an object with `.content`
        if hasattr(response, "content"):
            raw_output = response.content
        else:
            raw_output = str(response)

        cards = self._parse_flashcards(raw_output)
        return cards
