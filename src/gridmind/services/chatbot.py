import logging
from typing import List
from pydantic import BaseModel

from gridmind.rag.retriever_db import DatabaseRetriever
from gridmind.rag.retriever import RAGRetriever
from gridmind.services.llm import LLMService

logger = logging.getLogger(__name__)


class ChatResponse(BaseModel):
    """Pydantic model representing the chatbot's response schema."""
    answer: str
    sources: List[str] = []


class GridMindChatbot:
    """Hybrid Orchestrator using PostgreSQL for Tabular Data & Groq LLM."""

    def __init__(self):
        self.llm = LLMService()
        self.vector_retriever = RAGRetriever()
        self.db_retriever = DatabaseRetriever()

    def answer_question(self, user_message: str = None, question: str = None, **kwargs) -> ChatResponse:
        query_text = user_message or question or ""
        
        context = ""
        sources = []

        # 1. Ανάκτηση από PostgreSQL μέσω του DatabaseRetriever (Tabular Data)
        try:
            db_context, db_sources = self.db_retriever.get_consumption_by_date(query_text)
            if db_context:
                context += f"\n[PostgreSQL System Load Data]: {db_context}\n"
                sources.extend(db_sources)
        except Exception as e:
            logger.error(f"Database retrieval error: {e}")

        # 2. Vector Search στο Qdrant μέσω της retrieve_context
        try:
            vector_results = self.vector_retriever.retrieve_context(query_text)
            if vector_results:
                for res in vector_results:
                    # Προσαρμογή στο SearchResult schema (res.text ή res.payload)
                    text_content = getattr(res, 'text', str(res))
                    context += f"\n[Document Chunk]: {text_content}\n"
                    sources.append("Qdrant Vector Index")
        except Exception as e:
            logger.warning(f"Qdrant query bypassed or empty: {e}")

        # 3. Prompting στο Groq LLM
        system_prompt = (
            "Είσαι ο GridMind AI Assistant. Απάντησε στην ερώτηση του χρήστη "
            "βασιζόμενος ΑΠΟΚΛΕΙΣΤΙΚΑ στο παρακάτω Context. Αν το context περιέχει "
            "μετρήσεις φορτίου/κατανάλωσης, ανέφερέ τις με ακρίβεια.\n\n"
            f"Context:\n{context if context else 'Δεν βρέθηκαν σχετικά δεδομένα.'}"
        )

        response_text = self.llm.generate_response(system_prompt, query_text)

        return ChatResponse(
            answer=response_text,
            sources=sources
        )

    ask = answer_question