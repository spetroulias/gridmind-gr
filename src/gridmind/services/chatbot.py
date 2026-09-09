import logging
import re
from typing import List, Optional
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

    def _extract_date(self, text: str) -> Optional[str]:
        """Parses D/M/YY, D/M/YYYY or YYYY-MM-DD and normalizes to YYYY-MM-DD."""
        match = re.search(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})', text)
        if match:
            day, month, year = match.groups()
            if len(year) == 2:
                year = f"20{year}"
            return f"{year}-{int(month):02d}-{int(day):02d}"
        return None

    def answer_question(self, user_message: str = None, question: str = None, **kwargs) -> ChatResponse:
        # Δέχεται είτε 'user_message' είτε 'question'
        query_text = user_message or question or ""
        
        context = ""
        sources = []

        # 1. Parsing ημερομηνίας & Query στην PostgreSQL (system_load)
        extracted_date = self._extract_date(query_text)
        if extracted_date:
            logger.info(f"Target date parsed from query: {extracted_date}")
            db_data = self.db_retriever.get_consumption_by_date(extracted_date)
            if db_data:
                context += f"\n[PostgreSQL System Load Data]: {db_data}\n"
                sources.append(f"PostgreSQL Table: system_load ({extracted_date})")

        # 2. Vector Search στο Qdrant (Unstructured Context)
        try:
            vector_docs = self.vector_retriever.retrieve(query_text)
            if vector_docs:
                for doc in vector_docs:
                    context += f"\n[Document Chunk]: {doc.page_content}\n"
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

    # Alias για συμβατότητα με το API router (chat.py)
    ask = answer_question