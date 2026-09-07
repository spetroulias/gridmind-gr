import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from gridmind.rag.retriever import RAGRetriever
from gridmind.services.llm import LLMService

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Είσαι ο εξειδικευμένος AI Energy Analyst της πλατφόρμας GridMind GR.
Απάντησε στις ερωτήσεις του χρήστη με ακρίβεια, ευγένεια και αποκλειστικά στα Ελληνικά.

Χρησιμοποίησε αυστηρά το παρακάτω Context (αποσπάσματα από αναφορές του ΑΔΜΗΕ) για να διαμορφώσεις την απάντησή σου.
Αν η πληροφορία δεν υπάρχει στο Context, δήλωσε ευγενικά ότι δεν διαθέτεις τα απαραίτητα δεδομένα.

Context:
{context}
"""

class ChatResponse(BaseModel):
    answer: str = Field(description="Generated Greek response from OpenAI")
    sources: List[Dict[str, Any]] = Field(description="Source metadata for display")

class GridMindChatbot:
    """
    Direct RAG Orchestrator mapping user queries directly to Qdrant retrieval and OpenAI response synthesis.
    """
    def __init__(self, retriever: Optional[RAGRetriever] = None, llm_service: Optional[LLMService] = None):
        self.retriever = retriever or RAGRetriever()
        self.llm_service = llm_service or LLMService()

    def ask(self, question: str) -> ChatResponse:
        logger.info(f"Processing chat query: '{question}'")

        # Step 1: Retrieve context chunks from Qdrant
        retrieved_results = self.retriever.retrieve_context(query=question)

        # Step 2: Format context and build source metadata
        context_blocks = []
        sources_metadata = []

        for idx, res in enumerate(retrieved_results, 1):
            text_chunk = res.payload.get("text", "")
            source_file = res.payload.get("source", "Unknown")
            page_num = res.payload.get("page", 0)

            context_blocks.append(f"--- Απόσπασμα {idx} (Πηγή: {source_file}, Σελίδα: {page_num}) ---\n{text_chunk}")
            sources_metadata.append({
                "source": source_file,
                "page": page_num,
                "score": getattr(res, 'score', None)
            })

        formatted_context = "\n\n".join(context_blocks) if context_blocks else "Δεν βρέθηκαν σχετικές πληροφορίες στα έγγραφα του ΑΔΜΗΕ."

        # Step 3: LLM generation via OpenAI
        formatted_system_prompt = SYSTEM_PROMPT.format(context=formatted_context)
        answer_text = self.llm_service.generate_response(
            system_prompt=formatted_system_prompt,
            user_prompt=question
        )

        return ChatResponse(
            answer=answer_text,
            sources=sources_metadata
        )