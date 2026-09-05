import logging
from typing import List, Optional
from pydantic import BaseModel, Field

from gridmind.rag.embeddings import LangChainEmbeddingService
from gridmind.rag.vector_store import QdrantVectorStore, SearchResult

# Configure logger for module tracking
logger = logging.getLogger(__name__)


class RetrieverConfig(BaseModel):
    """
    Pydantic schema defining retrieval settings.
    """
    top_k: int = Field(default=4, description="Number of most relevant text chunks to retrieve from Qdrant")


class RAGRetriever:
    """
    Retriever module that connects embedding generation with vector similarity search.
    """

    def __init__(
        self, 
        vector_store: Optional[QdrantVectorStore] = None, 
        embedding_service: Optional[LangChainEmbeddingService] = None,
        config: RetrieverConfig = RetrieverConfig()
    ):
        """
        Constructor method initializing dependencies.
        """
        self.config = config
        self.vector_store = vector_store or QdrantVectorStore()
        self.embedding_service = embedding_service or LangChainEmbeddingService()

    def retrieve_context(self, query: str) -> List[SearchResult]:
        """
        Executes end-to-end vector retrieval for a given text prompt.

        Args:
            query (str): User question in Greek or English.

        Returns:
            List[SearchResult]: Validated list of top-K relevant SearchResult models.
        """
        if not query or not query.strip():
            logger.warning("Received empty query for context retrieval. Returning empty list.")
            return []

        logger.info(f"Generating query embedding for prompt: '{query}'...")
        # Step 1: Convert string query to 384-dimensional vector using LangChain embedding service
        query_vector = self.embedding_service.embed_query(query)

        logger.info(f"Searching Qdrant for top {self.config.top_k} matching chunks...")
        # Step 2: Search Qdrant vector database using cosine similarity
        results = self.vector_store.search_similar(query_vector=query_vector, limit=self.config.top_k)

        logger.info(f"Retrieved {len(results)} relevant documents from vector store.")
        return results