import logging
from typing import List
from pydantic import BaseModel, Field
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logger to capture module execution and workflow events
logger = logging.getLogger(__name__)


class EmbeddingConfig(BaseModel):
    """
    Pydantic schema model defining setup parameters for the LangChain Hugging Face embedding service.
    """
    model_name: str = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        description="Multilingual Hugging Face model repository identifier supporting Greek language."
    )
    encode_kwargs: dict = Field(
        default={"normalize_embeddings": True},
        description="Keyword arguments passed to the encoder (e.g., normalizes vectors for Cosine Similarity)."
    )


class LangChainEmbeddingService:
    """
    Service class wrapping LangChain's HuggingFaceEmbeddings integration to generate 
    vector representations for single prompts and batch text documents.
    """

    def __init__(self, config: EmbeddingConfig = EmbeddingConfig()):
        """
        Constructor method initializing the LangChain HuggingFaceEmbeddings engine.

        Args:
            config (EmbeddingConfig): Pydantic configuration instance holding model settings.
        """
        self.config = config
        logger.info(f"Initializing LangChain Embedding Service with model: {self.config.model_name}...")

        # Initialize LangChain's native wrapper for Hugging Face sentence transformers
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.model_name,
            encode_kwargs=self.config.encode_kwargs
        )
        logger.info("LangChain Embedding Service initialized successfully.")

    def embed_query(self, text: str) -> List[float]:
        """
        Generates a vector embedding for a single query string (e.g., user question).

        Args:
            text (str): Input query text.

        Returns:
            List[float]: A 384-dimensional list of float numbers representing vector coordinates.
        """
        if not text or not text.strip():
            logger.warning("Empty string provided for query embedding. Returning empty vector list.")
            return []

        # LangChain's native method for single string queries
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generates vector embeddings for a batch list of document chunks concurrently.

        Args:
            texts (List[str]): List of document paragraphs/chunks to embed.

        Returns:
            List[List[float]]: List of 384-dimensional vector embeddings.
        """
        if not texts:
            logger.warning("Empty text list provided for document batch embedding. Returning empty list.")
            return []

        logger.info(f"Generating batch embeddings for {len(texts)} document chunks via LangChain...")
        
        # LangChain's native method for batch processing document chunks
        return self.embeddings.embed_documents(texts)