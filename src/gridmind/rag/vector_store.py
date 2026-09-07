import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from qdrant_client import QdrantClient
from qdrant_client.http import models

logger = logging.getLogger(__name__)


class QdrantSettings(BaseSettings):
    """Pydantic Settings model that automatically reads configuration variables

    (only read not construct) from environment variables or a .env file with
    default fallbacks.
    """

    qdrant_host: str = Field(
        default="localhost", description="Qdrant server hostname"
    )
    qdrant_port: int = Field(default=6333, description="Qdrant REST API port")
    qdrant_url: Optional[str] = Field(
        default=None, description="Qdrant Cloud URL if applicable"
    )
    qdrant_api_key: Optional[str] = Field(
        default=None, description="API Key for Qdrant Cloud authentication"
    )
    collection_name: str = Field(
        default="energy_reports", description="Default vector collection name"
    )
    vector_size: int = Field(
        default=384,
        description="Embedding vector size (must match model output)",
    )

    # Σύνταξη για Pydantic v2 (αγνοεί extra μεταβλητές όπως το openai_api_key από το .env)
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class SearchResult(BaseModel):
    """Pydantic model representing a structured semantic search result returned

    to the RAG retriever module (Context of rag before the answer).
    """

    id: str = Field(description="Unique point identifier from Qdrant")
    score: float = Field(description="Similarity score (Cosine distance)")
    payload: Dict[str, Any] = Field(
        description="Metadata containing original text chunk, source, etc."
    )


class QdrantVectorStore:
    """Wrapper class around QdrantClient that manages collection

    initialization, vector point upserts, and semantic similarity queries.
    """

    def __init__(self, settings: Optional[QdrantSettings] = None):
        """Constructor method that initializes the Qdrant Client using Pydantic

        Settings.

        Args:
            settings (Optional[QdrantSettings]): Pydantic settings instance.
                                                  Instantiates default settings
                                                  if None.
        """
        self.settings = settings or QdrantSettings()

        # Determine whether to connect to remote Qdrant Cloud or local Docker instance
        if self.settings.qdrant_url:
            self.client = QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key,
            )
            logger.info(
                f"Connected to Qdrant Cloud at: {self.settings.qdrant_url}"
            )
        else:
            self.client = QdrantClient(
                host=self.settings.qdrant_host, port=self.settings.qdrant_port
            )
            logger.info(
                f"Connected to local Qdrant server at: {self.settings.qdrant_host}:{self.settings.qdrant_port}"
            )

        # Automatically check and initialize collection upon instance creation
        self.init_collection()

    def init_collection(self) -> None:
        """Verifies if the specified vector collection exists in Qdrant; creates

        a new collection if it does not exist.
        """
        try:
            collections = self.client.get_collections().collections
            exists = any(
                col.name == self.settings.collection_name for col in collections
            )

            if not exists:
                logger.info(
                    f"Creating collection '{self.settings.collection_name}' in Qdrant..."
                )
                self.client.create_collection(
                    collection_name=self.settings.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.settings.vector_size,  # Must match embedding model dimensions
                        distance=models.Distance.COSINE,  # Cosine similarity metric
                    ),
                )
                logger.info(
                    f"Collection '{self.settings.collection_name}' created successfully."
                )
            else:
                logger.info(
                    f"Collection '{self.settings.collection_name}' already exists."
                )

        except Exception as e:
            logger.error(f"Error during collection initialization: {e}")
            raise e

    def upsert_vectors(self, points: List[models.PointStruct]) -> None:
        """Inserts or updates vector points into the target Qdrant collection.

        Args:
            points (List[models.PointStruct]): List of Qdrant PointStruct items
              containing ID, vector embeddings, and metadata payload.
        """
        try:
            self.client.upsert(
                collection_name=self.settings.collection_name,
                points=points,
            )
            logger.info(
                f"Successfully upserted {len(points)} vector points into Qdrant."
            )
        except Exception as e:
            logger.error(f"Error upserting vectors: {e}")
            raise e

    def search_similar(
        self, query_vector: List[float], limit: int = 4
    ) -> List[SearchResult]:
        """Performs a cosine similarity search against the vector store using a

        query vector.

        Args:
            query_vector (List[float]): Generated embedding vector of the user's
              prompt.
            limit (int): Top-N most relevant document chunks to return. Defaults
              to 4.

        Returns:
            List[SearchResult]: List of validated Pydantic SearchResult objects.
        """
        try:
            raw_results = self.client.search(
                collection_name=self.settings.collection_name,
                query_vector=query_vector,
                limit=limit,
            )

            # Map raw Qdrant ScoredPoint items into strongly-typed Pydantic SearchResult models
            validated_results = [
                SearchResult(
                    id=str(res.id),
                    score=res.score,
                    payload=res.payload or {},
                )
                for res in raw_results
            ]

            return validated_results

        except Exception as e:
            logger.error(f"Error executing similarity search: {e}")
            raise e