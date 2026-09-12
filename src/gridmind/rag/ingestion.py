import logging
import os
import hashlib
import uuid
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.http import models

# Import our custom standalone modules from the same package
from gridmind.rag.embeddings import LangChainEmbeddingService
from gridmind.rag.vector_store import QdrantVectorStore

# Configure logger for workflow tracking
logger = logging.getLogger(__name__)


class IngestionConfig(BaseModel):
    """
    Pydantic model defining configuration parameters for the PDF ingestion pipeline.
    """
    pdf_path: str = Field(description="Filepath to the target ADMIE PDF report")
    chunk_size: int = Field(default=500, description="Target character count per text chunk")
    chunk_overlap: int = Field(default=50, description="Overlapping character count between adjacent chunks")


class PDFIngestionPipeline:
    """
    Data engineering pipeline that reads PDF reports, splits them into semantic chunks using LangChain,
    generates embeddings, and stores points in Qdrant.
    """

    def __init__(self, vector_store: Optional[QdrantVectorStore] = None, embedding_service: Optional[LangChainEmbeddingService] = None):
        """
        Constructor method initializing the Vector Store and Embedding Service dependencies.
        """
        self.vector_store = vector_store or QdrantVectorStore()
        self.embedding_service = embedding_service or LangChainEmbeddingService()

    def process_and_ingest(self, config: IngestionConfig) -> None:
        """
        Executes the full ingestion workflow: Load PDF -> Split -> Embed -> Upsert to Qdrant.

        Args:
            config (IngestionConfig): Pydantic configuration with PDF path and chunking settings.
        """
        if not os.path.exists(config.pdf_path):
            logger.error(f"Target PDF file not found at: {config.pdf_path}")
            raise FileNotFoundError(f"PDF missing: {config.pdf_path}")

        # Step 1: Load PDF pages using LangChain's PyPDFLoader
        logger.info(f"Loading PDF document from: {config.pdf_path}...")
        loader = PyPDFLoader(config.pdf_path)
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} document pages.")

        # Step 2: Split text into overlapping semantic chunks #recursuve chunks : better splitter for semanitic meaning
        logger.info("Splitting text into overlapping semantic chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        logger.info(f"Generated {len(chunks)} text chunks.")

        # Step 3: Extract plain text strings and batch-generate embeddings via LangChain
        text_contents = [chunk.page_content for chunk in chunks]
        logger.info("Generating embeddings for all chunks...")
        embeddings = self.embedding_service.embed_documents(text_contents)

        # Step 4: Construct Qdrant PointStruct items
        document_digest = hashlib.sha256(Path(config.pdf_path).read_bytes()).hexdigest()
        points: List[models.PointStruct] = []
        for idx, (chunk, vector) in enumerate(zip(chunks, embeddings)):
            point = models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, document_digest + f":{idx}")),
                vector=vector,
                payload={
                    "text": chunk.page_content,
                    "page": chunk.metadata.get("page", 0),
                    "source": os.path.basename(config.pdf_path)
                }
            )
            points.append(point)

        # Step 5: Upsert points into Qdrant Vector DB
        logger.info(f"Upserting {len(points)} points into Qdrant...")
        self.vector_store.upsert_vectors(points)
        logger.info("PDF document successfully ingested into Qdrant!")
