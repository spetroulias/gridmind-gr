import logging
import os
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from openai import OpenAI

# Configure logger for module execution tracking
logger = logging.getLogger(__name__)


class LLMSettings(BaseSettings):
    """
    Pydantic Settings model that automatically retrieves OpenAI configuration and credentials 
    from environment variables or a local .env file.
    """
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API authentication key")
    model_name: str = Field(default="gpt-4o-mini", description="Target OpenAI model identifier (e.g., gpt-4o, gpt-4o-mini)")
    temperature: float = Field(default=0.2, description="Sampling temperature (lower values ensure factual energy analysis)")
    max_tokens: int = Field(default=1024, description="Maximum token generation limit per response")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


class LLMService:
    """
    Wrapper class responsible for handling text generation requests using OpenAI's API.
    """

    def __init__(self, settings: Optional[LLMSettings] = None):
        """
        Constructor method initializing the OpenAI Client instance.
        """
        self.settings = settings or LLMSettings()
        
        # Resolve API Key from Pydantic settings or environment variables
        api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.warning("OPENAI_API_KEY is not set. LLM inference calls will fail until provided.")

        # Initialize official OpenAI client
        self.client = OpenAI(api_key=api_key)
        logger.info(f"Initialized LLMService with OpenAI model: {self.settings.model_name}")

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """
        Executes a chat completion query against the OpenAI Chat API.

        Args:
            system_prompt (str): Instructions defining system identity, role, and context constraints.
            user_prompt (str): The prompt containing the user query and retrieved context chunks.

        Returns:
            str: Generated text response from the OpenAI model.
        """
        try:
            logger.info(f"Sending chat completion request to OpenAI ({self.settings.model_name})...")
            
            response = self.client.chat.completions.create(
                model=self.settings.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )
            
            generated_text = response.choices[0].message.content
            logger.info("Successfully received LLM response from OpenAI.")
            return generated_text

        except Exception as e:
            logger.error(f"Error during OpenAI API text generation: {e}")
            raise e