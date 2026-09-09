import logging
import os
from typing import Optional

from groq import Groq
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class LLMSettings(BaseSettings):
    """Pydantic Settings model for Groq API configuration."""

    groq_api_key: Optional[str] = Field(
        default=None, description="Groq API authentication key"
    )
    model_name: str = Field(
        default="llama-3.3-70b-versatile",
        description="Target Groq model identifier",
    )
    temperature: float = Field(
        default=0.2, description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=1024, description="Maximum token generation limit"
    )

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )


class LLMService:
    """Wrapper class responsible for handling text generation requests using Groq API."""

    def __init__(self, settings: Optional[LLMSettings] = None):
        self.settings = settings or LLMSettings()

        api_key = self.settings.groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY is not set.")

        self.client = Groq(api_key=api_key)
        
        # Εκτύπωση διαθέσιμων μοντέλων στο terminal για verification
        try:
            available_models = [m.id for m in self.client.models.list().data]
            logger.info(f"Available Groq models for your API key: {available_models}")
            if self.settings.model_name not in available_models and available_models:
                logger.warning(
                    f"Model {self.settings.model_name} not found in available models. "
                    f"Falling back to: {available_models[0]}"
                )
                self.settings.model_name = available_models[0]
        except Exception as e:
            logger.error(f"Could not fetch Groq models list: {e}")

    def generate_response(self, system_prompt: str, user_prompt: str) -> str:
        """Executes a chat completion query against Groq API."""
        try:
            logger.info(f"Sending request to Groq using model: {self.settings.model_name}")

            response = self.client.chat.completions.create(
                model=self.settings.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Error during Groq API text generation: {e}")
            raise e