import os
import logging
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not set in environment variables or .env file.")
            
        self.client = Groq(api_key=self.api_key)
        # Χρήση του τρέχοντος ενεργού μοντέλου παραγωγής στη Groq
        self.model = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")

    def generate_response(self, system_prompt: str, user_message: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API Error με το {self.model}: {e}")
            
            # Ενεργό fallback μοντέλο
            fallback_model = "openai/gpt-oss-20b"
            logger.info(f"Retrying με fallback model: {fallback_model}")
            
            response = self.client.chat.completions.create(
                model=fallback_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.2,
            )
            return response.choices[0].message.content