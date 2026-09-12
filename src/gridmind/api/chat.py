import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from gridmind.services.chatbot import GridMindChatbot, ChatResponse
from gridmind.services.conversation import ChatContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chatbot"])

# Initialize single instance of chatbot service
chatbot_service = GridMindChatbot()

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000, description="User query message")
    context: ChatContext | None = None

@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
def chat_endpoint(request: ChatRequest):
    try:
        response = chatbot_service.ask(question=request.message, context=request.context)
        return response
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("Chat analytics failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat data service unavailable. Check database setup and server logs."
        )
