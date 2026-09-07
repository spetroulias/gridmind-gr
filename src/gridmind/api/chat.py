from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from gridmind.services.chatbot import GridMindChatbot, ChatResponse

router = APIRouter(prefix="/chat", tags=["Chatbot"])

# Initialize single instance of chatbot service
chatbot_service = GridMindChatbot()

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User query message")

@router.post("", response_model=ChatResponse, status_code=status.HTTP_200_OK)
async def chat_endpoint(request: ChatRequest):
    """
    Direct RAG Chat endpoint querying ADMIE documentation in Qdrant and generating response via gpt-4o-mini.
    """
    try:
        response = chatbot_service.ask(question=request.message)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating chat response: {str(e)}"
        )