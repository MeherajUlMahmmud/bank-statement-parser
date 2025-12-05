import logging
from app.core.config import settings
from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest
from app.services import groq_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", operation_id="chat_with_groq_v1")
async def chat_with_groq(request: ChatRequest):
    """
    Sends a prompt to the GroqCloud API and returns the model's response.

    This endpoint acts as a secure proxy to Groq, handling the API key
    and request formatting for you.
    """
    logger.info(f"Received chat request with model: {settings.GROQ_MODEL}")

    try:
        response = await groq_service.get_groq_response(request)
        logger.info("Successfully received response from Groq service")
        return response
    except HTTPException as e:
        logger.error(f"HTTPException occurred: {e.status_code} - {e.detail}")
        # Re-raise HTTPExceptions from the service layer
        raise e
    except Exception as e:
        logger.exception(f"Unexpected error occurred: {str(e)}")
        # Catch any other unexpected errors
        raise HTTPException(
            status_code=500, detail=f"An unexpected error occurred: {e}")
