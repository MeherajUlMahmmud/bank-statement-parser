import logging
import httpx
from fastapi import HTTPException
from app.schemas.chat import ChatRequest
from app.core.config import settings

logger = logging.getLogger(__name__)
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


async def get_groq_response(request_data: ChatRequest):
    """
    Sends a request to the Groq API and returns the response.
    """
    headers = {
        "Authorization": f"Bearer {settings.GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    # Convert the request to a properly formatted Groq API payload
    payload = {
        "model": settings.GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": request_data.message}
        ],
        "temperature": settings.MODEL_TEMP,
        "max_tokens": settings.MODEL_MAX_TOKEN
    }

    logger.info(
        f"Sending message to GroqCloud API with model: {settings.GROQ_MODEL}")
    logger.debug(f"Payload: {payload}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            logger.debug(f"Making POST request to {GROQ_API_URL}")
            response = await client.post(GROQ_API_URL, headers=headers, json=payload)

        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()

        response_data = response.json()
        logger.info(f"Response generated successfully from GroqCloud API")
        logger.debug(f"Response status code: {response.status_code}")

        if "usage" in response_data:
            logger.info(f"Token usage - Prompt: {response_data['usage'].get('prompt_tokens', 'N/A')}, "
                        f"Completion: {response_data['usage'].get('completion_tokens', 'N/A')}, "
                        f"Total: {response_data['usage'].get('total_tokens', 'N/A')}")

        return response_data

    except httpx.HTTPStatusError as e:
        logger.error(
            f"GroqCloud API returned error status: {e.response.status_code}")
        # Handle specific errors from the Groq API
        error_detail = e.response.json().get("error", {}).get(
            "message", "Unknown Groq API error")
        logger.error(f"Error detail: {error_detail}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Groq API error: {error_detail}"
        )
    except httpx.RequestError as e:
        logger.error(f"Network error connecting to GroqCloud API: {str(e)}")
        # Handle network-level errors
        raise HTTPException(
            status_code=503, detail=f"Could not connect to Groq API: {e}")
