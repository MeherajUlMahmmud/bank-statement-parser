import logging
from typing import Dict, Any, Optional
from groq import Groq

from ..core.config import settings

logger = logging.getLogger(__name__)


class GroqService:
    """
    Service for interacting with Groq API for LLM tasks.

    Used by agents for:
    - OCR cleanup
    - Data extraction
    - Normalization and validation
    """

    def __init__(
            self,
            api_key: Optional[str] = None,
            model: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None
    ):
        """
        Initialize Groq service.

        Args:
            api_key: Groq API key
            model: Model to use
            temperature: Temperature for generation
            max_tokens: Max tokens for completion
        """
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model = model or settings.GROQ_MODEL
        self.temperature = temperature if temperature is not None else settings.GROQ_TEMPERATURE
        self.max_tokens = max_tokens or settings.GROQ_MAX_TOKENS

        if not self.api_key:
            raise ValueError("GROQ_API_KEY is required")

        self.client = Groq(api_key=self.api_key)
        logger.info(f"GroqService initialized with model: {self.model}")

    async def call_llm(self, prompt: str) -> Dict[str, Any]:
        """
        Call Groq LLM with a prompt.

        Args:
            prompt: Prompt text

        Returns:
            dict: {
                'success': bool,
                'content': str,
                'error_message': str or None,
                'tokens_used': dict
            }
        """
        try:
            logger.debug(f"Calling Groq API with prompt length: {len(prompt)}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            content = response.choices[0].message.content
            tokens_used = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens,
            }

            logger.info(f"Groq API call successful - {tokens_used['total_tokens']} tokens used")

            return {
                'success': True,
                'content': content,
                'error_message': None,
                'tokens_used': tokens_used
            }

        except Exception as e:
            logger.error(f"Groq API call failed: {str(e)}")
            return {
                'success': False,
                'content': '',
                'error_message': str(e),
                'tokens_used': {}
            }
