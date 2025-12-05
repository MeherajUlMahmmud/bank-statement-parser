import logging
import json
from typing import Dict, Any

from ..services.groq_service import GroqService
from ..services.prompt_service import PromptService

logger = logging.getLogger(__name__)


class OCRCleanupAgent:
    """
    Agent 1: OCR Cleanup Agent

    Receives raw OCR text and cleans it up by:
    - Fixing common OCR errors
    - Removing noise and artifacts
    - Preserving table structure
    - Enhancing readability
    """

    def __init__(self):
        """Initialize the OCR Cleanup Agent."""
        self.groq_service = GroqService()
        self.prompt_service = PromptService()
        logger.info("OCRCleanupAgent initialized")

    async def process(self, raw_ocr_text: str) -> Dict[str, Any]:
        """
        Clean up raw OCR text.

        Args:
            raw_ocr_text: Raw text from OCR

        Returns:
            dict: {
                'cleaned_text': str,
                'success': bool,
                'error': str or None,
                'metadata': dict
            }
        """
        logger.info(f"Agent 1: Starting OCR cleanup ({len(raw_ocr_text)} chars)")

        try:
            # Create prompt
            prompt = self.prompt_service.create_ocr_cleanup_prompt(raw_ocr_text)

            # Call Groq API
            response = await self.groq_service.call_llm(prompt)

            if not response.get('success'):
                return {
                    'cleaned_text': '',
                    'success': False,
                    'error': response.get('error_message', 'Unknown error'),
                    'metadata': {}
                }

            cleaned_text = response.get('content', '')

            logger.info(f"Agent 1: Cleanup complete ({len(cleaned_text)} chars)")

            return {
                'cleaned_text': cleaned_text,
                'success': True,
                'error': None,
                'metadata': {
                    'input_length': len(raw_ocr_text),
                    'output_length': len(cleaned_text),
                    'tokens_used': response.get('tokens_used', {}),
                }
            }

        except Exception as e:
            logger.error(f"Agent 1: Error during cleanup: {str(e)}")
            return {
                'cleaned_text': '',
                'success': False,
                'error': str(e),
                'metadata': {}
            }
