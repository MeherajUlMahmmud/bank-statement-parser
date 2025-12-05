import logging
import json
from typing import Dict, Any

from ..services.groq_service import GroqService
from ..services.prompt_service import PromptService

logger = logging.getLogger(__name__)


class ExtractionAgent:
    """
    Agent 2: Structured Data Extraction Agent

    Receives cleaned OCR text and extracts structured JSON data including:
    - Account information
    - Statement period
    - Bank details
    - Balances
    - All transactions
    """

    def __init__(self):
        """Initialize the Extraction Agent."""
        self.groq_service = GroqService()
        self.prompt_service = PromptService()
        logger.info("ExtractionAgent initialized")

    async def process(self, cleaned_text: str) -> Dict[str, Any]:
        """
        Extract structured data from cleaned text.

        Args:
            cleaned_text: Cleaned text from Agent 1

        Returns:
            dict: {
                'extracted_data': dict,
                'success': bool,
                'error': str or None,
                'metadata': dict
            }
        """
        logger.info(f"Agent 2: Starting data extraction ({len(cleaned_text)} chars)")

        try:
            # Create prompt
            prompt = self.prompt_service.create_extraction_prompt(cleaned_text)

            # Call Groq API
            response = await self.groq_service.call_llm(prompt)

            if not response.get('success'):
                return {
                    'extracted_data': {},
                    'success': False,
                    'error': response.get('error_message', 'Unknown error'),
                    'metadata': {}
                }

            # Parse JSON from response
            content = response.get('content', '')
            extracted_data = self._parse_json_response(content)

            logger.info(
                f"Agent 2: Extraction complete - "
                f"{len(extracted_data.get('transactions', []))} transactions found"
            )

            return {
                'extracted_data': extracted_data,
                'success': True,
                'error': None,
                'metadata': {
                    'transactions_count': len(extracted_data.get('transactions', [])),
                    'tokens_used': response.get('tokens_used', {}),
                }
            }

        except Exception as e:
            logger.error(f"Agent 2: Error during extraction: {str(e)}")
            return {
                'extracted_data': {},
                'success': False,
                'error': str(e),
                'metadata': {}
            }

    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON from LLM response using robust brace matching."""
        # Find the first opening brace
        json_start = response_text.find('{')
        if json_start < 0:
            logger.warning("No opening brace found in response")
            return {}

        # Find matching closing brace by counting braces
        brace_count = 0
        json_end = -1
        for i in range(json_start, len(response_text)):
            if response_text[i] == '{':
                brace_count += 1
            elif response_text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_end = i + 1
                    break

        if json_end <= json_start:
            logger.warning("No matching closing brace found in response")
            return {}

        # Extract and parse JSON
        json_str = response_text[json_start:json_end]
        try:
            data = json.loads(json_str)
            return data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            logger.debug(f"Attempted to parse: {json_str[:200]}...")
            return {}
