import logging
import json
import copy
from typing import Dict, Any

from ..services.groq_service import GroqService
from ..services.prompt_service import PromptService
from ..services.normalization_service import NormalizationService

logger = logging.getLogger(__name__)


class NormalizationAgent:
    """
    Agent 3: Data Normalization & Validation Agent

    Receives extracted JSON and:
    - Normalizes dates and amounts
    - Validates data consistency
    - Verifies balance calculations
    - Provides confidence scores
    """

    def __init__(self):
        """Initialize the Normalization Agent."""
        self.groq_service = GroqService()
        self.prompt_service = PromptService()
        self.normalization_service = NormalizationService()
        logger.info("NormalizationAgent initialized")

    async def process(self, extracted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize and validate extracted data.

        Args:
            extracted_data: Extracted data from Agent 2

        Returns:
            dict: {
                'normalized_data': dict,
                'validation_results': dict,
                'success': bool,
                'error': str or None,
                'metadata': dict
            }
        """
        logger.info("Agent 3: Starting normalization and validation")

        try:
            # Create prompt
            prompt = self.prompt_service.create_normalization_prompt(extracted_data)

            # Call Groq API for validation logic
            response = await self.groq_service.call_llm(prompt)

            if not response.get('success'):
                return {
                    'normalized_data': {},
                    'validation_results': {},
                    'success': False,
                    'error': response.get('error_message', 'Unknown error'),
                    'metadata': {}
                }

            # Parse JSON from response
            content = response.get('content', '')
            result = self._parse_json_response(content)

            # Apply our own normalization as well
            normalized_data = result.get('normalized_data', extracted_data)
            normalized_data = self._apply_normalization(normalized_data)

            validation_results = result.get('validation_results', {})

            logger.info(
                f"Agent 3: Normalization complete - "
                f"Overall confidence: {validation_results.get('overall_confidence', 0):.2f}"
            )

            return {
                'normalized_data': normalized_data,
                'validation_results': validation_results,
                'success': True,
                'error': None,
                'metadata': {
                    'overall_confidence': validation_results.get('overall_confidence', 0),
                    'issues_count': len(validation_results.get('issues', [])),
                    'tokens_used': response.get('tokens_used', {}),
                }
            }

        except Exception as e:
            logger.error(f"Agent 3: Error during normalization: {str(e)}")
            return {
                'normalized_data': {},
                'validation_results': {},
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

    def _apply_normalization(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply additional normalization using our normalization service."""
        # Use deep copy to avoid mutating the original data structure
        normalized = copy.deepcopy(data)

        # Normalize transactions
        if 'transactions' in normalized:
            for transaction in normalized['transactions']:
                if 'date' in transaction and 'value' in transaction['date']:
                    normalized_date = self.normalization_service.normalize_date(transaction['date']['value'])
                    if normalized_date:
                        transaction['date']['value'] = normalized_date

        return normalized
