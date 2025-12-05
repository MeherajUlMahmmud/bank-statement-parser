import json
import logging
from typing import Dict, Any, Optional

from document_control.helper.helper import Helper
from document_control.services.llm_service import GroqService, OllamaService, LLMProvider
from base import settings

logger = logging.getLogger(__name__)


class DocumentClassifier:
    """
    Service for classifying document types using VLM (Vision Language Model).
    
    Analyzes the first page of a document image to determine its type:
    - bank_statement
    - invoice
    - receipt
    - generic
    """

    DOCUMENT_TYPES = ['bank_statement', 'invoice', 'receipt', 'generic']

    def __init__(self):
        """Initialize the document classifier with VLM service."""
        use_ollama = getattr(settings, 'USE_OLLAMA', False)

        try:
            if use_ollama:
                model_name = getattr(settings, 'OLLAMA_MODEL_NAME', 'deepseek-r1:7b')
                base_url = getattr(settings, 'OLLAMA_BASE_URL', 'http://localhost:11434')
                self.vlm_service = OllamaService(model_name=str(model_name), base_url=str(base_url))
                if hasattr(self.vlm_service, 'is_service_ready') and not self.vlm_service.is_service_ready():
                    logger.warning("Ollama service not ready, falling back to GroqService")
                    self.vlm_service = GroqService()
            else:
                self.vlm_service = GroqService()

            logger.info(f"DocumentClassifier initialized with VLM service: {type(self.vlm_service).__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize VLM service: {str(e)}")
            logger.info("Falling back to GroqService")
            self.vlm_service = GroqService()

    def classify_document(self, image: Any, document_type_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify document type from an image using VLM.
        
        Args:
            image: Image array (numpy array or PIL Image) - first page of the document
            document_type_hint: Optional hint from user (if provided, may skip classification)
        
        Returns:
            dict: {
                'document_type': str,  # One of: bank_statement, invoice, receipt, generic
                'confidence': float,   # Confidence score (0.0 to 1.0)
                'reasoning': str       # Brief explanation of classification
            }
        """
        if document_type_hint and document_type_hint in self.DOCUMENT_TYPES:
            logger.info(f"Using user-provided document type hint: {document_type_hint}")
            return {
                'document_type': document_type_hint,
                'confidence': 1.0,
                'reasoning': f'User-specified document type: {document_type_hint}'
            }

        logger.info("Classifying document type using VLM")

        try:
            # Convert image to data URL
            image_data_url = Helper.image_to_data_url(image)

            # Create classification prompt
            prompt = self._create_classification_prompt()

            # Call VLM with image
            response = self.vlm_service.call_llm_with_image(prompt, image_data_url)

            if not response.success:
                logger.error(f"VLM classification failed: {response.error_message}")
                return {
                    'document_type': 'generic',
                    'confidence': 0.0,
                    'reasoning': f'Classification failed: {response.error_message}'
                }

            # Parse response
            result = self._parse_classification_response(response.content)

            logger.info(f"Document classified as: {result['document_type']} (confidence: {result['confidence']:.2f})")
            return result

        except Exception as e:
            logger.exception(f"Error classifying document: {str(e)}")
            return {
                'document_type': 'generic',
                'confidence': 0.0,
                'reasoning': f'Classification error: {str(e)}'
            }

    def _create_classification_prompt(self) -> str:
        """
        Create a prompt for document type classification.
        
        Returns:
            str: Classification prompt with few-shot examples
        """
        return """
You are an expert document classifier. Analyze the provided document image and determine its type.

DOCUMENT TYPES:
1. bank_statement - Bank account statements showing transactions, balances, account details
2. invoice - Bills or invoices from vendors/suppliers with line items, totals, due dates
3. receipt - Purchase receipts with items, prices, payment information
4. generic - Any other document type (forms, letters, contracts, etc.)

CLASSIFICATION TASK:
- Analyze the visual layout, text content, and structure of the document
- Identify key indicators (e.g., "Account Statement", "Invoice #", "Receipt", transaction tables)
- Classify the document into one of the four types above

OUTPUT FORMAT:
Return ONLY a valid JSON object with this exact structure:
{
  "document_type": "bank_statement" | "invoice" | "receipt" | "generic",
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation of why this type was chosen"
}

FEW-SHOT EXAMPLES:

Example 1 (Bank Statement):
{
  "document_type": "bank_statement",
  "confidence": 0.95,
  "reasoning": "Document contains account number, statement period, transaction table with dates/descriptions/debits/credits/balances"
}

Example 2 (Invoice):
{
  "document_type": "invoice",
  "confidence": 0.92,
  "reasoning": "Document has invoice number, vendor details, line items with quantities/prices, subtotal, tax, total amount due"
}

Example 3 (Receipt):
{
  "document_type": "receipt",
  "confidence": 0.88,
  "reasoning": "Document shows purchase date, store name, itemized list of purchased items with prices, payment method, total paid"
}

CRITICAL: Return ONLY the JSON object. No explanatory text before or after.
"""

    def _parse_classification_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the VLM response into structured classification result.
        
        Args:
            response_text: Raw response from VLM
        
        Returns:
            dict: Parsed classification result
        """
        default_result = {
            'document_type': 'generic',
            'confidence': 0.0,
            'reasoning': 'Failed to parse classification response'
        }

        if not response_text:
            logger.warning("Empty classification response")
            return default_result

        try:
            # Extract JSON from response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start < 0 or json_end <= json_start:
                logger.warning("No valid JSON found in classification response")
                return default_result

            data = json.loads(response_text[json_start:json_end])

            # Validate and extract fields
            document_type = data.get('document_type', 'generic')
            if document_type not in self.DOCUMENT_TYPES:
                logger.warning(f"Invalid document_type '{document_type}', defaulting to 'generic'")
                document_type = 'generic'

            confidence = float(data.get('confidence', 0.0))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to [0, 1]

            reasoning = data.get('reasoning', 'No reasoning provided')

            return {
                'document_type': document_type,
                'confidence': confidence,
                'reasoning': reasoning
            }

        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in classification response: {str(e)}")
            logger.debug(f"Response text: {response_text}")
            return default_result
        except Exception as e:
            logger.exception(f"Error parsing classification response: {str(e)}")
            return default_result
