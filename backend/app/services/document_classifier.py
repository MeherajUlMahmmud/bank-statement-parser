import json
import logging
from typing import Dict, Any, Optional
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from .groq_service import GroqService
from .ollama_service import OllamaService
from ..core.config import settings

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
                self.vlm_service = OllamaService()
                # Check if service is ready
                import asyncio
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                if hasattr(self.vlm_service, 'is_service_ready'):
                    is_ready = loop.run_until_complete(self.vlm_service.is_service_ready())
                    if not is_ready:
                        logger.warning("Ollama service not ready, falling back to GroqService")
                        self.vlm_service = GroqService()
            else:
                self.vlm_service = GroqService()

            logger.info(f"DocumentClassifier initialized with VLM service: {type(self.vlm_service).__name__}")
        except Exception as e:
            logger.error(f"Failed to initialize VLM service: {str(e)}")
            logger.info("Falling back to GroqService")
            self.vlm_service = GroqService()

    @staticmethod
    def image_to_data_url(image: Any) -> str:
        """
        Convert image to data URL for VLM processing.

        Args:
            image: Image (numpy array, PIL Image, or file path)

        Returns:
            str: Data URL (data:image/jpeg;base64,...)
        """
        try:
            # If it's a file path
            if isinstance(image, str):
                with open(image, 'rb') as f:
                    image_data = f.read()
                    base64_data = base64.b64encode(image_data).decode('utf-8')
                    # Detect mime type from extension
                    ext = image.lower().split('.')[-1]
                    mime_type = f"image/{ext}" if ext in ['png', 'jpeg', 'jpg', 'gif'] else "image/jpeg"
                    return f"data:{mime_type};base64,{base64_data}"

            # If it's a PIL Image
            elif hasattr(image, 'save'):
                buffer = BytesIO()
                image.save(buffer, format='JPEG')
                base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_data}"

            # If it's a numpy array
            elif isinstance(image, np.ndarray):
                pil_image = Image.fromarray(image)
                buffer = BytesIO()
                pil_image.save(buffer, format='JPEG')
                base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
                return f"data:image/jpeg;base64,{base64_data}"

            else:
                raise ValueError(f"Unsupported image type: {type(image)}")

        except Exception as e:
            logger.error(f"Error converting image to data URL: {str(e)}")
            raise

    async def classify_document(self, image: Any, document_type_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify document type from an image using VLM.

        Args:
            image: Image array (numpy array, PIL Image, or file path) - first page of the document
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
            image_data_url = self.image_to_data_url(image)

            # Create classification prompt
            prompt = self._create_classification_prompt()

            # Call VLM with image
            if isinstance(self.vlm_service, OllamaService):
                response_text = await self.vlm_service.process_with_image(prompt, image_data_url)
                response = {'success': True, 'content': response_text}
            else:
                # For GroqService, assume synchronous
                response = self.vlm_service.call_llm_with_image(prompt, image_data_url)

            if not response.get('success', True):
                logger.error(f"VLM classification failed: {response.get('error_message', 'Unknown error')}")
                return {
                    'document_type': 'generic',
                    'confidence': 0.0,
                    'reasoning': f'Classification failed: {response.get("error_message", "Unknown error")}'
                }

            # Parse response
            result = self._parse_classification_response(response.get('content', ''))

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
