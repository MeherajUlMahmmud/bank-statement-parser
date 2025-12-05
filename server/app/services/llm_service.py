import base64
import json
import logging
import asyncio
from typing import Dict, Optional, List, Any
import httpx
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# Pydantic models for response validation
class CustomerDetails(BaseModel):
    account_holder_name: Optional[str] = None
    account_number: Optional[str] = None


class BankDetails(BaseModel):
    bank_name: Optional[str] = None
    statement_period_start: Optional[str] = None
    statement_period_end: Optional[str] = None
    opening_balance: Optional[float] = None
    closing_balance: Optional[float] = None
    currency: Optional[str] = None


class PageInfo(BaseModel):
    appears_to_be_first_page: bool
    appears_to_be_last_page: bool
    has_header_info: bool
    has_transactions: bool


class BankStatementExtraction(BaseModel):
    customer_details: CustomerDetails
    bank_details: BankDetails
    transaction_columns: List[str]
    transactions: List[Dict[str, Any]]
    page_info: PageInfo
    confidence_scores: Optional[Dict[str, float]] = Field(
        default=None,
        description="Confidence scores (0.0-1.0) for extracted fields"
    )


class GroqCloudService:
    """
    Responsible for sending images to GroqCloud vision model and extracting structured data
    """

    SYSTEM_PROMPT = """You are an expert financial document analysis assistant specialized in bank statement data extraction. Your core responsibilities:

- Analyze bank statement images with high precision and accuracy
- Extract financial data while preserving the exact formatting, column names, and structure from the original document
- Handle diverse bank statement formats from various financial institutions worldwide
- Return only valid JSON without markdown code blocks or explanatory text
- Use null for any unclear, missing, or unreadable values - never guess or infer

Remember: Accuracy and fidelity to the source document are paramount."""

    EXTRACTION_PROMPT = """Extract all data from this bank statement image into the following JSON structure:

{
    "customer_details": {
        "account_holder_name": "string or null",
        "account_number": "string or null"
    },
    "bank_details": {
        "bank_name": "string or null",
        "statement_period_start": "YYYY-MM-DD or null",
        "statement_period_end": "YYYY-MM-DD or null",
        "opening_balance": number or null,
        "closing_balance": number or null,
        "currency": "string - currency name or ISO code (e.g., 'Taka', 'BDT', 'USD', 'Dollar', 'Euro', 'Rupee') or null"
    },
    "transaction_columns": ["exact", "column", "headers", "from", "statement"],
    "transactions": [
        {
            "ColumnName1": "value",
            "ColumnName2": 123.45,
            "ColumnName3": null
        }
    ],
    "page_info": {
        "appears_to_be_first_page": boolean,
        "appears_to_be_last_page": boolean,
        "has_header_info": boolean,
        "has_transactions": boolean
    },
    "confidence_scores": {
        "overall": 0.95,
        "customer_details": 0.98,
        "bank_details": 0.92,
        "transactions": 0.96
    }
}

CRITICAL RULES:
1. **transaction_columns**: List EXACT column headers as they appear in the transaction table (e.g., ["Date", "Particulars", "Withdrawals", "Deposits", "Balance"])
   - Preserve case sensitivity and spacing
   - This may vary per bank statement

2. **transactions**: Each object must use keys from transaction_columns
   - Extract ALL visible transactions on the page
   - Use numbers for amounts, null for empty cells
   - Preserve complete text for description/narration fields
   - Keep date strings as-is (don't reformat)
   - Include any Dr/Cr or +/- indicators if present

3. **currency**: Extract the EXACT currency as written on the statement
   - Look for currency symbols (৳, $, €, £, ₹, ¥)
   - Look for currency names (Taka, Dollar, Euro, Rupee, Pound)
   - Look for ISO codes (BDT, USD, EUR, INR, GBP)
   - Check near amounts, headers, or "Currency:" labels
   - Return exactly what you see (e.g., "Taka", "BDT", "৳", "Dollar", "USD")
   - DO NOT convert or standardize - just extract the exact text

4. **statement_period_start and statement_period_end**: Extract the statement period dates
   - Look for labels like "Statement Period:", "From:", "To:", "Period:", "Statement Date:"
   - Look in the header section of the first page
   - Common formats: "01-Jan-2024", "2024-01-01", "January 1, 2024"
   - Convert to YYYY-MM-DD format (e.g., "2024-01-01")
   - If only one date is shown, use it for both start and end
   - If no period is found, set to null

5. **confidence_scores**: Rate your confidence (0.0-1.0) for:
   - overall: Overall confidence in the extraction
   - customer_details: Confidence in customer information
   - bank_details: Confidence in bank/statement metadata
   - transactions: Confidence in transaction data accuracy

6. **Preserve original formatting**:
   - Column names exactly as shown (case-sensitive)
   - Don't standardize or rename columns
   - Don't infer missing values

7. Return ONLY the JSON object, no explanations or markdown.

EXAMPLES:

Example 1 - Standard format with Debit/Credit columns:
{
    "customer_details": {"account_holder_name": "John Doe", "account_number": "1234567890"},
    "bank_details": {"bank_name": "State Bank", "statement_period_start": "2024-01-01", "statement_period_end": "2024-01-31", "opening_balance": 5000.00, "closing_balance": 4850.50, "currency": "USD"},
    "transaction_columns": ["Date", "Description", "Debit", "Credit", "Balance"],
    "transactions": [
        {"Date": "2024-01-05", "Description": "ATM Withdrawal", "Debit": 100.00, "Credit": null, "Balance": 4900.00},
        {"Date": "2024-01-10", "Description": "Salary Credit", "Debit": null, "Credit": 2000.00, "Balance": 6900.00}
    ],
    "page_info": {"appears_to_be_first_page": true, "appears_to_be_last_page": false, "has_header_info": true, "has_transactions": true},
    "confidence_scores": {"overall": 0.97, "customer_details": 0.99, "bank_details": 0.96, "transactions": 0.98}
}

Example 2 - Alternative format with Withdrawals/Deposits:
{
    "customer_details": {"account_holder_name": "Jane Smith", "account_number": "9876543210"},
    "bank_details": {"bank_name": "National Bank", "statement_period_start": "2024-02-01", "statement_period_end": "2024-02-29", "opening_balance": 10000.00, "closing_balance": 9500.00, "currency": "EUR"},
    "transaction_columns": ["Transaction Date", "Particulars", "Withdrawals", "Deposits", "Running Balance"],
    "transactions": [
        {"Transaction Date": "01-Feb-2024", "Particulars": "Online Transfer", "Withdrawals": 500.00, "Deposits": null, "Running Balance": 9500.00}
    ],
    "page_info": {"appears_to_be_first_page": true, "appears_to_be_last_page": true, "has_header_info": true, "has_transactions": true},
    "confidence_scores": {"overall": 0.94, "customer_details": 0.97, "bank_details": 0.93, "transactions": 0.95}
}

Example 3 - Bangladesh Bank Statement with Taka currency:
{
    "customer_details": {"account_holder_name": "আব্দুল করিম", "account_number": "1234567890123"},
    "bank_details": {"bank_name": "Bangladesh Bank", "statement_period_start": "2024-01-01", "statement_period_end": "2024-01-31", "opening_balance": 50000.00, "closing_balance": 48500.00, "currency": "Taka"},
    "transaction_columns": ["Date", "Description", "Withdrawal", "Deposit", "Balance"],
    "transactions": [
        {"Date": "01-Jan-2024", "Description": "ATM Cash Withdrawal", "Withdrawal": 5000.00, "Deposit": null, "Balance": 45000.00},
        {"Date": "15-Jan-2024", "Description": "Salary Credit", "Withdrawal": null, "Deposit": 30000.00, "Balance": 75000.00}
    ],
    "page_info": {"appears_to_be_first_page": true, "appears_to_be_last_page": false, "has_header_info": true, "has_transactions": true},
    "confidence_scores": {"overall": 0.96, "customer_details": 0.98, "bank_details": 0.95, "transactions": 0.97}
}"""

    def __init__(self, config):
        self.api_key = config['GROQ_API_KEY']
        self.model = config['GROQ_MODEL']
        self.max_tokens = config['GROQ_MAX_TOKENS']
        self.timeout = config['GROQ_TIMEOUT']
        self.retry_attempts = config['RETRY_ATTEMPTS']
        self.retry_delay = config['RETRY_DELAY']
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.total_tokens_used = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0

    async def extract_from_image_async(self, image_path: str, page_number: int) -> Dict:
        """
        Send image to Groq and extract structured data (async version)

        Args:
            image_path: Path to the image file
            page_number: Page number for context

        Returns:
            Validated and parsed JSON response from LLM with token usage metadata
        """
        logger.info(f"Processing page {page_number}: {image_path}")

        # Encode image to base64
        image_base64 = await asyncio.to_thread(self._encode_image_base64, image_path)

        # Build messages
        messages = self._build_messages(image_base64, page_number)

        # Call API with retry
        response_data = await self._call_api_with_retry_async(messages, page_number)

        # Log token usage
        usage = response_data.get('usage', {})
        prompt_tokens = usage.get('prompt_tokens', 0)
        completion_tokens = usage.get('completion_tokens', 0)
        total_tokens = usage.get('total_tokens', 0)

        self.total_tokens_used += total_tokens
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        logger.info(
            f"Page {page_number} token usage - "
            f"Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}"
        )
        logger.info(
            f"Cumulative token usage - "
            f"Prompt: {self.total_prompt_tokens}, Completion: {self.total_completion_tokens}, "
            f"Total: {self.total_tokens_used}"
        )

        # Extract and parse JSON from response
        try:
            content = response_data['choices'][0]['message']['content']

            # Try to extract JSON if wrapped in markdown
            if '```json' in content:
                json_start = content.find('```json') + 7
                json_end = content.find('```', json_start)
                content = content[json_start:json_end].strip()
            elif '```' in content:
                json_start = content.find('```') + 3
                json_end = content.find('```', json_start)
                content = content[json_start:json_end].strip()

            parsed = json.loads(content)

            # Validate with Pydantic
            try:
                validated = BankStatementExtraction(**parsed)
                logger.info(f"Successfully extracted and validated data from page {page_number}")

                # Log confidence scores if available
                if validated.confidence_scores:
                    logger.info(f"Page {page_number} confidence scores: {validated.confidence_scores}")

                # Return as dict with metadata
                result = validated.model_dump()
                result['_metadata'] = {
                    'page_number': page_number,
                    'tokens_used': total_tokens,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens
                }
                return result

            except ValidationError as e:
                logger.warning(f"Page {page_number} validation errors: {e}")
                logger.warning("Returning unvalidated data - please review")
                # Return unvalidated data with warning
                parsed['_validation_errors'] = str(e)
                parsed['_metadata'] = {
                    'page_number': page_number,
                    'tokens_used': total_tokens,
                    'prompt_tokens': prompt_tokens,
                    'completion_tokens': completion_tokens,
                    'validated': False
                }
                return parsed

        except json.JSONDecodeError as e:
            logger.error(f"Page {page_number}: Failed to parse JSON response: {e}")
            logger.error(f"Raw response: {content}")
            raise ValueError(f"Page {page_number}: LLM returned invalid JSON: {e}")

    def extract_from_image(self, image_path: str, page_number: int) -> Dict:
        """
        Synchronous wrapper for extract_from_image_async

        Args:
            image_path: Path to the image file
            page_number: Page number for context

        Returns:
            Validated and parsed JSON response from LLM
        """
        return asyncio.run(self.extract_from_image_async(image_path, page_number))

    def _encode_image_base64(self, image_path: str) -> str:
        """Encode image to base64 string"""
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def _build_messages(self, image_base64: str, page_number: int) -> list:
        """Build the messages array for the API request with system prompt"""
        return [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Page {page_number} of a bank statement.\n\n{self.EXTRACTION_PROMPT}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    }
                ]
            }
        ]

    async def _call_api_with_retry_async(self, messages: list, page_number: int) -> Dict:
        """Call API with exponential backoff retry (async version)"""
        last_error = None

        for attempt in range(self.retry_attempts):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    response = await client.post(
                        self.base_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": self.model,
                            "messages": messages,
                            "max_tokens": self.max_tokens,
                            "temperature": 0.1  # Low temperature for more deterministic output
                        }
                    )

                    response.raise_for_status()
                    return response.json()

            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:  # Rate limit
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Page {page_number}: Rate limited, waiting {wait_time}s "
                        f"before retry {attempt + 1}/{self.retry_attempts}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                elif e.response.status_code >= 500:  # Server error
                    wait_time = self.retry_delay * (2 ** attempt)
                    logger.warning(
                        f"Page {page_number}: Server error, waiting {wait_time}s "
                        f"before retry {attempt + 1}/{self.retry_attempts}"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        f"Page {page_number}: HTTP error: {e.response.status_code} - {e.response.text}"
                    )
                    raise

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(f"Page {page_number}: Timeout, retry {attempt + 1}/{self.retry_attempts}")
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(self.retry_delay)
                    continue
                raise

            except Exception as e:
                logger.exception(f"Page {page_number}: Unexpected error calling Groq API: {e}")
                raise

        # If we exhausted all retries
        raise Exception(
            f"Page {page_number}: Failed after {self.retry_attempts} attempts. Last error: {last_error}"
        )

    def get_token_usage(self) -> Dict[str, int]:
        """
        Get cumulative token usage statistics

        Returns:
            Dict with prompt_tokens, completion_tokens, and total_tokens
        """
        return {
            'prompt_tokens': self.total_prompt_tokens,
            'completion_tokens': self.total_completion_tokens,
            'total_tokens': self.total_tokens_used
        }

    def reset_token_usage(self):
        """Reset token usage counters"""
        self.total_tokens_used = 0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
