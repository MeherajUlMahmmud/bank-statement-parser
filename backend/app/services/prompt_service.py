import json
import logging

logger = logging.getLogger(__name__)


class PromptService:
    """
    Service for generating prompts for the multi-agent bank statement processing pipeline.

    Provides prompts for:
    - Agent 1: OCR Cleanup (cleans raw OCR text)
    - Agent 2: Structured Data Extraction (extracts structured JSON from cleaned text)
    - Agent 3: Data Normalization & Validation (normalizes and validates extracted data)
    """

    def __init__(self):
        """Initialize the PromptService."""
        logger.info("PromptService initialized for bank statement processing")

    # ============================================================================
    # Agent 1: OCR Cleanup Agent
    # ============================================================================

    def create_ocr_cleanup_prompt(self, raw_ocr_text: str) -> str:
        """
        Create prompt for Agent 1: OCR Cleanup.

        This agent receives raw OCR text and cleans it up by:
        - Removing OCR noise and artifacts
        - Fixing common OCR errors (l→1, O→0, etc.)
        - Preserving table structure
        - Maintaining column alignment

        Args:
            raw_ocr_text: Raw text from OlmOCR

        Returns:
            str: Prompt for OCR cleanup
        """
        return f"""You are an expert OCR cleanup specialist for bank statements. Your task is to clean and fix the raw OCR text while preserving the original structure and layout.

RAW OCR TEXT:
```
{raw_ocr_text}
```

CLEANUP TASKS:

1. **Fix Common OCR Errors:**
   - Replace common character substitutions (l→1, O→0, S→5, etc.)
   - Fix broken words and spacing issues
   - Correct misread dates and numbers
   - Fix currency symbols and decimal points

2. **Preserve Structure:**
   - Maintain table alignment and columns
   - Keep transaction rows intact
   - Preserve headers and section labels
   - Keep date-description-amount groupings

3. **Remove Noise:**
   - Remove OCR artifacts (random characters, symbols)
   - Clean up extra whitespace while preserving alignment
   - Remove duplicate characters or lines
   - Fix line breaks that split data incorrectly

4. **Enhance Readability:**
   - Ensure dates are in consistent format
   - Align numbers properly
   - Separate sections clearly
   - Fix truncated or merged words

OUTPUT FORMAT:
Return ONLY the cleaned text. Do NOT add explanations or JSON. Just output the cleaned, structured text that maintains the original bank statement layout.

CRITICAL: Preserve all financial data (dates, amounts, descriptions) exactly - just fix the OCR errors. Do not modify or interpret the data.
"""

    # ============================================================================
    # Agent 2: Structured Data Extraction Agent
    # ============================================================================

    def create_extraction_prompt(self, cleaned_text: str) -> str:
        """
        Create prompt for Agent 2: Structured Data Extraction.

        This agent receives cleaned OCR text and extracts structured JSON data.

        Args:
            cleaned_text: Cleaned text from Agent 1

        Returns:
            str: Prompt for data extraction
        """
        sample_json = {
            "account": {
                "account_number": {"value": "XXXX1234", "confidence": 0.92},
                "account_holder": {"value": "John Doe", "confidence": 0.87},
                "account_type": {"value": "Savings", "confidence": 0.85}
            },
            "period": {
                "start_date": {"value": "2025-01-01", "confidence": 0.95},
                "end_date": {"value": "2025-01-31", "confidence": 0.94}
            },
            "bank": {
                "bank_name": {"value": "Example Bank", "confidence": 0.98},
                "branch_name": {"value": "Main Branch", "confidence": 0.90},
                "currency": {"value": "BDT", "confidence": 0.99}
            },
            "balances": {
                "opening_balance": {"value": 17500.00, "confidence": 0.95},
                "closing_balance": {"value": 15000.00, "confidence": 0.95},
                "total_debits": {"value": 5500.00, "confidence": 0.92},
                "total_credits": {"value": 3000.00, "confidence": 0.91}
            },
            "transactions": [
                {
                    "date": {"value": "2025-01-02", "confidence": 0.98},
                    "description": {"value": "ATM Withdrawal", "confidence": 0.93},
                    "debit": {"value": 2500.00, "confidence": 0.98},
                    "credit": {"value": 0.00, "confidence": 0.98},
                    "balance": {"value": 15000.00, "confidence": 0.90}
                },
                {
                    "date": {"value": "2025-01-05", "confidence": 0.97},
                    "description": {"value": "Salary Credit", "confidence": 0.95},
                    "debit": {"value": 0.00, "confidence": 0.98},
                    "credit": {"value": 50000.00, "confidence": 0.97},
                    "balance": {"value": 65000.00, "confidence": 0.92}
                }
            ]
        }

        sample_json_str = json.dumps(sample_json, indent=2)

        return f"""You are an expert data extractor for bank statements. Extract structured information from the cleaned bank statement text into JSON format.

CLEANED BANK STATEMENT TEXT:
```
{cleaned_text}
```

EXTRACTION TASKS:

1. **Account Information:**
   - Account number (mask if needed: XXXX1234)
   - Account holder name
   - Account type (Savings, Current, etc.)

2. **Statement Period:**
   - Start date (convert to YYYY-MM-DD)
   - End date (convert to YYYY-MM-DD)

3. **Bank Information:**
   - Bank name
   - Branch name
   - Currency code (BDT, USD, EUR, etc.)

4. **Summary Balances:**
   - Opening balance
   - Closing balance
   - Total debits (sum of all withdrawals)
   - Total credits (sum of all deposits)

5. **All Transactions:**
   For EACH transaction, extract:
   - Date (convert to YYYY-MM-DD format)
   - Description (transaction details)
   - Debit amount (0.00 if none)
   - Credit amount (0.00 if none)
   - Balance (running balance after transaction)

6. **Confidence Scores:**
   For each field, provide confidence (0.0 to 1.0) based on:
   - Text clarity and readability
   - Format consistency
   - Data completeness

OUTPUT FORMAT:
Return ONLY valid JSON following this structure:

{sample_json_str}

JSON REQUIREMENTS:
- Use double quotes (") for all keys and string values
- Dates must be ISO 8601 format (YYYY-MM-DD)
- Amounts must be numbers (not strings)
- Include confidence for every field
- Use 0.00 for missing debit/credit values
- Preserve original description text

CRITICAL: Return ONLY the JSON object. No explanatory text before or after.
"""

    # ============================================================================
    # Agent 3: Data Normalization & Validation Agent
    # ============================================================================

    def create_normalization_prompt(self, extracted_data: dict) -> str:
        """
        Create prompt for Agent 3: Data Normalization & Validation.

        This agent receives extracted JSON and normalizes/validates it.

        Args:
            extracted_data: Extracted data from Agent 2

        Returns:
            str: Prompt for normalization and validation
        """
        extracted_json_str = json.dumps(extracted_data, indent=2)

        return f"""You are an expert data validator for bank statements. Normalize and validate the extracted data to ensure accuracy and consistency.

EXTRACTED DATA:
```json
{extracted_json_str}
```

NORMALIZATION & VALIDATION TASKS:

1. **Date Validation:**
   - Verify all dates are valid and in YYYY-MM-DD format
   - Check that transaction dates fall within statement period
   - Ensure chronological ordering
   - Flag dates that seem incorrect

2. **Amount Validation:**
   - Verify all amounts are valid numbers
   - Check that debits/credits are not both non-zero for same transaction
   - Validate that running balance calculations are correct
   - Verify opening + credits - debits = closing balance
   - Flag any mathematical inconsistencies

3. **Currency Consistency:**
   - Ensure single currency throughout
   - Standardize to ISO 4217 code (BDT, USD, EUR, etc.)

4. **Data Normalization:**
   - Standardize date formats to ISO 8601
   - Remove extra spaces from descriptions
   - Normalize currency symbols to codes
   - Clean up formatting inconsistencies

5. **Balance Verification:**
   - Recalculate running balances from opening balance
   - Flag discrepancies between stated and calculated balances
   - Verify total debits and credits match transaction sums

6. **Confidence Adjustment:**
   - Increase confidence for validated fields
   - Decrease confidence for fields with inconsistencies
   - Add validation flags for problematic fields

OUTPUT FORMAT:
Return JSON with normalized data and validation results:

```json
{{
  "normalized_data": {{
    "account": {{ ... }},
    "period": {{ ... }},
    "bank": {{ ... }},
    "balances": {{ ... }},
    "transactions": [ ... ]
  }},
  "validation_results": {{
    "balance_verification": {{
      "calculated_closing": 15000.00,
      "stated_closing": 15000.00,
      "matches": true,
      "confidence": 0.98
    }},
    "date_validation": {{
      "all_dates_valid": true,
      "chronological": true,
      "within_period": true,
      "confidence": 0.95
    }},
    "amount_validation": {{
      "all_amounts_valid": true,
      "running_balance_correct": true,
      "confidence": 0.93
    }},
    "issues": [],
    "overall_confidence": 0.94
  }}
}}
```

CRITICAL:
- Return ONLY valid JSON
- Include both normalized_data and validation_results
- Flag all issues in the "issues" array
- Provide overall confidence score (0.0 to 1.0)
"""

    # ============================================================================
    # Helper Methods
    # ============================================================================

    def create_pipeline_summary_prompt(
            self,
            raw_ocr: str,
            cleaned_text: str,
            extracted_data: dict,
            normalized_data: dict
    ) -> str:
        """
        Create a summary prompt for reviewing the entire pipeline output.

        Args:
            raw_ocr: Original OCR text
            cleaned_text: Cleaned text from Agent 1
            extracted_data: Extracted data from Agent 2
            normalized_data: Normalized data from Agent 3

        Returns:
            str: Summary of the pipeline
        """
        return f"""
MULTI-AGENT PROCESSING PIPELINE SUMMARY

=== STAGE 1: RAW OCR ===
Length: {len(raw_ocr)} characters
Preview: {raw_ocr[:200]}...

=== STAGE 2: CLEANED TEXT ===
Length: {len(cleaned_text)} characters
Preview: {cleaned_text[:200]}...

=== STAGE 3: EXTRACTED DATA ===
Transactions found: {len(extracted_data.get('transactions', []))}
Account number: {extracted_data.get('account', {}).get('account_number', {}).get('value', 'N/A')}
Period: {extracted_data.get('period', {}).get('start_date', {}).get('value', 'N/A')} to {extracted_data.get('period', {}).get('end_date', {}).get('value', 'N/A')}

=== STAGE 4: NORMALIZED & VALIDATED ===
Overall Confidence: {normalized_data.get('validation_results', {}).get('overall_confidence', 0.0):.2%}
Issues Found: {len(normalized_data.get('validation_results', {}).get('issues', []))}
Balance Verified: {normalized_data.get('validation_results', {}).get('balance_verification', {}).get('matches', False)}

PIPELINE COMPLETE
"""
