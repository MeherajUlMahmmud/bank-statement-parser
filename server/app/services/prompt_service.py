import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)


class PromptService:
    """
    Service class for generating prompts for document processing.
    
    This class handles the creation of structured prompts for data extraction
    from various document types (bank statements, invoices, receipts, generic documents)
    using both OCR text and images. Supports both legacy document-specific prompts
    and modern canonical schema prompts.
    """

    def __init__(self):
        """Initialize the PromptService."""
        logger.info("PromptService initialized - Using canonical extraction prompts")

    # ============================================================================
    # Canonical Schema Extraction Prompts (Document Type Agnostic)
    # ============================================================================

    def create_canonical_extraction_prompt(
            self,
            document_type: str,
            include_confidence: bool = True,
            include_bbox: bool = True
    ) -> str:
        """
        Create a VLM prompt for extracting data into canonical JSON schema.
        
        Args:
            document_type: Type of document (bank_statement, invoice, receipt, generic)
            include_confidence: Whether to request confidence scores per field
            include_bbox: Whether to request bounding box coordinates
        
        Returns:
            str: VLM prompt for canonical extraction
        """
        if document_type == 'bank_statement':
            return self._create_bank_statement_canonical_prompt(include_confidence, include_bbox)
        elif document_type == 'invoice':
            return self._create_invoice_canonical_prompt(include_confidence, include_bbox)
        elif document_type == 'receipt':
            return self._create_receipt_canonical_prompt(include_confidence, include_bbox)
        else:
            return self._create_generic_canonical_prompt(include_confidence, include_bbox)

    def _create_bank_statement_canonical_prompt(self, include_confidence: bool, include_bbox: bool) -> str:
        """Create canonical extraction prompt for bank statements."""
        confidence_instruction = """
- For each field, provide a confidence score (0.0 to 1.0) indicating extraction certainty
- Confidence should reflect: text clarity, format consistency, field completeness
""" if include_confidence else ""

        bbox_instruction = """
- For each field, provide bounding box coordinates [x, y, width, height] in image pixels
- Coordinates should indicate where the field appears on the page
""" if include_bbox else ""

        sample_json = {
            "document_id": "<uuid>",
            "source_filename": "statement.pdf",
            "document_type": "bank_statement",
            "pages": [
                {
                    "page_number": 1,
                    "raw_text": "...",
                }
            ],
            "extractions": {
                "account": {
                    "account_number": {
                        "value": "XXXX1234",
                        "confidence": 0.92,
                        "page": 1,
                        "bbox": [100, 50, 200, 20] if include_bbox else None,
                    },
                    "account_holder": {"value": "John Doe", "confidence": 0.87}
                },
                "period": {
                    "start_date": {"value": "2025-01-01", "confidence": 0.95},
                    "end_date": {"value": "2025-01-31", "confidence": 0.94}
                },
                "transactions": [
                    {
                        "date": {"value": "2025-01-02", "confidence": 0.98},
                        "description": {"value": "ATM Withdrawal", "confidence": 0.93},
                        "debit": {"value": 2500.00, "currency": "BDT", "confidence": 0.98},
                        "credit": {"value": 0.00, "confidence": 0.98},
                        "balance": {"value": 15000.00, "confidence": 0.90},
                        "page": 2,
                        "bbox": [50, 200, 500, 30] if include_bbox else None,
                    }
                ],
                "summary": {
                    "opening_balance": {"value": 17500.00, "confidence": 0.95},
                    "closing_balance": {"value": 15000.00, "confidence": 0.95}
                }
            }
        }

        sample_json_str = json.dumps(sample_json, indent=2)

        return f"""
You are an expert document extractor. Analyze the provided document image and extract structured data following the canonical JSON schema.

EXTRACTION TASKS

1. DOCUMENT ANALYSIS
   - Identify the document type and structure
   - Locate key sections: header, transactions/items, summary/totals
   - Understand the layout (tables, key-value pairs, free text)

2. DATA EXTRACTION
   Extract all available information and map to canonical schema:
   - Account information (account number, holder name)
   - Statement period (start date, end date)
   - All transactions with: date, description, debit, credit, balance
   - Summary information (opening balance, closing balance)
   - Currency detection

3. FIELD-LEVEL METADATA{confidence_instruction}{bbox_instruction}

4. DATA NORMALIZATION
   - Dates: Convert to ISO 8601 format (YYYY-MM-DD)
   - Amounts: Remove commas, handle decimals, detect currency
   - Text: Clean whitespace, preserve original formatting where needed

OUTPUT FORMAT
Return ONLY valid JSON following this exact structure:

{sample_json_str}

JSON REQUIREMENTS:
- Use double quotes (") for all keys and string values
- Use "null" for missing values
- Numeric values should be numbers, not strings
- Dates must be ISO 8601 format (YYYY-MM-DD)
- Include confidence scores for each field (0.0 to 1.0)
- Include bbox coordinates [x, y, width, height] for each field if available
- Ensure valid JSON parseable by json.loads()

CRITICAL: Return ONLY the JSON object. No explanatory text before or after.
"""

    def _create_invoice_canonical_prompt(self, include_confidence: bool, include_bbox: bool) -> str:
        """Create canonical extraction prompt for invoices."""
        confidence_instruction = """
- For each field, provide a confidence score (0.0 to 1.0) indicating extraction certainty
""" if include_confidence else ""

        bbox_instruction = """
- For each field, provide bounding box coordinates [x, y, width, height] in image pixels
""" if include_bbox else ""

        sample_json = {
            "document_id": "<uuid>",
            "source_filename": "invoice.pdf",
            "document_type": "invoice",
            "pages": [{"page_number": 1, "raw_text": "..."}],
            "extractions": {
                "vendor": {
                    "vendor_name": {"value": "Acme Corp", "confidence": 0.95},
                    "vendor_address": {"value": "123 Business St", "confidence": 0.90}
                },
                "invoice_details": {
                    "invoice_number": {"value": "INV-2025-001", "confidence": 0.98},
                    "invoice_date": {"value": "2025-01-15", "confidence": 0.95},
                    "due_date": {"value": "2025-02-15", "confidence": 0.93}
                },
                "customer": {
                    "customer_name": {"value": "John Doe", "confidence": 0.92},
                    "customer_address": {"value": "456 Main St", "confidence": 0.88}
                },
                "line_items": [
                    {
                        "description": {"value": "Product A", "confidence": 0.95},
                        "quantity": {"value": 10, "confidence": 0.98},
                        "unit_price": {"value": 25.00, "currency": "USD", "confidence": 0.95},
                        "total": {"value": 250.00, "currency": "USD", "confidence": 0.95},
                        "page": 1,
                        "bbox": [50, 200, 500, 30] if include_bbox else None,
                    }
                ],
                "summary": {
                    "subtotal": {"value": 250.00, "currency": "USD", "confidence": 0.95},
                    "tax": {"value": 25.00, "currency": "USD", "confidence": 0.93},
                    "total": {"value": 275.00, "currency": "USD", "confidence": 0.95}
                }
            }
        }

        sample_json_str = json.dumps(sample_json, indent=2)

        return f"""
You are an expert invoice extractor. Analyze the provided invoice image and extract structured data following the canonical JSON schema.

EXTRACTION TASKS

1. INVOICE ANALYSIS
   - Identify vendor information (name, address, contact)
   - Locate invoice details (number, date, due date)
   - Extract customer information
   - Identify line items table
   - Find summary/totals section

2. DATA EXTRACTION
   Extract all available information:
   - Vendor details (name, address)
   - Invoice metadata (number, dates)
   - Customer information
   - All line items (description, quantity, unit price, total)
   - Summary (subtotal, tax, total, currency)

3. FIELD-LEVEL METADATA{confidence_instruction}{bbox_instruction}

4. DATA NORMALIZATION
   - Dates: Convert to ISO 8601 format (YYYY-MM-DD)
   - Amounts: Remove commas, handle decimals, detect currency
   - Numbers: Extract as numeric values

OUTPUT FORMAT
Return ONLY valid JSON following this exact structure:

{sample_json_str}

JSON REQUIREMENTS:
- Use double quotes (") for all keys and string values
- Use "null" for missing values
- Numeric values should be numbers, not strings
- Dates must be ISO 8601 format (YYYY-MM-DD)
- Include confidence scores for each field (0.0 to 1.0)
- Include bbox coordinates [x, y, width, height] for each field if available

CRITICAL: Return ONLY the JSON object. No explanatory text before or after.
"""

    def _create_receipt_canonical_prompt(self, include_confidence: bool, include_bbox: bool) -> str:
        """Create canonical extraction prompt for receipts."""
        confidence_instruction = """
- For each field, provide a confidence score (0.0 to 1.0) indicating extraction certainty
""" if include_confidence else ""

        bbox_instruction = """
- For each field, provide bounding box coordinates [x, y, width, height] in image pixels
""" if include_bbox else ""

        sample_json = {
            "document_id": "<uuid>",
            "source_filename": "receipt.pdf",
            "document_type": "receipt",
            "pages": [{"page_number": 1, "raw_text": "..."}],
            "extractions": {
                "merchant": {
                    "merchant_name": {"value": "Store ABC", "confidence": 0.95},
                    "merchant_address": {"value": "789 Shop St", "confidence": 0.88}
                },
                "receipt_details": {
                    "receipt_number": {"value": "RCP-12345", "confidence": 0.92},
                    "date": {"value": "2025-01-20", "confidence": 0.95},
                    "time": {"value": "14:30:00", "confidence": 0.90}
                },
                "items": [
                    {
                        "description": {"value": "Item 1", "confidence": 0.95},
                        "quantity": {"value": 2, "confidence": 0.90},
                        "price": {"value": 15.99, "currency": "USD", "confidence": 0.95},
                        "page": 1,
                        "bbox": [50, 150, 300, 25] if include_bbox else None,
                    }
                ],
                "payment": {
                    "payment_method": {"value": "Credit Card", "confidence": 0.93},
                    "total": {"value": 31.98, "currency": "USD", "confidence": 0.95},
                    "tax": {"value": 2.56, "currency": "USD", "confidence": 0.90}
                }
            }
        }

        sample_json_str = json.dumps(sample_json, indent=2)

        return f"""
You are an expert receipt extractor. Analyze the provided receipt image and extract structured data following the canonical JSON schema.

EXTRACTION TASKS

1. RECEIPT ANALYSIS
   - Identify merchant information (name, address)
   - Locate receipt details (number, date, time)
   - Extract purchased items list
   - Find payment information (method, total, tax)

2. DATA EXTRACTION
   Extract all available information:
   - Merchant details
   - Receipt metadata (number, date, time)
   - All purchased items (description, quantity, price)
   - Payment information (method, totals, tax, currency)

3. FIELD-LEVEL METADATA{confidence_instruction}{bbox_instruction}

4. DATA NORMALIZATION
   - Dates: Convert to ISO 8601 format (YYYY-MM-DD)
   - Times: Convert to 24-hour format (HH:MM:SS)
   - Amounts: Remove commas, handle decimals, detect currency
   - Numbers: Extract as numeric values

OUTPUT FORMAT
Return ONLY valid JSON following this exact structure:

{sample_json_str}

JSON REQUIREMENTS:
- Use double quotes (") for all keys and string values
- Use "null" for missing values
- Numeric values should be numbers, not strings
- Dates must be ISO 8601 format (YYYY-MM-DD)
- Include confidence scores for each field (0.0 to 1.0)
- Include bbox coordinates [x, y, width, height] for each field if available

CRITICAL: Return ONLY the JSON object. No explanatory text before or after.
"""

    def _create_generic_canonical_prompt(self, include_confidence: bool, include_bbox: bool) -> str:
        """Create canonical extraction prompt for generic documents."""
        confidence_instruction = """
- For each field, provide a confidence score (0.0 to 1.0) indicating extraction certainty
""" if include_confidence else ""

        bbox_instruction = """
- For each field, provide bounding box coordinates [x, y, width, height] in image pixels
""" if include_bbox else ""

        sample_json = {
            "document_id": "<uuid>",
            "source_filename": "document.pdf",
            "document_type": "generic",
            "pages": [{"page_number": 1, "raw_text": "..."}],
            "extractions": {
                "header": {
                    "title": {"value": "Document Title", "confidence": 0.90},
                    "date": {"value": "2025-01-20", "confidence": 0.85}
                },
                "key_value_pairs": [
                    {
                        "key": {"value": "Field Name", "confidence": 0.92},
                        "value": {"value": "Field Value", "confidence": 0.88},
                        "page": 1,
                        "bbox": [100, 200, 300, 20] if include_bbox else None,
                    }
                ],
                "tables": [
                    {
                        "table_data": [
                            {"row": 1, "columns": ["Header1", "Header2"], "confidence": 0.90}
                        ],
                        "page": 1,
                        "bbox": [50, 300, 500, 200] if include_bbox else None,
                    }
                ],
                "text_blocks": [
                    {
                        "text": {"value": "Paragraph text...", "confidence": 0.85},
                        "page": 1,
                        "bbox": [50, 500, 500, 100] if include_bbox else None,
                    }
                ]
            }
        }

        sample_json_str = json.dumps(sample_json, indent=2)

        return f"""
You are an expert document extractor. Analyze the provided document image and extract structured data following the canonical JSON schema.

EXTRACTION TASKS

1. DOCUMENT ANALYSIS
   - Identify document structure (headers, paragraphs, tables, key-value pairs)
   - Locate important information sections
   - Understand the layout and organization

2. DATA EXTRACTION
   Extract all available information:
   - Header information (title, dates, identifiers)
   - Key-value pairs (labels and their values)
   - Tables (if present) with row and column data
   - Text blocks (paragraphs, notes, descriptions)

3. FIELD-LEVEL METADATA{confidence_instruction}{bbox_instruction}

4. DATA NORMALIZATION
   - Dates: Convert to ISO 8601 format (YYYY-MM-DD) if detected
   - Numbers: Extract as numeric values where appropriate
   - Text: Preserve original formatting, clean whitespace

OUTPUT FORMAT
Return ONLY valid JSON following this exact structure:

{sample_json_str}

JSON REQUIREMENTS:
- Use double quotes (") for all keys and string values
- Use "null" for missing values
- Numeric values should be numbers, not strings
- Dates must be ISO 8601 format (YYYY-MM-DD) if present
- Include confidence scores for each field (0.0 to 1.0)
- Include bbox coordinates [x, y, width, height] for each field if available

CRITICAL: Return ONLY the JSON object. No explanatory text before or after.
"""
