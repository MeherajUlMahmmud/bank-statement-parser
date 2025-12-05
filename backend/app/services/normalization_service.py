import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime
from decimal import Decimal, InvalidOperation

logger = logging.getLogger(__name__)


class NormalizationService:
    """
    Service for normalizing extracted data to canonical formats.
    
    Handles:
    - Date normalization to ISO 8601
    - Amount normalization (remove commas, handle decimals)
    - Currency detection and standardization
    - Account number masking (PII minimization)
    """

    # Currency symbols and codes mapping
    CURRENCY_SYMBOLS = {
        '$': 'USD',
        '€': 'EUR',
        '£': 'GBP',
        '¥': 'JPY',
        '₹': 'INR',
        '৳': 'BDT',
        'A$': 'AUD',
        'C$': 'CAD',
        'R$': 'BRL',
        '₽': 'RUB',
        '₨': 'PKR',
    }

    # Common currency codes
    CURRENCY_CODES = [
        'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'INR', 'BDT', 'AUD', 'CAD',
        'BRL', 'RUB', 'PKR', 'SGD', 'HKD', 'KRW', 'MXN', 'ZAR', 'NZD'
    ]

    # Date format patterns
    DATE_PATTERNS = [
        (r'^\d{4}-\d{2}-\d{2}$', '%Y-%m-%d'),  # ISO 8601
        (r'^\d{2}-\w{3}-\d{4}$', '%d-%b-%Y'),  # DD-MMM-YYYY
        (r'^\d{2}/\d{2}/\d{4}$', '%d/%m/%Y'),  # DD/MM/YYYY
        (r'^\d{2}/\d{2}/\d{4}$', '%m/%d/%Y'),  # MM/DD/YYYY
        (r'^\d{4}/\d{2}/\d{2}$', '%Y/%m/%d'),  # YYYY/MM/DD
        (r'^\d{2}\.\d{2}\.\d{4}$', '%d.%m.%Y'),  # DD.MM.YYYY
    ]

    def __init__(self, mask_pii: bool = True, pii_mask_char: str = 'X'):
        """
        Initialize the normalization service.
        
        Args:
            mask_pii: Whether to mask PII by default
            pii_mask_char: Character to use for masking
        """
        self.mask_pii = mask_pii
        self.pii_mask_char = pii_mask_char
        logger.info(f"NormalizationService initialized (mask_pii={mask_pii})")

    def normalize_date(self, date_value: Any, source_format: Optional[str] = None) -> Optional[str]:
        """
        Normalize date to ISO 8601 format (YYYY-MM-DD).
        
        Args:
            date_value: Date value (string, datetime, or other)
            source_format: Optional hint about source format
        
        Returns:
            str: ISO 8601 formatted date (YYYY-MM-DD) or None if invalid
        """
        if date_value is None:
            return None

        # If already a datetime object
        if isinstance(date_value, datetime):
            return date_value.strftime('%Y-%m-%d')

        # Convert to string
        date_str = str(date_value).strip()
        if not date_str:
            return None

        # Try provided format first
        if source_format:
            try:
                dt = datetime.strptime(date_str, source_format)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass

        # Try common patterns
        for pattern, fmt in self.DATE_PATTERNS:
            if re.match(pattern, date_str):
                try:
                    dt = datetime.strptime(date_str, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue

        # Try parsing with dateutil if available (more flexible)
        try:
            from dateutil import parser
            dt = parser.parse(date_str)
            return dt.strftime('%Y-%m-%d')
        except (ImportError, ValueError, TypeError):
            pass

        logger.warning(f"Could not normalize date: {date_value}")
        return None

    def normalize_amount(
            self,
            amount_value: Any,
            currency: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Normalize amount to decimal number and detect currency.
        
        Args:
            amount_value: Amount value (string with symbols, number, etc.)
            currency: Optional currency hint
        
        Returns:
            dict: {
                'value': float,  # Normalized numeric value
                'currency': str,  # Detected currency code
                'original': str   # Original value
            }
        """
        result = {
            'value': 0.0,
            'currency': currency or 'USD',
            'original': str(amount_value) if amount_value is not None else ''
        }

        if amount_value is None:
            return result

        # If already a number
        if isinstance(amount_value, (int, float)):
            result['value'] = float(amount_value)
            return result

        # Convert to string and clean
        amount_str = str(amount_value).strip()
        if not amount_str:
            return result

        # Detect and remove currency symbol
        detected_currency = None
        for symbol, code in self.CURRENCY_SYMBOLS.items():
            if symbol in amount_str:
                detected_currency = code
                amount_str = amount_str.replace(symbol, '')
                break

        # Check for currency code at end (e.g., "100.50 USD")
        for code in self.CURRENCY_CODES:
            if amount_str.upper().endswith(f' {code}'):
                detected_currency = code
                amount_str = amount_str[:-len(code) - 1].strip()
                break

        if detected_currency:
            result['currency'] = detected_currency

        # Remove commas, spaces, and other formatting
        cleaned = re.sub(r'[,\s]', '', amount_str)

        # Try to parse as float
        try:
            result['value'] = float(cleaned)
        except ValueError:
            # Try to extract just the numeric part
            numeric_match = re.search(r'-?\d+\.?\d*', cleaned)
            if numeric_match:
                try:
                    result['value'] = float(numeric_match.group())
                except ValueError:
                    logger.warning(f"Could not parse amount: {amount_value}")

        return result

    def detect_currency(self, text: str) -> Optional[str]:
        """
        Detect currency from text.
        
        Args:
            text: Text that may contain currency information
        
        Returns:
            str: Currency code (e.g., 'USD') or None
        """
        if not text:
            return None

        text_upper = text.upper()

        # Check for currency codes
        for code in self.CURRENCY_CODES:
            if code in text_upper:
                return code

        # Check for currency symbols
        for symbol, code in self.CURRENCY_SYMBOLS.items():
            if symbol in text:
                return code

        return None

    def mask_account_number(
            self,
            account_number: Any,
            show_last: int = 4,
            mask_char: Optional[str] = None
    ) -> str:
        """
        Mask account number for PII protection.
        
        Args:
            account_number: Account number to mask
            show_last: Number of digits to show at the end
            mask_char: Character to use for masking (default: self.pii_mask_char)
        
        Returns:
            str: Masked account number (e.g., "XXXX1234")
        """
        if account_number is None:
            return ""

        mask = mask_char or self.pii_mask_char
        account_str = str(account_number).strip()

        # Remove spaces and dashes
        cleaned = re.sub(r'[\s-]', '', account_str)

        if len(cleaned) <= show_last:
            # If too short, mask all but last character
            return mask * (len(cleaned) - 1) + cleaned[-1] if cleaned else ""

        # Mask all but last N digits
        masked = mask * (len(cleaned) - show_last) + cleaned[-show_last:]
        return masked

    def mask_pii_field(
            self,
            field_name: str,
            field_value: Any,
            show_last: int = 4
    ) -> Any:
        """
        Mask PII in a field if it's a sensitive field type.
        
        Args:
            field_name: Name of the field
            field_value: Value to potentially mask
            show_last: Number of characters to show at the end
        
        Returns:
            Masked value if field is PII, original value otherwise
        """
        if not self.mask_pii:
            return field_value

        field_name_lower = field_name.lower()

        # Check if field contains PII indicators
        pii_indicators = [
            'account', 'ssn', 'social', 'tax', 'id', 'passport',
            'credit', 'card', 'routing', 'iban', 'swift'
        ]

        is_pii = any(indicator in field_name_lower for indicator in pii_indicators)

        if is_pii and field_value:
            if 'account' in field_name_lower and 'number' in field_name_lower:
                return self.mask_account_number(field_value, show_last)
            # For other PII, mask all but last few characters
            value_str = str(field_value)
            if len(value_str) > show_last:
                mask = self.pii_mask_char * (len(value_str) - show_last)
                return mask + value_str[-show_last:]

        return field_value

    def normalize_field(
            self,
            field_name: str,
            field_value: Any,
            field_type: str,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Normalize a single field based on its type.
        
        Args:
            field_name: Name of the field
            field_value: Value to normalize
            field_type: Type of field ('date', 'amount', 'number', 'string', etc.)
            context: Additional context (currency, date format hints, etc.)
        
        Returns:
            dict: {
                'value': normalized_value,
                'original': original_value,
                'normalized': True/False,
                'metadata': {...}
            }
        """
        result = {
            'value': field_value,
            'original': field_value,
            'normalized': False,
            'metadata': {}
        }

        if field_value is None:
            return result

        context = context or {}

        try:
            if field_type == 'date':
                normalized_date = self.normalize_date(
                    field_value,
                    source_format=context.get('date_format')
                )
                if normalized_date:
                    result['value'] = normalized_date
                    result['normalized'] = True
                    result['metadata']['format'] = 'ISO 8601'

            elif field_type in ['amount', 'number', 'currency']:
                normalized_amount = self.normalize_amount(
                    field_value,
                    currency=context.get('currency')
                )
                result['value'] = normalized_amount['value']
                result['metadata']['currency'] = normalized_amount['currency']
                result['metadata']['original'] = normalized_amount['original']
                if normalized_amount['value'] != 0.0 or normalized_amount['original']:
                    result['normalized'] = True

            elif field_type == 'string':
                # Basic string normalization (trim, etc.)
                if isinstance(field_value, str):
                    normalized = field_value.strip()
                    if normalized != field_value:
                        result['value'] = normalized
                        result['normalized'] = True

            # Apply PII masking if enabled
            if self.mask_pii:
                masked_value = self.mask_pii_field(field_name, result['value'])
                if masked_value != result['value']:
                    result['value'] = masked_value
                    result['metadata']['masked'] = True

        except Exception as e:
            logger.warning(f"Error normalizing field {field_name}: {str(e)}")
            result['metadata']['error'] = str(e)

        return result

    def normalize_extraction(
            self,
            extraction_data: Dict[str, Any],
            document_type: str,
            mask_pii: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Normalize an entire extraction following canonical schema.
        
        Args:
            extraction_data: Raw extraction data from VLM
            document_type: Type of document
            mask_pii: Override default PII masking setting
        
        Returns:
            dict: Normalized extraction data
        """
        # Temporarily override mask_pii if provided
        original_mask_pii = self.mask_pii
        if mask_pii is not None:
            self.mask_pii = mask_pii

        try:
            normalized = extraction_data.copy()

            # Detect document currency
            document_currency = self._detect_document_currency(extraction_data)

            # Normalize dates
            normalized = self._normalize_dates_in_extraction(normalized)

            # Normalize amounts
            normalized = self._normalize_amounts_in_extraction(normalized, document_currency)

            # Apply PII masking
            if self.mask_pii:
                normalized = self._apply_pii_masking(normalized)

            return normalized

        finally:
            # Restore original mask_pii setting
            self.mask_pii = original_mask_pii

    def _detect_document_currency(self, extraction_data: Dict[str, Any]) -> Optional[str]:
        """Detect currency from extraction data by searching all extraction paths."""

        def search_for_currency(data: Any, path: str = "") -> Optional[str]:
            """Recursively search for currency in extraction data."""
            if isinstance(data, dict):
                # Check if this dict has a currency field
                if 'currency' in data:
                    currency_value = data['currency']
                    if isinstance(currency_value, dict) and 'value' in currency_value:
                        currency_value = currency_value['value']
                    if currency_value and str(currency_value).upper() in self.CURRENCY_CODES:
                        return str(currency_value).upper()

                # Recurse into nested dictionaries
                for key, value in data.items():
                    result = search_for_currency(value, f"{path}.{key}" if path else key)
                    if result:
                        return result
            elif isinstance(data, list):
                # Search in list items
                for idx, item in enumerate(data):
                    result = search_for_currency(item, f"{path}[{idx}]")
                    if result:
                        return result
            return None

        # Search recursively through all extraction paths
        currency = search_for_currency(extraction_data.get('extractions', {}))
        if currency:
            return currency

        # Fallback: Try to detect from amounts text
        amounts_text = str(extraction_data)
        detected = self.detect_currency(amounts_text)
        return detected

    def _normalize_dates_in_extraction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively normalize dates in extraction data."""
        if isinstance(data, dict):
            normalized = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    # Check if this is a field with a 'value' key
                    if 'value' in value and ('date' in key.lower() or 'date' in str(value.get('value', '')).lower()):
                        normalized_value = self.normalize_date(value['value'])
                        if normalized_value:
                            normalized[key] = {**value, 'value': normalized_value}
                        else:
                            normalized[key] = value
                    else:
                        normalized[key] = self._normalize_dates_in_extraction(value)
                elif isinstance(value, list):
                    normalized[key] = [self._normalize_dates_in_extraction(item) for item in value]
                else:
                    normalized[key] = value
            return normalized
        elif isinstance(data, list):
            return [self._normalize_dates_in_extraction(item) for item in data]
        else:
            return data

    def _normalize_amounts_in_extraction(
            self,
            data: Dict[str, Any],
            document_currency: Optional[str] = None
    ) -> Dict[str, Any]:
        """Recursively normalize amounts in extraction data."""
        if isinstance(data, dict):
            normalized = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    # Check if this is an amount field
                    if 'value' in value and (
                            'amount' in key.lower() or
                            'price' in key.lower() or
                            'total' in key.lower() or
                            'balance' in key.lower() or
                            'debit' in key.lower() or
                            'credit' in key.lower()
                    ):
                        normalized_amount = self.normalize_amount(value['value'], document_currency)
                        normalized[key] = {
                            **value,
                            'value': normalized_amount['value'],
                            'currency': normalized_amount['currency']
                        }
                    else:
                        normalized[key] = self._normalize_amounts_in_extraction(value, document_currency)
                elif isinstance(value, list):
                    normalized[key] = [
                        self._normalize_amounts_in_extraction(item, document_currency)
                        for item in value
                    ]
                else:
                    normalized[key] = value
            return normalized
        elif isinstance(data, list):
            return [
                self._normalize_amounts_in_extraction(item, document_currency)
                for item in data
            ]
        else:
            return data

    def _apply_pii_masking(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively apply PII masking to extraction data."""
        if isinstance(data, dict):
            masked = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    if 'value' in value:
                        # This is a field with a value
                        masked_value = self.mask_pii_field(key, value['value'])
                        masked[key] = {**value, 'value': masked_value}
                    else:
                        masked[key] = self._apply_pii_masking(value)
                elif isinstance(value, list):
                    masked[key] = [self._apply_pii_masking(item) for item in value]
                else:
                    masked[key] = self.mask_pii_field(key, value)
            return masked
        elif isinstance(data, list):
            return [self._apply_pii_masking(item) for item in data]
        else:
            return data
