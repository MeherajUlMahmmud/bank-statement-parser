"""
Data processing service for validating and cleaning extracted data.
Handles error correction, validation, and data quality improvements.
"""
import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Service for processing, validating, and cleaning extracted data.
    
    Handles:
    - Data consistency validation
    - Detection and marking of invalid entries
    - Auto-correction of common OCR/LLM errors
    - Removal of duplicate or empty rows
    - Data quality assessment
    """

    # Common OCR/LLM error patterns
    OCR_ERROR_PATTERNS = {
        # Number character confusions
        r'[Oo]': '0',  # O/o -> 0
        r'[Il1]': '1',  # I/l/1 -> 1
        r'[Zz]': '2',  # Z/z -> 2
        r'[Ss]': '5',  # S/s -> 5
        r'[Gg]': '6',  # G/g -> 6
        r'[Bb]': '8',  # B/b -> 8
    }

    # Common date format errors
    DATE_ERROR_PATTERNS = [
        (r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', r'\1/\2/\3'),  # Normalize separators
        (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', r'\1-\2-\3'),  # ISO format
    ]

    def __init__(self, auto_correct: bool = True, remove_duplicates: bool = True):
        """
        Initialize the DataProcessor.
        
        Args:
            auto_correct: Whether to automatically correct common errors
            remove_duplicates: Whether to remove duplicate entries
        """
        self.auto_correct = auto_correct
        self.remove_duplicates = remove_duplicates
        logger.info(f"DataProcessor initialized (auto_correct={auto_correct}, remove_duplicates={remove_duplicates})")

    def resolve_issues(self, canonical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and clean canonical data, resolving issues and validating consistency.
        
        Args:
            canonical_data: Canonical JSON data structure from extraction
            
        Returns:
            Dict containing cleaned data and validation status
        """
        try:
            logger.info("Starting data processing and validation")

            # Create a copy to avoid modifying original
            processed_data = self._deep_copy(canonical_data)

            # Track validation issues
            validation_status = {
                'is_valid': True,
                'issues_found': [],
                'issues_resolved': [],
                'duplicates_removed': 0,
                'empty_entries_removed': 0,
                'corrections_applied': 0
            }

            # Process extractions
            if 'extractions' in processed_data:
                processed_data['extractions'], extraction_status = self._process_extractions(
                    processed_data['extractions']
                )
                validation_status.update(extraction_status)

            # Validate overall consistency
            consistency_issues = self._validate_consistency(processed_data)
            if consistency_issues:
                validation_status['issues_found'].extend(consistency_issues)
                validation_status['is_valid'] = False

            # Add validation metadata to processed data
            processed_data['validation'] = validation_status

            logger.info(
                f"Data processing completed. "
                f"Issues found: {len(validation_status['issues_found'])}, "
                f"Resolved: {len(validation_status['issues_resolved'])}, "
                f"Valid: {validation_status['is_valid']}"
            )

            return processed_data

        except Exception as e:
            logger.error(f"Error during data processing: {e}", exc_info=True)
            # Return original data with error status
            canonical_data['validation'] = {
                'is_valid': False,
                'error': str(e),
                'issues_found': [f"Processing error: {str(e)}"]
            }
            return canonical_data

    def _process_extractions(self, extractions: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Process extraction data recursively.
        
        Args:
            extractions: Extraction data dictionary
            
        Returns:
            Tuple of (processed_extractions, status_dict)
        """
        status = {
            'issues_found': [],
            'issues_resolved': [],
            'duplicates_removed': 0,
            'empty_entries_removed': 0,
            'corrections_applied': 0
        }

        processed = {}

        for key, value in extractions.items():
            if isinstance(value, dict):
                processed[key] = self._process_dict_value(value, key, status)
            elif isinstance(value, list):
                processed[key], list_status = self._process_list_value(value, key, status)
                status.update(list_status)
            else:
                processed[key] = self._process_simple_value(value, key, status)

        return processed, status

    def _process_dict_value(
            self,
            value: Dict[str, Any],
            key: str,
            status: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a dictionary value."""
        processed = {}

        for sub_key, sub_value in value.items():
            if isinstance(sub_value, dict):
                processed[sub_key] = self._process_dict_value(sub_value, f"{key}.{sub_key}", status)
            elif isinstance(sub_value, list):
                processed[sub_key], list_status = self._process_list_value(
                    sub_value, f"{key}.{sub_key}", status
                )
                status.update(list_status)
            else:
                processed[sub_key] = self._process_simple_value(sub_value, f"{key}.{sub_key}", status)

        return processed

    def _process_list_value(
            self,
            value: List[Any],
            key: str,
            status: Dict[str, Any]
    ) -> Tuple[List[Any], Dict[str, Any]]:
        """Process a list value, handling duplicates and empty entries."""
        if not value:
            return value, {}

        processed_list = []
        seen_items = set()
        list_status = {
            'duplicates_removed': 0,
            'empty_entries_removed': 0
        }

        for idx, item in enumerate(value):
            # Check if item is empty
            if self._is_empty_entry(item):
                list_status['empty_entries_removed'] += 1
                status['empty_entries_removed'] += 1
                status['issues_found'].append(f"Empty entry removed from {key}[{idx}]")
                continue

            # Process the item
            if isinstance(item, dict):
                processed_item = self._process_dict_value(item, f"{key}[{idx}]", status)
            elif isinstance(item, list):
                processed_item, _ = self._process_list_value(item, f"{key}[{idx}]", status)
            else:
                processed_item = self._process_simple_value(item, f"{key}[{idx}]", status)

            # Check for duplicates if enabled
            if self.remove_duplicates:
                item_hash = self._hash_item(processed_item)
                if item_hash in seen_items:
                    list_status['duplicates_removed'] += 1
                    status['duplicates_removed'] += 1
                    status['issues_found'].append(f"Duplicate entry removed from {key}[{idx}]")
                    continue
                seen_items.add(item_hash)

            processed_list.append(processed_item)

        return processed_list, list_status

    def _process_simple_value(
            self,
            value: Any,
            key: str,
            status: Dict[str, Any]
    ) -> Any:
        """Process a simple value (string, number, etc.)."""
        if value is None:
            return value

        # Handle field structure with value/confidence/bbox
        if isinstance(value, dict) and 'value' in value:
            original_value = value['value']
            corrected_value = self._correct_value(original_value, key, status)

            if corrected_value != original_value:
                status['corrections_applied'] += 1
                status['issues_resolved'].append(f"Corrected value in {key}: {original_value} -> {corrected_value}")
                return {**value, 'value': corrected_value, 'auto_corrected': True}

            return value

        # Handle simple values
        if isinstance(value, (str, int, float)):
            corrected = self._correct_value(value, key, status)
            if corrected != value:
                status['corrections_applied'] += 1
                status['issues_resolved'].append(f"Corrected value in {key}: {value} -> {corrected}")
                return corrected

        return value

    def _correct_value(self, value: Any, key: str, status: Dict[str, Any]) -> Any:
        """
        Attempt to correct common OCR/LLM errors in a value.
        
        Args:
            value: Value to correct
            key: Field key for context
            status: Status dictionary to update
            
        Returns:
            Corrected value
        """
        if not self.auto_correct or value is None:
            return value

        if isinstance(value, str):
            original = value
            corrected = value

            # Only apply corrections to numeric-looking strings or specific field types
            if self._is_numeric_field(key) or re.match(r'^[\d\s\.,\-+OIl1ZzSsGgBb]+$', value):
                # Fix common OCR errors in numbers
                for pattern, replacement in self.OCR_ERROR_PATTERNS.items():
                    if re.search(pattern, corrected):
                        # Only replace if it makes sense (e.g., in a number context)
                        if self._is_numeric_field(key):
                            corrected = re.sub(pattern, replacement, corrected)

            # Clean up whitespace
            corrected = ' '.join(corrected.split())

            # Validate and correct dates
            if 'date' in key.lower():
                corrected = self._correct_date_format(corrected)

            if corrected != original:
                return corrected

        return value

    def _correct_date_format(self, date_str: str) -> str:
        """Attempt to correct date format issues."""
        if not date_str or not isinstance(date_str, str):
            return date_str

        # Try to parse and reformat dates
        date_patterns = [
            (r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', '%m/%d/%Y', '%Y-%m-%d'),
            (r'(\d{4})[/-](\d{1,2})[/-](\d{1,2})', '%Y-%m-%d', '%Y-%m-%d'),
        ]

        for pattern, parse_fmt, output_fmt in date_patterns:
            match = re.match(pattern, date_str)
            if match:
                try:
                    # Try to parse and reformat
                    parts = match.groups()
                    if len(parts) == 3:
                        # Simple reformat attempt
                        if len(parts[2]) == 2:
                            year = '20' + parts[2] if int(parts[2]) < 50 else '19' + parts[2]
                        else:
                            year = parts[2]
                        return f"{year}-{parts[1].zfill(2)}-{parts[0].zfill(2)}"
                except Exception:
                    pass

        return date_str

    def _is_numeric_field(self, key: str) -> bool:
        """Check if a field key suggests a numeric value."""
        numeric_indicators = [
            'amount', 'price', 'total', 'balance', 'quantity', 'count',
            'number', 'num', 'id', 'account', 'debit', 'credit', 'fee'
        ]
        key_lower = key.lower()
        return any(indicator in key_lower for indicator in numeric_indicators)

    def _is_empty_entry(self, entry: Any) -> bool:
        """Check if an entry is considered empty."""
        if entry is None:
            return True

        if isinstance(entry, dict):
            # Check if dict has meaningful values
            if not entry:
                return True
            # Check if all values are None or empty
            if all(
                    v is None or (isinstance(v, str) and not v.strip()) or
                    (isinstance(v, dict) and not v)
                    for v in entry.values()
            ):
                return True

        if isinstance(entry, str):
            return not entry.strip()

        if isinstance(entry, list):
            return len(entry) == 0

        return False

    def _hash_item(self, item: Any) -> str:
        """Create a hash string for an item to detect duplicates."""
        if isinstance(item, dict):
            # Sort keys and create hash string
            sorted_items = sorted(item.items())
            return str(sorted_items)
        elif isinstance(item, (list, tuple)):
            return str(tuple(item))
        else:
            return str(item)

    def _validate_consistency(self, data: Dict[str, Any]) -> List[str]:
        """
        Validate data consistency across the document.
        
        Args:
            data: Canonical data structure
            
        Returns:
            List of consistency issues found
        """
        issues = []

        try:
            extractions = data.get('extractions', {})

            # Check for required fields based on document type
            document_type = data.get('document_type', 'generic')

            # Validate transaction consistency (if transactions exist)
            if 'transactions' in extractions:
                transactions = extractions['transactions']
                if isinstance(transactions, list):
                    issues.extend(self._validate_transactions(transactions))

            # Validate date ranges
            issues.extend(self._validate_date_ranges(extractions))

            # Validate amount consistency
            issues.extend(self._validate_amount_consistency(extractions))

        except Exception as e:
            logger.warning(f"Error during consistency validation: {e}")
            issues.append(f"Consistency validation error: {str(e)}")

        return issues

    def _validate_transactions(self, transactions: List[Dict[str, Any]]) -> List[str]:
        """Validate transaction list consistency."""
        issues = []

        if not transactions:
            return issues

        # Check for required fields in transactions
        required_fields = ['date', 'amount', 'description']

        for idx, transaction in enumerate(transactions):
            if isinstance(transaction, dict):
                # Check if transaction has value structure
                transaction_values = {}
                for field in required_fields:
                    if field in transaction:
                        field_data = transaction[field]
                        if isinstance(field_data, dict) and 'value' in field_data:
                            transaction_values[field] = field_data['value']
                        else:
                            transaction_values[field] = field_data

                # Validate required fields
                for field in required_fields:
                    if field not in transaction_values or not transaction_values[field]:
                        issues.append(f"Transaction {idx} missing required field: {field}")

        return issues

    def _validate_date_ranges(self, extractions: Dict[str, Any]) -> List[str]:
        """Validate date ranges for consistency."""
        issues = []

        # Extract all dates from extractions
        dates = self._extract_dates(extractions)

        if len(dates) > 1:
            # Check for reasonable date ranges
            try:
                parsed_dates = []
                for date_str in dates:
                    try:
                        # Try ISO format first
                        parsed = datetime.strptime(date_str, '%Y-%m-%d')
                        parsed_dates.append(parsed)
                    except ValueError:
                        pass

                if len(parsed_dates) > 1:
                    min_date = min(parsed_dates)
                    max_date = max(parsed_dates)
                    date_range = (max_date - min_date).days

                    # Flag if date range seems unreasonable (more than 10 years)
                    if date_range > 3650:
                        issues.append(
                            f"Unusual date range detected: {min_date.date()} to {max_date.date()} ({date_range} days)")
            except Exception as e:
                logger.debug(f"Error validating date ranges: {e}")

        return issues

    def _validate_amount_consistency(self, extractions: Dict[str, Any]) -> List[str]:
        """Validate amount consistency."""
        issues = []

        # Extract amounts
        amounts = self._extract_amounts(extractions)

        if amounts:
            # Check for negative balances that might be errors
            balance_amounts = [a for a in amounts if 'balance' in str(a).lower()]
            for amount in balance_amounts:
                try:
                    if isinstance(amount, (int, float)) and amount < -1000000:
                        issues.append(f"Unusually large negative balance detected: {amount}")
                except Exception:
                    pass

        return issues

    def _extract_dates(self, data: Any, dates: Optional[List[str]] = None) -> List[str]:
        """Recursively extract all date values from data."""
        if dates is None:
            dates = []

        if isinstance(data, dict):
            for key, value in data.items():
                if 'date' in key.lower():
                    if isinstance(value, dict) and 'value' in value:
                        date_value = value['value']
                        if date_value and isinstance(date_value, str):
                            dates.append(date_value)
                    elif isinstance(value, str):
                        dates.append(value)
                else:
                    self._extract_dates(value, dates)
        elif isinstance(data, list):
            for item in data:
                self._extract_dates(item, dates)

        return dates

    def _extract_amounts(self, data: Any, amounts: Optional[List[Any]] = None) -> List[Any]:
        """Recursively extract all amount values from data."""
        if amounts is None:
            amounts = []

        if isinstance(data, dict):
            for key, value in data.items():
                if any(term in key.lower() for term in ['amount', 'price', 'total', 'balance']):
                    if isinstance(value, dict) and 'value' in value:
                        amount_value = value['value']
                        if amount_value is not None:
                            amounts.append(amount_value)
                    elif isinstance(value, (int, float, str)):
                        amounts.append(value)
                else:
                    self._extract_amounts(value, amounts)
        elif isinstance(data, list):
            for item in data:
                self._extract_amounts(item, amounts)

        return amounts

    def _deep_copy(self, data: Any) -> Any:
        """Create a deep copy of data structure."""
        if isinstance(data, dict):
            return {key: self._deep_copy(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._deep_copy(item) for item in data]
        else:
            return data
