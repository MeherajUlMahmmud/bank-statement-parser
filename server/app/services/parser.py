import logging
import re
from decimal import Decimal, InvalidOperation
from datetime import datetime, date
from typing import List, Dict, Tuple, Any, Optional
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


# Currency name to ISO 4217 code mapping
CURRENCY_MAPPING = {
    # Common currency names and variations
    'taka': 'BDT',
    'bangladesh taka': 'BDT',
    'bangladeshi taka': 'BDT',
    'tk': 'BDT',
    '৳': 'BDT',

    'dollar': 'USD',
    'us dollar': 'USD',
    'usd': 'USD',
    '$': 'USD',

    'euro': 'EUR',
    'eur': 'EUR',
    '€': 'EUR',

    'pound': 'GBP',
    'pound sterling': 'GBP',
    'gbp': 'GBP',
    '£': 'GBP',

    'rupee': 'INR',
    'indian rupee': 'INR',
    'inr': 'INR',
    '₹': 'INR',

    'yen': 'JPY',
    'japanese yen': 'JPY',
    'jpy': 'JPY',
    '¥': 'JPY',

    'yuan': 'CNY',
    'chinese yuan': 'CNY',
    'renminbi': 'CNY',
    'cny': 'CNY',

    'riyal': 'SAR',
    'saudi riyal': 'SAR',
    'sar': 'SAR',

    'dirham': 'AED',
    'uae dirham': 'AED',
    'aed': 'AED',

    'ringgit': 'MYR',
    'malaysian ringgit': 'MYR',
    'myr': 'MYR',

    'baht': 'THB',
    'thai baht': 'THB',
    'thb': 'THB',

    'won': 'KRW',
    'south korean won': 'KRW',
    'krw': 'KRW',

    'franc': 'CHF',
    'swiss franc': 'CHF',
    'chf': 'CHF',

    'rand': 'ZAR',
    'south african rand': 'ZAR',
    'zar': 'ZAR',
}


def normalize_currency(currency_str: Optional[str]) -> str:
    """
    Normalize currency string to ISO 4217 code

    Args:
        currency_str: Currency name, code, or symbol

    Returns:
        ISO 4217 currency code (default: BDT for Bangladesh)
    """
    if not currency_str:
        return 'BDT'

    # Clean the input
    cleaned = str(currency_str).strip().lower()

    # Remove common words
    cleaned = re.sub(r'\b(currency|code)\b', '', cleaned).strip()

    # Check if it's already a valid 3-letter ISO code
    if len(cleaned) == 3 and cleaned.upper() in ['USD', 'EUR', 'GBP', 'INR', 'JPY', 'CNY', 'BDT',
                                                    'SAR', 'AED', 'MYR', 'THB', 'KRW', 'CHF', 'ZAR',
                                                    'PKR', 'LKR', 'NPR', 'CAD', 'AUD', 'NZD', 'SGD']:
        return cleaned.upper()

    # Try to find in mapping
    if cleaned in CURRENCY_MAPPING:
        return CURRENCY_MAPPING[cleaned]

    # Try partial match for compound names
    for key, code in CURRENCY_MAPPING.items():
        if key in cleaned or cleaned in key:
            return code

    # Default to BDT if not found
    logger.warning(f"Currency '{currency_str}' not recognized, defaulting to BDT")
    return 'BDT'


class StatementParser:
    """
    Responsible for parsing and normalizing LLM responses into structured data
    """

    # Common patterns for detecting transaction type
    DEBIT_INDICATORS = ['debit', 'dr', 'withdrawal', 'paid_out', 'withdrawals', 'withdraw']
    CREDIT_INDICATORS = ['credit', 'cr', 'deposit', 'paid_in', 'deposits']

    # Common date column names
    DATE_COLUMNS = [
        'date', 'Date', 'DATE',
        'transaction_date', 'Transaction Date', 'TRANSACTION DATE',
        'txn_date', 'Txn Date', 'TXN DATE',
        'posting_date', 'Posting Date', 'POSTING DATE',
        'value_date', 'Value Date', 'VALUE DATE',
        'booking_date', 'Booking Date', 'BOOKING DATE'
    ]

    def parse_and_merge(self, responses: List[dict]) -> dict:
        """
        Merge responses from multiple pages into unified structure

        Args:
            responses: List of LLM responses, one per page

        Returns:
            Merged and normalized data structure
        """
        if not responses:
            raise ValueError("No responses to parse")

        # Merge customer and bank details from first page that has them
        customer_details = self._merge_customer_details(responses)
        bank_details = self._merge_bank_details(responses)

        # Detect transaction schema
        schema = self._detect_schema(responses)

        # Normalize and merge transactions
        all_transactions = []
        for idx, response in enumerate(responses):
            page_number = idx + 1
            transactions = response.get('transactions', [])

            for row_idx, txn in enumerate(transactions):
                normalized = self._normalize_transaction(txn, schema, page_number, row_idx)
                all_transactions.append(normalized)

        # Deduplicate transactions
        all_transactions = self._deduplicate_transactions(all_transactions)

        return {
            'customer_details': customer_details,
            'bank_details': bank_details,
            'transaction_schema': schema,
            'transactions': all_transactions
        }

    def _merge_customer_details(self, responses: List[dict]) -> dict:
        """Take customer details from first page that has them"""
        for response in responses:
            customer = response.get('customer_details', {})
            if customer and (customer.get('account_holder_name') or customer.get('account_number')):
                return customer

        return {}

    def _merge_bank_details(self, responses: List[dict]) -> dict:
        """Merge bank details, preferring non-null values"""
        merged = {}

        for response in responses:
            bank = response.get('bank_details', {})
            if not bank:
                continue

            for key, value in bank.items():
                if value is not None and (key not in merged or merged[key] is None):
                    merged[key] = value

        # Normalize currency to ISO 4217 code
        if 'currency' in merged:
            merged['currency'] = normalize_currency(merged['currency'])
        else:
            merged['currency'] = 'BDT'

        return merged

    def _detect_schema(self, responses: List[dict]) -> dict:
        """
        Detect transaction column schema from responses

        Returns:
            {
                "columns": ["Date", "Description", "Debit", "Credit", "Balance"],
                "column_metadata": {...}
            }
        """
        # Get columns from first response that has them
        columns = []
        for response in responses:
            cols = response.get('transaction_columns', [])
            if cols:
                columns = cols
                break

        if not columns:
            logger.warning("No transaction columns detected, using default schema")
            columns = ["date", "description", "amount", "balance"]

        # Infer column types from sample data
        column_metadata = {}
        sample_transactions = []
        for response in responses:
            sample_transactions.extend(response.get('transactions', [])[:5])

        for col in columns:
            col_type = self._infer_column_type(col, [txn.get(col) for txn in sample_transactions])
            column_metadata[col] = {
                'type': col_type,
                'display_name': col
            }

        return {
            'columns': columns,
            'column_metadata': column_metadata
        }

    def _normalize_transaction(self, txn: dict, schema: dict, page_num: int, row_idx: int) -> dict:
        """
        Normalize a single transaction for storage

        Returns:
            {
                'data': {...},  # Original JSON
                'transaction_date': date or None,
                'amount': Decimal or None,
                'transaction_type': str or None,
                'page_number': int,
                'row_index': int
            }
        """
        # Extract normalized fields
        txn_date = self._extract_date(txn, schema)
        amount, txn_type = self._extract_amount_and_type(txn, schema)

        return {
            'data': txn,
            'transaction_date': txn_date,
            'amount': amount,
            'transaction_type': txn_type,
            'page_number': page_num,
            'row_index': row_idx
        }

    def _extract_date(self, txn: dict, schema: dict) -> Optional[date]:
        """Try to extract and parse date from transaction"""
        # Try all common date column names
        for col_name in self.DATE_COLUMNS:
            if col_name in txn:
                date_val = txn[col_name]
                if date_val:
                    parsed = self._parse_date(date_val)
                    if parsed:
                        return parsed

        # Try first column if it looks like a date
        columns = schema.get('columns', [])
        if columns:
            first_col = columns[0]
            if first_col in txn:
                date_val = txn[first_col]
                if date_val:
                    parsed = self._parse_date(date_val)
                    if parsed:
                        return parsed

        return None

    def _extract_amount_and_type(self, txn: dict, schema: dict) -> Tuple[Optional[Decimal], Optional[str]]:
        """
        Extract amount and determine transaction type

        Strategies:
        1. Separate Debit/Credit columns: check which has value
        2. Single Amount column with +/-: check sign
        3. Withdrawals/Deposits columns: check which has value
        """
        columns = schema.get('columns', [])

        # Strategy 1: Look for separate debit/credit columns
        debit_col = None
        credit_col = None

        for col in columns:
            col_lower = col.lower()
            if any(indicator in col_lower for indicator in self.DEBIT_INDICATORS):
                debit_col = col
            elif any(indicator in col_lower for indicator in self.CREDIT_INDICATORS):
                credit_col = col

        if debit_col and credit_col:
            debit_val = self._parse_amount(txn.get(debit_col))
            credit_val = self._parse_amount(txn.get(credit_col))

            # Check debit column first (handle both positive and negative values)
            if debit_val and debit_val != 0:
                return abs(debit_val), 'debit'
            # Then check credit column (handle both positive and negative values)
            if credit_val and credit_val != 0:
                return abs(credit_val), 'credit'

            # If both columns exist but both are empty/zero, continue to other strategies

        # If only debit column exists
        if debit_col and not credit_col:
            debit_val = self._parse_amount(txn.get(debit_col))
            if debit_val and debit_val != 0:
                return abs(debit_val), 'debit'

        # If only credit column exists
        if credit_col and not debit_col:
            credit_val = self._parse_amount(txn.get(credit_col))
            if credit_val and credit_val != 0:
                return abs(credit_val), 'credit'

        # Strategy 2: Single amount column with sign
        for col in columns:
            col_lower = col.lower()
            if 'amount' in col_lower:
                amount_val = self._parse_amount(txn.get(col))
                if amount_val is not None:
                    if amount_val < 0:
                        return abs(amount_val), 'debit'
                    else:
                        return amount_val, 'credit'

        # Strategy 3: Try any numeric column
        for col in columns:
            if col in txn:
                amount_val = self._parse_amount(txn[col])
                if amount_val is not None and amount_val != 0:
                    return abs(amount_val), None

        return None, None

    def _parse_date(self, date_str: Any) -> Optional[date]:
        """
        Parse date from various formats
        """
        if isinstance(date_str, date):
            return date_str

        if not date_str or not isinstance(date_str, str):
            return None

        try:
            # Try using dateutil parser which handles many formats
            parsed = date_parser.parse(date_str, dayfirst=True)
            return parsed.date()
        except (ValueError, TypeError):
            return None

    def _parse_amount(self, value: Any) -> Optional[Decimal]:
        """
        Parse amount from various formats
        """
        if value is None or value == '':
            return None

        if isinstance(value, (int, float)):
            try:
                return Decimal(str(value))
            except InvalidOperation:
                return None

        if isinstance(value, str):
            # Remove currency symbols, commas, spaces
            cleaned = re.sub(r'[^\d.+-]', '', value)

            # Handle parentheses as negative
            if '(' in str(value) and ')' in str(value):
                cleaned = '-' + cleaned

            try:
                return Decimal(cleaned)
            except InvalidOperation:
                return None

        return None

    def _deduplicate_transactions(self, transactions: List[dict]) -> List[dict]:
        """
        Remove duplicate transactions that might appear on page boundaries

        Strategy: Compare date + amount + first few chars of description
        """
        if len(transactions) <= 1:
            return transactions

        seen = set()
        deduplicated = []

        for txn in transactions:
            # Create signature from date, amount, and description
            date_str = str(txn.get('transaction_date', ''))
            amount_str = str(txn.get('amount', ''))

            # Try to get description from data
            data = txn.get('data', {})
            desc = ''
            for key in data:
                if 'desc' in key.lower() or 'narr' in key.lower() or 'particular' in key.lower():
                    desc = str(data[key])[:50]
                    break

            signature = f"{date_str}|{amount_str}|{desc}"

            if signature not in seen:
                seen.add(signature)
                deduplicated.append(txn)
            else:
                logger.debug(f"Skipping duplicate transaction: {signature}")

        logger.info(f"Deduplicated {len(transactions)} -> {len(deduplicated)} transactions")
        return deduplicated

    def _infer_column_type(self, column_name: str, sample_values: List[Any]) -> str:
        """
        Infer column data type from name and sample values

        Returns: 'date', 'currency', 'text', 'number'
        """
        col_lower = column_name.lower()

        # Check by name first
        if 'date' in col_lower:
            return 'date'

        if any(indicator in col_lower for indicator in self.DEBIT_INDICATORS + self.CREDIT_INDICATORS):
            return 'currency'

        if 'balance' in col_lower or 'amount' in col_lower:
            return 'currency'

        # Check sample values
        numeric_count = 0
        for val in sample_values:
            if val is not None:
                if isinstance(val, (int, float)):
                    numeric_count += 1
                elif isinstance(val, str) and self._parse_amount(val) is not None:
                    numeric_count += 1

        if numeric_count > len(sample_values) * 0.5:
            return 'currency'

        return 'text'

    def validate_response(self, response: dict) -> Tuple[bool, List[str]]:
        """
        Validate LLM response structure

        Returns:
            (is_valid, list_of_error_messages)
        """
        errors = []

        if not isinstance(response, dict):
            errors.append("Response is not a dictionary")
            return False, errors

        # Check required keys
        if 'customer_details' not in response:
            errors.append("Missing 'customer_details' key")

        if 'bank_details' not in response:
            errors.append("Missing 'bank_details' key")

        if 'transactions' not in response:
            errors.append("Missing 'transactions' key")

        if 'transaction_columns' not in response:
            errors.append("Missing 'transaction_columns' key")

        return len(errors) == 0, errors
