import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfidenceScorer:
    """
    Service for calculating confidence scores for extracted fields.
    
    Combines VLM-provided confidence with heuristic confidence based on:
    - Regex pattern matching
    - Data type validation
    - Format consistency
    - Parse success
    """

    # Default weights: 60% heuristic, 40% VLM
    HEURISTIC_WEIGHT = 0.6
    VLM_WEIGHT = 0.4

    # Confidence threshold for flagging fields for review
    DEFAULT_THRESHOLD = 0.70

    def __init__(self, heuristic_weight: float = None, vlm_weight: float = None, threshold: float = None):
        """
        Initialize the confidence scorer.
        
        Args:
            heuristic_weight: Weight for heuristic confidence (default: 0.6)
            vlm_weight: Weight for VLM confidence (default: 0.4)
            threshold: Confidence threshold for review flagging (default: 0.70)
        """
        self.heuristic_weight = heuristic_weight or self.HEURISTIC_WEIGHT
        self.vlm_weight = vlm_weight or self.VLM_WEIGHT
        self.threshold = threshold or self.DEFAULT_THRESHOLD

        # Ensure weights sum to 1.0
        total_weight = self.heuristic_weight + self.vlm_weight
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1.0 ({total_weight}), normalizing")
            self.heuristic_weight = self.heuristic_weight / total_weight
            self.vlm_weight = self.vlm_weight / total_weight

        logger.info(
            f"ConfidenceScorer initialized: heuristic={self.heuristic_weight:.2f}, "
            f"vlm={self.vlm_weight:.2f}, threshold={self.threshold:.2f}"
        )

    def calculate_confidence(
            self,
            field_name: str,
            field_value: Any,
            field_type: str,
            vlm_confidence: Optional[float] = None,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate combined confidence score for a field.
        
        Args:
            field_name: Name of the field (e.g., 'account_number', 'date')
            field_value: Extracted value for the field
            field_type: Expected data type ('string', 'number', 'date', 'email', etc.)
            vlm_confidence: Confidence score from VLM (0.0 to 1.0)
            context: Additional context (e.g., document type, page number)
        
        Returns:
            dict: {
                'confidence': float,  # Combined confidence (0.0 to 1.0)
                'heuristic_confidence': float,
                'vlm_confidence': float or None,
                'needs_review': bool,
                'reasons': List[str]  # Reasons for confidence score
            }
        """
        # Calculate heuristic confidence
        heuristic_conf = self._calculate_heuristic_confidence(
            field_name, field_value, field_type, context
        )

        # Normalize VLM confidence
        vlm_conf = None
        if vlm_confidence is not None:
            vlm_conf = max(0.0, min(1.0, float(vlm_confidence)))

        # Combine confidences
        if vlm_conf is not None:
            combined_confidence = (
                    self.heuristic_weight * heuristic_conf['confidence'] +
                    self.vlm_weight * vlm_conf
            )
        else:
            # If no VLM confidence, use only heuristic
            combined_confidence = heuristic_conf['confidence']
            logger.debug(f"No VLM confidence for {field_name}, using heuristic only")

        result = {
            'confidence': max(0.0, min(1.0, combined_confidence)),
            'heuristic_confidence': heuristic_conf['confidence'],
            'vlm_confidence': vlm_conf,
            'needs_review': combined_confidence < self.threshold,
            'reasons': heuristic_conf['reasons'].copy()
        }

        if vlm_conf is not None:
            result['reasons'].append(f"VLM confidence: {vlm_conf:.2f}")

        if result['needs_review']:
            result['reasons'].append(f"Below threshold ({self.threshold:.2f})")

        return result

    def _calculate_heuristic_confidence(
            self,
            field_name: str,
            field_value: Any,
            field_type: str,
            context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Calculate heuristic confidence based on validation rules.
        
        Returns:
            dict: {
                'confidence': float,
                'reasons': List[str]
            }
        """
        reasons = []
        confidence = 0.0

        # Check if value is None or empty
        if field_value is None or (isinstance(field_value, str) and not field_value.strip()):
            return {
                'confidence': 0.0,
                'reasons': ['Field is empty or null']
            }

        # Type-specific validation
        if field_type == 'date':
            conf, reason = self._validate_date(field_value)
            confidence += conf * 0.4
            reasons.append(reason)

        elif field_type == 'number' or field_type == 'amount':
            conf, reason = self._validate_number(field_value)
            confidence += conf * 0.4
            reasons.append(reason)

        elif field_type == 'email':
            conf, reason = self._validate_email(field_value)
            confidence += conf * 0.4
            reasons.append(reason)

        elif field_type == 'string':
            conf, reason = self._validate_string(field_value)
            confidence += conf * 0.3
            reasons.append(reason)

        # Field-name-specific validation
        field_specific_conf, field_reason = self._validate_field_specific(field_name, field_value)
        confidence += field_specific_conf * 0.3
        if field_reason:
            reasons.append(field_reason)

        # Format consistency check
        format_conf, format_reason = self._check_format_consistency(field_name, field_value, context)
        confidence += format_conf * 0.3
        if format_reason:
            reasons.append(format_reason)

        # Normalize to [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        return {
            'confidence': confidence,
            'reasons': reasons
        }

    def _validate_date(self, value: Any) -> tuple[float, str]:
        """Validate date format and return confidence."""
        if not isinstance(value, str):
            return 0.3, "Date is not a string"

        # Try to parse ISO 8601 format (YYYY-MM-DD)
        iso_pattern = r'^\d{4}-\d{2}-\d{2}$'
        if re.match(iso_pattern, value):
            try:
                datetime.strptime(value, '%Y-%m-%d')
                return 1.0, "Valid ISO 8601 date format"
            except ValueError:
                return 0.5, "Date format looks correct but invalid date"

        # Try other common formats
        common_formats = [
            '%d-%b-%Y',  # DD-MMM-YYYY
            '%d/%m/%Y',  # DD/MM/YYYY
            '%m/%d/%Y',  # MM/DD/YYYY
            '%Y-%m-%d',  # YYYY-MM-DD
        ]

        for fmt in common_formats:
            try:
                datetime.strptime(value, fmt)
                return 0.7, f"Valid date in format {fmt}"
            except ValueError:
                continue

        # Check if it looks like a date
        if re.search(r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}', value):
            return 0.4, "Looks like a date but format unclear"

        return 0.2, "Does not appear to be a valid date"

    def _validate_number(self, value: Any) -> tuple[float, str]:
        """Validate number format and return confidence."""
        if isinstance(value, (int, float)):
            return 1.0, "Valid numeric type"

        if isinstance(value, str):
            # Remove common formatting
            cleaned = value.replace(',', '').replace('$', '').replace(' ', '')

            # Try to parse as float
            try:
                float(cleaned)
                return 0.9, "Valid numeric string"
            except ValueError:
                pass

            # Check if it looks numeric
            if re.match(r'^[\d,.\s$€£¥]+$', value):
                return 0.6, "Looks numeric but parsing failed"

        return 0.2, "Does not appear to be a number"

    def _validate_email(self, value: Any) -> tuple[float, str]:
        """Validate email format and return confidence."""
        if not isinstance(value, str):
            return 0.0, "Email is not a string"

        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, value):
            return 1.0, "Valid email format"

        if '@' in value and '.' in value:
            return 0.5, "Looks like email but format invalid"

        return 0.1, "Does not appear to be an email"

    def _validate_string(self, value: Any) -> tuple[float, str]:
        """Validate string value and return confidence."""
        if isinstance(value, str):
            if len(value.strip()) > 0:
                return 0.8, "Non-empty string"
            return 0.3, "Empty or whitespace-only string"

        return 0.5, "Not a string type"

    def _validate_field_specific(self, field_name: str, value: Any) -> tuple[float, str]:
        """Validate field-specific patterns."""
        field_name_lower = field_name.lower()

        # Account number validation
        if 'account' in field_name_lower and 'number' in field_name_lower:
            if isinstance(value, str):
                # Account numbers are typically numeric or alphanumeric
                if re.match(r'^[A-Z0-9\s-]+$', value.upper()):
                    length = len(value.replace(' ', '').replace('-', ''))
                    if 8 <= length <= 20:
                        return 0.9, "Valid account number format"
                    return 0.6, "Account number length unusual"
            return 0.4, "Account number format invalid"

        # Currency validation
        if 'currency' in field_name_lower:
            if isinstance(value, str):
                currency_codes = ['USD', 'EUR', 'GBP', 'JPY', 'CNY', 'INR', 'BDT', 'AUD', 'CAD']
                if value.upper() in currency_codes:
                    return 1.0, "Valid currency code"
                if len(value) == 3 and value.isalpha():
                    return 0.7, "Looks like currency code"
            return 0.3, "Invalid currency format"

        # Phone number validation
        if 'phone' in field_name_lower:
            if isinstance(value, str):
                # Remove common formatting
                cleaned = re.sub(r'[\s\-\(\)]', '', value)
                if re.match(r'^\+?[\d]{10,15}$', cleaned):
                    return 0.9, "Valid phone number format"
                if re.search(r'\d{10,}', value):
                    return 0.6, "Contains digits but format unclear"
            return 0.3, "Invalid phone number format"

        # Default: no specific validation
        return 0.5, "No specific validation rules for this field"

    def _check_format_consistency(
            self,
            field_name: str,
            value: Any,
            context: Optional[Dict[str, Any]] = None
    ) -> tuple[float, str]:
        """Check format consistency with context."""
        if not context:
            return 0.5, "No context for consistency check"

        # Check currency consistency across document
        if 'currency' in field_name.lower() and 'document_currency' in context:
            if str(value).upper() == str(context['document_currency']).upper():
                return 0.8, "Currency matches document currency"
            return 0.4, "Currency mismatch with document"

        # Check date consistency (dates should be in chronological order)
        if 'date' in field_name.lower() and 'previous_date' in context:
            try:
                current_date = datetime.strptime(str(value), '%Y-%m-%d')
                prev_date = datetime.strptime(str(context['previous_date']), '%Y-%m-%d')
                if current_date >= prev_date:
                    return 0.8, "Date is chronologically consistent"
                return 0.5, "Date appears out of chronological order"
            except (ValueError, TypeError):
                pass

        return 0.5, "No consistency issues detected"

    def flag_fields_for_review(
            self,
            field_extractions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Flag fields that need human review based on confidence threshold.
        
        Args:
            field_extractions: List of field extraction dicts with 'confidence' key
        
        Returns:
            List of field extractions with 'needs_review' flag set
        """
        flagged = []
        for field in field_extractions:
            confidence = field.get('confidence', 0.0)
            field['needs_review'] = confidence < self.threshold
            if field['needs_review']:
                flagged.append(field)

        logger.info(f"Flagged {len(flagged)}/{len(field_extractions)} fields for review")
        return flagged

    def calculate_overall_confidence(
            self,
            field_extractions: List[Dict[str, Any]],
            weights: Optional[Dict[str, float]] = None
    ) -> float:
        """
        Calculate overall document confidence as weighted average of field confidences.
        
        Args:
            field_extractions: List of field extraction dicts with 'confidence' key
            weights: Optional dict mapping field paths to weights (default: equal weights)
        
        Returns:
            float: Overall confidence score (0.0 to 1.0)
        """
        if not field_extractions:
            return 0.0

        if weights is None:
            # Equal weights for all fields
            total_confidence = sum(field.get('confidence', 0.0) for field in field_extractions)
            return total_confidence / len(field_extractions)

        # Weighted average
        total_weighted_confidence = 0.0
        total_weight = 0.0

        for field in field_extractions:
            field_path = field.get('field_path', '')
            weight = weights.get(field_path, 1.0)
            confidence = field.get('confidence', 0.0)

            total_weighted_confidence += confidence * weight
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return total_weighted_confidence / total_weight
