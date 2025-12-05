import os
import logging
from datetime import datetime
from app.models import db, BankStatement, CustomerDetails, BankDetails, Transaction, TransactionSchema
from app.services.pdf_processor import PDFProcessor
from app.services.llm_service import GroqCloudService
from app.services.parser import StatementParser

logger = logging.getLogger(__name__)


class StatementProcessor:
    """
    Orchestrates the complete statement processing pipeline
    """

    def __init__(self, config):
        self.pdf_processor = PDFProcessor(config)
        self.llm_service = GroqCloudService(config)
        self.parser = StatementParser()
        self.cleanup_temp_files = config['CLEANUP_TEMP_FILES']
        self.upload_folder = config['UPLOAD_FOLDER']

    def process_statement(self, statement_id: int) -> bool:
        """
        Main processing pipeline

        Steps:
        1. Update status to 'processing'
        2. Extract images from PDF
        3. Send each image to LLM
        4. Parse and merge responses
        5. Save to database
        6. Update status to 'completed'
        7. Cleanup temporary files

        Returns:
            True if successful, False otherwise
        """
        statement = BankStatement.query.get(statement_id)
        if not statement:
            logger.error(f"Statement {statement_id} not found")
            return False

        try:
            # Step 1: Update status
            statement.processing_status = 'processing'
            statement.processing_started_at = datetime.utcnow()
            db.session.commit()
            logger.info(f"Started processing statement {statement_id}")

            # Step 2: Extract images
            pdf_path = os.path.join(self.upload_folder, statement.filename)
            images = self.pdf_processor.extract_images(pdf_path, statement_id)
            statement.page_count = len(images)
            db.session.commit()
            logger.info(f"Extracted {len(images)} pages from statement {statement_id}")

            # Step 3: Process each page with LLM
            llm_responses = []
            processing_logs = []
            all_validation_errors = {}

            for img_info in images:
                try:
                    log_entry = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'page': img_info['page_number'],
                        'action': 'processing_page'
                    }

                    response = self.llm_service.extract_from_image(
                        img_info['image_path'],
                        img_info['page_number']
                    )

                    # Extract metadata from response
                    metadata = response.get('_metadata', {})
                    validation_errors = response.get('_validation_errors')

                    log_entry['status'] = 'success'
                    log_entry['tokens_used'] = metadata.get('tokens_used', 0)
                    log_entry['validated'] = metadata.get('validated', True)

                    if validation_errors:
                        all_validation_errors[f'page_{img_info["page_number"]}'] = validation_errors
                        log_entry['validation_errors'] = validation_errors

                    processing_logs.append(log_entry)

                    # Validate response
                    is_valid, errors = self.parser.validate_response(response)
                    if not is_valid:
                        logger.warning(f"Invalid response for page {img_info['page_number']}: {errors}")
                        # Continue with partial data

                    llm_responses.append(response)
                except Exception as e:
                    logger.error(f"Error processing page {img_info['page_number']}: {e}")
                    processing_logs.append({
                        'timestamp': datetime.utcnow().isoformat(),
                        'page': img_info['page_number'],
                        'action': 'processing_page',
                        'status': 'error',
                        'error': str(e)
                    })
                    # Continue with other pages
                    continue

            if not llm_responses:
                raise Exception("Failed to process any pages successfully")

            # Step 4: Parse and merge
            parsed_data = self.parser.parse_and_merge(llm_responses)
            logger.info(f"Parsed {len(parsed_data.get('transactions', []))} transactions")

            # Collect metadata
            token_usage = self.llm_service.get_token_usage()
            confidence_scores = self._extract_confidence_scores(llm_responses)

            # Store metadata in statement
            statement.raw_extraction_data = llm_responses
            statement.token_usage = token_usage
            statement.confidence_scores = confidence_scores
            statement.processing_logs = processing_logs
            if all_validation_errors:
                statement.validation_errors = all_validation_errors

            # Step 5: Save to database
            self._save_to_database(statement_id, parsed_data)
            logger.info(f"Saved data to database for statement {statement_id}")

            # Step 6: Update status
            statement.processing_status = 'completed'
            statement.processing_completed_at = datetime.utcnow()
            db.session.commit()

            # Step 7: Cleanup
            if self.cleanup_temp_files:
                self.pdf_processor.cleanup_images(statement_id)
                logger.info(f"Cleaned up temporary files for statement {statement_id}")

            logger.info(f"Successfully processed statement {statement_id}")
            return True

        except Exception as e:
            logger.exception(f"Error processing statement {statement_id}: {e}")
            statement.processing_status = 'failed'
            statement.error_message = str(e)
            statement.processing_completed_at = datetime.utcnow()
            db.session.commit()
            return False

    def _save_to_database(self, statement_id: int, parsed_data: dict) -> None:
        """Save parsed data to database"""

        # Save customer details
        customer_data = parsed_data.get('customer_details', {}).copy()
        if customer_data:
            # Extract core fields
            account_holder_name = customer_data.pop('account_holder_name', None)
            account_number = customer_data.pop('account_number', None)

            customer = CustomerDetails(
                statement_id=statement_id,
                account_holder_name=account_holder_name,
                account_number=account_number,
                additional_info=customer_data,  # Remaining fields as JSON
                raw_extracted=str(parsed_data.get('customer_details', {}))
            )
            db.session.add(customer)

        # Save bank details
        bank_data = parsed_data.get('bank_details', {}).copy()
        if bank_data:
            # Extract core fields
            bank_name = bank_data.pop('bank_name', None)
            statement_period_start = bank_data.pop('statement_period_start', None)
            statement_period_end = bank_data.pop('statement_period_end', None)
            opening_balance = bank_data.pop('opening_balance', None)
            closing_balance = bank_data.pop('closing_balance', None)
            currency = bank_data.pop('currency', 'BDT')

            # Parse dates if strings
            if statement_period_start and isinstance(statement_period_start, str):
                try:
                    statement_period_start = datetime.strptime(statement_period_start, '%Y-%m-%d').date()
                except ValueError:
                    statement_period_start = None

            if statement_period_end and isinstance(statement_period_end, str):
                try:
                    statement_period_end = datetime.strptime(statement_period_end, '%Y-%m-%d').date()
                except ValueError:
                    statement_period_end = None

            bank = BankDetails(
                statement_id=statement_id,
                bank_name=bank_name,
                statement_period_start=statement_period_start,
                statement_period_end=statement_period_end,
                opening_balance=opening_balance,
                closing_balance=closing_balance,
                currency=currency,
                additional_info=bank_data,  # Remaining fields as JSON
                raw_extracted=str(parsed_data.get('bank_details', {}))
            )
            db.session.add(bank)

        # Save transaction schema
        schema_data = parsed_data.get('transaction_schema', {})
        if schema_data:
            schema = TransactionSchema(
                statement_id=statement_id,
                columns=schema_data.get('columns', []),
                column_metadata=schema_data.get('column_metadata', {}),
                detected_bank_format=None
            )
            db.session.add(schema)

        # Save transactions
        for txn in parsed_data.get('transactions', []):
            transaction = Transaction(
                statement_id=statement_id,
                transaction_date=txn.get('transaction_date'),
                amount=txn.get('amount'),
                transaction_type=txn.get('transaction_type'),
                data=txn.get('data'),
                page_number=txn.get('page_number'),
                row_index=txn.get('row_index')
            )
            db.session.add(transaction)

        db.session.commit()

    def _extract_confidence_scores(self, llm_responses: list) -> dict:
        """Extract and aggregate confidence scores from LLM responses"""
        all_scores = []

        for response in llm_responses:
            scores = response.get('confidence_scores')
            if scores:
                all_scores.append(scores)

        if not all_scores:
            return {}

        # Calculate averages
        avg_scores = {
            'overall': sum(s.get('overall', 0) for s in all_scores) / len(all_scores),
            'customer_details': sum(s.get('customer_details', 0) for s in all_scores) / len(all_scores),
            'bank_details': sum(s.get('bank_details', 0) for s in all_scores) / len(all_scores),
            'transactions': sum(s.get('transactions', 0) for s in all_scores) / len(all_scores)
        }

        return {
            'average': avg_scores,
            'by_page': all_scores,
            'min_overall': min(s.get('overall', 1.0) for s in all_scores),
            'max_overall': max(s.get('overall', 0.0) for s in all_scores)
        }


def process_statement_sync(app, statement_id: int):
    """
    Process statement synchronously

    Args:
        app: Flask application instance
        statement_id: ID of the statement to process
    """
    with app.app_context():
        processor = StatementProcessor(app.config)
        try:
            result = processor.process_statement(statement_id)
            return result
        except Exception as e:
            logger.exception(f"Error processing statement {statement_id}: {e}")
            return False
