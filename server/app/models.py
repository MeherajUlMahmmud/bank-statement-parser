from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Index
from sqlalchemy.dialects.sqlite import JSON

db = SQLAlchemy()


class BankStatement(db.Model):
    """Main statement record tracking upload and processing status"""
    __tablename__ = 'bank_statements'

    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    file_hash = db.Column(db.String(64), unique=True, nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    processing_status = db.Column(
        db.String(20),
        default='pending',
        nullable=False
    )  # pending, processing, completed, failed
    processing_started_at = db.Column(db.DateTime, nullable=True)
    processing_completed_at = db.Column(db.DateTime, nullable=True)
    error_message = db.Column(db.Text, nullable=True)
    page_count = db.Column(db.Integer, nullable=True)

    # Extraction metadata
    raw_extraction_data = db.Column(JSON, nullable=True)  # Complete raw response from LLM
    token_usage = db.Column(JSON, nullable=True)  # Token statistics
    confidence_scores = db.Column(JSON, nullable=True)  # AI confidence scores
    processing_logs = db.Column(JSON, default=[], nullable=False)  # Processing logs/events
    validation_errors = db.Column(JSON, nullable=True)  # Pydantic validation errors if any

    # Relationships
    customer_details = db.relationship('CustomerDetails', back_populates='statement', uselist=False, cascade='all, delete-orphan')
    bank_details = db.relationship('BankDetails', back_populates='statement', uselist=False, cascade='all, delete-orphan')
    transactions = db.relationship('Transaction', back_populates='statement', cascade='all, delete-orphan')
    transaction_schema = db.relationship('TransactionSchema', back_populates='statement', uselist=False, cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'filename': self.original_filename,
            'upload_date': self.upload_date.isoformat(),
            'status': self.processing_status,
            'page_count': self.page_count,
            'bank_name': self.bank_details.bank_name if self.bank_details else None,
            'account_number': self.customer_details.account_number if self.customer_details else None,
            'period_start': self.bank_details.statement_period_start.isoformat() if self.bank_details and self.bank_details.statement_period_start else None,
            'period_end': self.bank_details.statement_period_end.isoformat() if self.bank_details and self.bank_details.statement_period_end else None,
            'transaction_count': len(self.transactions)
        }


class CustomerDetails(db.Model):
    """Customer/Account holder information"""
    __tablename__ = 'customer_details'

    id = db.Column(db.Integer, primary_key=True)
    statement_id = db.Column(db.Integer, db.ForeignKey('bank_statements.id'), unique=True, nullable=False)

    # Core fields
    account_holder_name = db.Column(db.String(255), nullable=True)
    account_number = db.Column(db.String(50), nullable=True)

    # Flexible JSON for additional fields
    additional_info = db.Column(JSON, default={}, nullable=False)

    # Raw extracted text for debugging
    raw_extracted = db.Column(db.Text, nullable=True)

    # Relationship
    statement = db.relationship('BankStatement', back_populates='customer_details')

    def to_dict(self):
        result = {
            'account_holder_name': self.account_holder_name,
            'account_number': self.account_number
        }
        if self.additional_info:
            result.update(self.additional_info)
        return result


class BankDetails(db.Model):
    """Bank and statement period information"""
    __tablename__ = 'bank_details'

    id = db.Column(db.Integer, primary_key=True)
    statement_id = db.Column(db.Integer, db.ForeignKey('bank_statements.id'), unique=True, nullable=False)

    # Core fields
    bank_name = db.Column(db.String(255), nullable=True)
    statement_period_start = db.Column(db.Date, nullable=True)
    statement_period_end = db.Column(db.Date, nullable=True)
    opening_balance = db.Column(db.Numeric(15, 2), nullable=True)
    closing_balance = db.Column(db.Numeric(15, 2), nullable=True)
    currency = db.Column(db.String(10), default='BDT', nullable=False)

    # Flexible JSON for additional fields
    additional_info = db.Column(JSON, default={}, nullable=False)

    # Raw extracted text for debugging
    raw_extracted = db.Column(db.Text, nullable=True)

    # Relationship
    statement = db.relationship('BankStatement', back_populates='bank_details')

    def to_dict(self):
        result = {
            'bank_name': self.bank_name,
            'statement_period_start': self.statement_period_start.isoformat() if self.statement_period_start else None,
            'statement_period_end': self.statement_period_end.isoformat() if self.statement_period_end else None,
            'opening_balance': str(self.opening_balance) if self.opening_balance else None,
            'closing_balance': str(self.closing_balance) if self.closing_balance else None,
            'currency': self.currency
        }
        if self.additional_info:
            result.update(self.additional_info)
        return result


class Transaction(db.Model):
    """Individual transaction records with flexible schema"""
    __tablename__ = 'transactions'

    id = db.Column(db.Integer, primary_key=True)
    statement_id = db.Column(db.Integer, db.ForeignKey('bank_statements.id'), nullable=False, index=True)

    # Indexed fields for common queries
    transaction_date = db.Column(db.Date, nullable=True, index=True)
    amount = db.Column(db.Numeric(15, 2), nullable=True, index=True)
    transaction_type = db.Column(db.String(20), nullable=True, index=True)  # credit, debit, or null

    # Complete transaction data as JSON
    data = db.Column(JSON, nullable=False)

    # Page and row info for ordering
    page_number = db.Column(db.Integer, nullable=True)
    row_index = db.Column(db.Integer, nullable=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    statement = db.relationship('BankStatement', back_populates='transactions')

    __table_args__ = (
        Index('idx_statement_date', 'statement_id', 'transaction_date'),
        Index('idx_statement_type', 'statement_id', 'transaction_type'),
    )

    def to_dict(self):
        return {
            'id': self.id,
            'transaction_date': self.transaction_date.isoformat() if self.transaction_date else None,
            'amount': str(self.amount) if self.amount else None,
            'transaction_type': self.transaction_type,
            'data': self.data,
            'page_number': self.page_number
        }


class TransactionSchema(db.Model):
    """Stores the detected column schema for each statement's transactions"""
    __tablename__ = 'transaction_schemas'

    id = db.Column(db.Integer, primary_key=True)
    statement_id = db.Column(db.Integer, db.ForeignKey('bank_statements.id'), unique=True, nullable=False)

    # Detected columns in order
    columns = db.Column(JSON, nullable=False)

    # Column metadata
    column_metadata = db.Column(JSON, default={}, nullable=False)

    # Bank-specific identifier if detected
    detected_bank_format = db.Column(db.String(100), nullable=True)

    # Relationship
    statement = db.relationship('BankStatement', back_populates='transaction_schema')

    def to_dict(self):
        return {
            'columns': self.columns,
            'column_metadata': self.column_metadata,
            'detected_bank_format': self.detected_bank_format
        }
