import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Integer, Float, DateTime, Text, ForeignKey, JSON, Enum as SQLEnum
from sqlalchemy.orm import relationship, declarative_base
import enum

Base = declarative_base()


class ProcessingStatus(str, enum.Enum):
    """Processing status enumeration."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class BankStatement(Base):
    """
    Main bank statement model representing uploaded PDF documents.
    """
    __tablename__ = "bank_statements"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_hash = Column(String(64), unique=True, nullable=False, index=True)
    file_size = Column(Integer, nullable=False)

    # Processing status
    status = Column(SQLEnum(ProcessingStatus), default=ProcessingStatus.PENDING, nullable=False, index=True)
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    processing_error = Column(Text, nullable=True)

    # Document classification
    document_type = Column(String(50), default="bank_statement", nullable=False)
    classification_confidence = Column(Float, nullable=True)

    # Metadata
    page_count = Column(Integer, default=0)
    total_transactions = Column(Integer, default=0)

    # AI/Processing metadata
    model_used = Column(String(100), nullable=True)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    processing_time_seconds = Column(Float, default=0.0)

    # Overall confidence score
    overall_confidence = Column(Float, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    customer_details = relationship("CustomerDetails", back_populates="statement", uselist=False, cascade="all, delete-orphan")
    bank_details = relationship("BankDetails", back_populates="statement", uselist=False, cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="statement", cascade="all, delete-orphan", order_by="Transaction.date")
    processing_logs = relationship("ProcessingLog", back_populates="statement", cascade="all, delete-orphan", order_by="ProcessingLog.created_at")

    def __repr__(self):
        return f"<BankStatement(id={self.id}, filename={self.filename}, status={self.status})>"


class CustomerDetails(Base):
    """
    Customer/account holder details extracted from statement.
    """
    __tablename__ = "customer_details"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    statement_id = Column(String(36), ForeignKey("bank_statements.id", ondelete="CASCADE"), nullable=False, unique=True)

    # Customer information
    account_holder_name = Column(String(255), nullable=True)
    account_number = Column(String(100), nullable=True)
    account_number_masked = Column(String(100), nullable=True)  # PII-safe version
    account_type = Column(String(50), nullable=True)

    # Contact information
    address = Column(Text, nullable=True)
    email = Column(String(255), nullable=True)
    phone = Column(String(50), nullable=True)

    # Additional details
    customer_id = Column(String(100), nullable=True)
    branch_code = Column(String(50), nullable=True)

    # Confidence scores
    confidence_scores = Column(JSON, nullable=True)  # Field-level confidence scores

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    statement = relationship("BankStatement", back_populates="customer_details")

    def __repr__(self):
        return f"<CustomerDetails(id={self.id}, account_holder={self.account_holder_name})>"


class BankDetails(Base):
    """
    Bank and statement period details.
    """
    __tablename__ = "bank_details"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    statement_id = Column(String(36), ForeignKey("bank_statements.id", ondelete="CASCADE"), nullable=False, unique=True)

    # Bank information
    bank_name = Column(String(255), nullable=True)
    bank_code = Column(String(50), nullable=True)
    branch_name = Column(String(255), nullable=True)
    branch_address = Column(Text, nullable=True)

    # Statement period
    statement_date = Column(DateTime, nullable=True)
    period_start_date = Column(DateTime, nullable=True)
    period_end_date = Column(DateTime, nullable=True)

    # Balances
    opening_balance = Column(Float, nullable=True)
    closing_balance = Column(Float, nullable=True)
    currency = Column(String(3), default="USD", nullable=False)  # ISO 4217 currency code

    # Summary totals
    total_debits = Column(Float, nullable=True)
    total_credits = Column(Float, nullable=True)

    # Confidence scores
    confidence_scores = Column(JSON, nullable=True)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    statement = relationship("BankStatement", back_populates="bank_details")

    def __repr__(self):
        return f"<BankDetails(id={self.id}, bank={self.bank_name}, period={self.period_start_date} to {self.period_end_date})>"


class Transaction(Base):
    """
    Individual transaction extracted from bank statement.
    Flexible schema preserves original column names.
    """
    __tablename__ = "transactions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    statement_id = Column(String(36), ForeignKey("bank_statements.id", ondelete="CASCADE"), nullable=False, index=True)

    # Transaction basics
    date = Column(DateTime, nullable=True, index=True)
    description = Column(Text, nullable=True)

    # Amounts
    debit = Column(Float, nullable=True)
    credit = Column(Float, nullable=True)
    balance = Column(Float, nullable=True)
    amount = Column(Float, nullable=True)  # Generic amount field

    # Transaction metadata
    transaction_type = Column(String(50), nullable=True)  # withdrawal, deposit, transfer, etc.
    reference_number = Column(String(100), nullable=True)
    check_number = Column(String(50), nullable=True)
    category = Column(String(100), nullable=True)

    # Flexible data storage
    raw_data = Column(JSON, nullable=True)  # Preserves all original fields

    # Confidence scores
    confidence = Column(Float, nullable=True)
    confidence_scores = Column(JSON, nullable=True)  # Field-level confidence

    # Page reference
    page_number = Column(Integer, nullable=True)
    bbox = Column(JSON, nullable=True)  # Bounding box coordinates [x, y, width, height]

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)

    # Relationships
    statement = relationship("BankStatement", back_populates="transactions")

    def __repr__(self):
        return f"<Transaction(id={self.id}, date={self.date}, amount={self.debit or self.credit or self.amount})>"


class ProcessingLog(Base):
    """
    Logs for tracking processing steps and debugging.
    """
    __tablename__ = "processing_logs"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    statement_id = Column(String(36), ForeignKey("bank_statements.id", ondelete="CASCADE"), nullable=False, index=True)

    # Log details
    step = Column(String(100), nullable=False)  # pdf_extraction, ocr, cleanup, extraction, normalization
    status = Column(String(20), nullable=False)  # started, completed, failed
    message = Column(Text, nullable=True)

    # Timing
    duration_seconds = Column(Float, nullable=True)

    # Metadata
    metadata = Column(JSON, nullable=True)  # Additional context (tokens, model, errors, etc.)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)

    # Relationships
    statement = relationship("BankStatement", back_populates="processing_logs")

    def __repr__(self):
        return f"<ProcessingLog(id={self.id}, step={self.step}, status={self.status})>"
