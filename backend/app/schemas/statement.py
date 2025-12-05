from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class TransactionResponse(BaseModel):
    """Response model for a single transaction."""
    id: str
    date: Optional[datetime]
    description: Optional[str]
    debit: Optional[float]
    credit: Optional[float]
    balance: Optional[float]
    confidence: Optional[float]
    
    class Config:
        from_attributes = True


class CustomerDetailsResponse(BaseModel):
    """Response model for customer details."""
    account_holder_name: Optional[str]
    account_number_masked: Optional[str]
    account_type: Optional[str]
    
    class Config:
        from_attributes = True


class BankDetailsResponse(BaseModel):
    """Response model for bank details."""
    bank_name: Optional[str]
    branch_name: Optional[str]
    period_start_date: Optional[datetime]
    period_end_date: Optional[datetime]
    opening_balance: Optional[float]
    closing_balance: Optional[float]
    currency: str = "USD"
    
    class Config:
        from_attributes = True


class StatementResponse(BaseModel):
    """Response model for a complete bank statement."""
    id: str
    filename: str
    status: str
    page_count: int
    total_transactions: int
    overall_confidence: Optional[float]
    created_at: datetime
    customer_details: Optional[CustomerDetailsResponse]
    bank_details: Optional[BankDetailsResponse]
    transactions: List[TransactionResponse] = []
    
    class Config:
        from_attributes = True


class StatementListResponse(BaseModel):
    """Response model for list of statements."""
    total: int
    statements: List[StatementResponse]


class ProcessingStatusResponse(BaseModel):
    """Response model for processing status."""
    id: str
    status: str
    progress: Dict[str, Any] = {}
    error: Optional[str] = None
    
    class Config:
        from_attributes = True


class UploadResponse(BaseModel):
    """Response model for file upload."""
    statement_id: str
    filename: str
    status: str
    message: str
