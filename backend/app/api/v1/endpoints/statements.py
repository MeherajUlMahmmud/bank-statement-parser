import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from ....core.database import get_db
from ....models.statement import BankStatement, Transaction, CustomerDetails, BankDetails, ProcessingStatus
from ....schemas.statement import (
    StatementResponse,
    StatementListResponse,
    UploadResponse,
    ProcessingStatusResponse,
    CustomerDetailsResponse,
    BankDetailsResponse,
    TransactionResponse,
)
from ....services.storage_service import StorageService
from ....services.pipeline_orchestrator import PipelineOrchestrator

router = APIRouter()
logger = logging.getLogger(__name__)


async def process_statement_background(statement_id: str, file_path: str):
    """Background task to process uploaded statement."""
    from ....core.database import AsyncSessionLocal
    from datetime import datetime
    
    logger.info(f"Starting background processing for statement: {statement_id}")
    
    async with AsyncSessionLocal() as db:
        try:
            # Update status to processing
            result = await db.execute(
                select(BankStatement).where(BankStatement.id == statement_id)
            )
            statement = result.scalar_one_or_none()
            
            if not statement:
                logger.error(f"Statement {statement_id} not found")
                return
            
            statement.status = ProcessingStatus.PROCESSING
            statement.processing_started_at = datetime.utcnow()
            await db.commit()
            
            # Run pipeline
            orchestrator = PipelineOrchestrator()
            result = await orchestrator.process_bank_statement(file_path)
            
            if result['success']:
                # Save results to database
                final_data = result['final_data']
                validation = result['validation_results']
                
                # Update statement
                statement.status = ProcessingStatus.COMPLETED
                statement.processing_completed_at = datetime.utcnow()
                statement.page_count = result['metadata']['pdf_pages']
                statement.total_transactions = result['metadata']['transactions_found']
                statement.overall_confidence = result['metadata']['overall_confidence']
                statement.processing_time_seconds = result['metadata']['total_duration']

                # Save schema information (flexible column structure)
                if 'schema_info' in final_data:
                    statement.schema_info = final_data['schema_info']

                # Save customer details
                if 'account' in final_data:
                    customer = CustomerDetails(
                        statement_id=statement_id,
                        account_holder_name=final_data['account'].get('account_holder', {}).get('value'),
                        account_number_masked=final_data['account'].get('account_number', {}).get('value'),
                        account_type=final_data['account'].get('account_type', {}).get('value'),
                    )
                    db.add(customer)
                
                # Save bank details
                if 'bank' in final_data and 'period' in final_data and 'balances' in final_data:
                    from datetime import datetime as dt
                    bank = BankDetails(
                        statement_id=statement_id,
                        bank_name=final_data['bank'].get('bank_name', {}).get('value'),
                        branch_name=final_data['bank'].get('branch_name', {}).get('value'),
                        currency=final_data['bank'].get('currency', {}).get('value', 'USD'),
                        period_start_date=dt.fromisoformat(final_data['period']['start_date']['value']) if final_data['period'].get('start_date', {}).get('value') else None,
                        period_end_date=dt.fromisoformat(final_data['period']['end_date']['value']) if final_data['period'].get('end_date', {}).get('value') else None,
                        opening_balance=final_data['balances'].get('opening_balance', {}).get('value'),
                        closing_balance=final_data['balances'].get('closing_balance', {}).get('value'),
                        total_debits=final_data['balances'].get('total_debits', {}).get('value'),
                        total_credits=final_data['balances'].get('total_credits', {}).get('value'),
                    )
                    db.add(bank)
                
                # Save transactions
                if 'transactions' in final_data:
                    for txn_data in final_data['transactions']:
                        from datetime import datetime as dt
                        transaction = Transaction(
                            statement_id=statement_id,
                            date=dt.fromisoformat(txn_data['date']['value']) if txn_data.get('date', {}).get('value') else None,
                            description=txn_data.get('description', {}).get('value'),
                            debit=txn_data.get('debit', {}).get('value'),
                            credit=txn_data.get('credit', {}).get('value'),
                            balance=txn_data.get('balance', {}).get('value'),
                            confidence=txn_data.get('date', {}).get('confidence', 0.0),
                            raw_data=txn_data,
                        )
                        db.add(transaction)
                
                await db.commit()
                logger.info(f"Statement {statement_id} processed successfully")
            else:
                # Processing failed
                statement.status = ProcessingStatus.FAILED
                statement.processing_completed_at = datetime.utcnow()
                statement.processing_error = '; '.join(result.get('errors', ['Unknown error']))
                await db.commit()
                logger.error(f"Statement {statement_id} processing failed: {statement.processing_error}")
                
        except Exception as e:
            logger.error(f"Error processing statement {statement_id}: {str(e)}", exc_info=True)
            # Update status to failed
            try:
                result = await db.execute(
                    select(BankStatement).where(BankStatement.id == statement_id)
                )
                statement = result.scalar_one_or_none()
                if statement:
                    statement.status = ProcessingStatus.FAILED
                    statement.processing_completed_at = datetime.utcnow()
                    statement.processing_error = str(e)
                    await db.commit()
            except Exception:
                pass


@router.post("/upload", response_model=UploadResponse)
async def upload_statement(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db)
):
    """
    Upload a bank statement PDF for processing.
    
    The file will be processed in the background through the multi-agent pipeline:
    1. PDF to Images
    2. OCR Extraction
    3. Text Cleanup
    4. Data Extraction
    5. Normalization & Validation
    """
    logger.info(f"Received upload request: {file.filename}")
    
    # Validate file type
    if not file.filename or not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Save file
        storage = StorageService()
        save_result = await storage.save_upload_file(file, check_duplicate=True)
        
        # Check if duplicate
        if save_result['duplicate']:
            # Find existing statement
            result = await db.execute(
                select(BankStatement).where(BankStatement.file_hash == save_result['hash'])
            )
            existing = result.scalar_one_or_none()
            
            if existing:
                return UploadResponse(
                    statement_id=existing.id,
                    filename=file.filename,
                    status=existing.status.value,
                    message=f"Duplicate file detected. Using existing statement."
                )
        
        # Create statement record
        statement = BankStatement(
            filename=file.filename,
            file_path=save_result['path'],
            file_hash=save_result['hash'],
            file_size=save_result['size'],
            status=ProcessingStatus.PENDING,
        )
        
        db.add(statement)
        await db.commit()
        await db.refresh(statement)
        
        # Add background task for processing
        background_tasks.add_task(
            process_statement_background,
            statement.id,
            save_result['path']
        )
        
        logger.info(f"Statement {statement.id} created, processing queued")
        
        return UploadResponse(
            statement_id=statement.id,
            filename=file.filename,
            status=statement.status.value,
            message="File uploaded successfully. Processing started in background."
        )
        
    except Exception as e:
        logger.error(f"Error during upload: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{statement_id}/status", response_model=ProcessingStatusResponse)
async def get_processing_status(
    statement_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get the processing status of a statement."""
    result = await db.execute(
        select(BankStatement).where(BankStatement.id == statement_id)
    )
    statement = result.scalar_one_or_none()
    
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")
    
    progress = {
        "page_count": statement.page_count,
        "total_transactions": statement.total_transactions,
        "overall_confidence": statement.overall_confidence,
        "processing_time": statement.processing_time_seconds,
    }
    
    return ProcessingStatusResponse(
        id=statement.id,
        status=statement.status.value,
        progress=progress,
        error=statement.processing_error
    )


@router.get("/{statement_id}", response_model=StatementResponse)
async def get_statement(
    statement_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Get complete statement details with transactions."""
    result = await db.execute(
        select(BankStatement).where(BankStatement.id == statement_id)
    )
    statement = result.scalar_one_or_none()
    
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")
    
    # Get related data
    customer_result = await db.execute(
        select(CustomerDetails).where(CustomerDetails.statement_id == statement_id)
    )
    customer = customer_result.scalar_one_or_none()
    
    bank_result = await db.execute(
        select(BankDetails).where(BankDetails.statement_id == statement_id)
    )
    bank = bank_result.scalar_one_or_none()
    
    transactions_result = await db.execute(
        select(Transaction).where(Transaction.statement_id == statement_id).order_by(Transaction.date)
    )
    transactions = transactions_result.scalars().all()
    
    return StatementResponse(
        id=statement.id,
        filename=statement.filename,
        status=statement.status.value,
        page_count=statement.page_count,
        total_transactions=statement.total_transactions,
        overall_confidence=statement.overall_confidence,
        created_at=statement.created_at,
        customer_details=CustomerDetailsResponse.from_orm(customer) if customer else None,
        bank_details=BankDetailsResponse.from_orm(bank) if bank else None,
        transactions=[TransactionResponse.from_orm(t) for t in transactions]
    )


@router.get("", response_model=StatementListResponse)
async def list_statements(
    skip: int = 0,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """List all statements with pagination."""
    # Get total count
    count_result = await db.execute(select(func.count(BankStatement.id)))
    total = count_result.scalar()
    
    # Get statements
    result = await db.execute(
        select(BankStatement)
        .order_by(BankStatement.created_at.desc())
        .offset(skip)
        .limit(limit)
    )
    statements = result.scalars().all()
    
    return StatementListResponse(
        total=total,
        statements=[StatementResponse(
            id=s.id,
            filename=s.filename,
            status=s.status.value,
            page_count=s.page_count,
            total_transactions=s.total_transactions,
            overall_confidence=s.overall_confidence,
            created_at=s.created_at,
            customer_details=None,
            bank_details=None,
            transactions=[]
        ) for s in statements]
    )


@router.delete("/{statement_id}")
async def delete_statement(
    statement_id: str,
    db: AsyncSession = Depends(get_db)
):
    """Delete a statement and all related data."""
    result = await db.execute(
        select(BankStatement).where(BankStatement.id == statement_id)
    )
    statement = result.scalar_one_or_none()
    
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")
    
    # Delete file
    storage = StorageService()
    storage.delete_file(statement.file_path)
    
    # Delete from database (cascade will handle related records)
    await db.delete(statement)
    await db.commit()
    
    return {"message": "Statement deleted successfully"}
