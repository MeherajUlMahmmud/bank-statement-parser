import logging
import csv
from io import StringIO
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ....core.database import get_db
from ....models.statement import BankStatement, Transaction, CustomerDetails, BankDetails

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/{statement_id}/csv")
async def export_to_csv(
    statement_id: str,
    db: AsyncSession = Depends(get_db)
):
    """
    Export statement transactions to CSV format.
    
    CSV includes:
    - Statement header (bank, account, period)
    - All transactions with date, description, debit, credit, balance
    - Summary totals
    """
    logger.info(f"Exporting statement {statement_id} to CSV")
    
    # Get statement
    result = await db.execute(
        select(BankStatement).where(BankStatement.id == statement_id)
    )
    statement = result.scalar_one_or_none()
    
    if not statement:
        raise HTTPException(status_code=404, detail="Statement not found")
    
    if statement.status.value != "completed":
        raise HTTPException(status_code=400, detail="Statement processing not completed")
    
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
        select(Transaction)
        .where(Transaction.statement_id == statement_id)
        .order_by(Transaction.date)
    )
    transactions = transactions_result.scalars().all()
    
    # Create CSV
    output = StringIO()
    writer = csv.writer(output)
    
    # Header section
    writer.writerow(["Bank Statement Export"])
    writer.writerow([])
    
    if bank:
        writer.writerow(["Bank Name:", bank.bank_name or "N/A"])
        writer.writerow(["Branch:", bank.branch_name or "N/A"])
        writer.writerow(["Currency:", bank.currency or "USD"])
        writer.writerow([])
    
    if customer:
        writer.writerow(["Account Holder:", customer.account_holder_name or "N/A"])
        writer.writerow(["Account Number:", customer.account_number_masked or "N/A"])
        writer.writerow(["Account Type:", customer.account_type or "N/A"])
        writer.writerow([])
    
    if bank:
        writer.writerow(["Statement Period:", 
                        f"{bank.period_start_date.date() if bank.period_start_date else 'N/A'} to {bank.period_end_date.date() if bank.period_end_date else 'N/A'}"])
        writer.writerow(["Opening Balance:", f"{bank.opening_balance:.2f}" if bank.opening_balance else "N/A"])
        writer.writerow(["Closing Balance:", f"{bank.closing_balance:.2f}" if bank.closing_balance else "N/A"])
        writer.writerow([])
    
    # Transactions table
    writer.writerow(["TRANSACTIONS"])
    writer.writerow(["Date", "Description", "Debit", "Credit", "Balance"])
    
    for txn in transactions:
        writer.writerow([
            txn.date.date() if txn.date else "",
            txn.description or "",
            f"{txn.debit:.2f}" if txn.debit else "0.00",
            f"{txn.credit:.2f}" if txn.credit else "0.00",
            f"{txn.balance:.2f}" if txn.balance else "",
        ])
    
    # Summary
    writer.writerow([])
    if bank:
        writer.writerow(["SUMMARY"])
        writer.writerow(["Total Debits:", f"{bank.total_debits:.2f}" if bank.total_debits else "0.00"])
        writer.writerow(["Total Credits:", f"{bank.total_credits:.2f}" if bank.total_credits else "0.00"])
        writer.writerow(["Final Balance:", f"{bank.closing_balance:.2f}" if bank.closing_balance else "0.00"])
    
    # Return as downloadable file
    output.seek(0)
    filename = f"statement_{statement_id}_{statement.created_at.strftime('%Y%m%d')}.csv"
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )
