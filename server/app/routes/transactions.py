import logging
from decimal import Decimal
from flask import Blueprint, request, jsonify
from sqlalchemy import desc, asc, cast, String
from app.models import db, BankStatement, Transaction

logger = logging.getLogger(__name__)

transactions_bp = Blueprint('transactions', __name__)


@transactions_bp.route('/statements/<int:statement_id>/transactions', methods=['GET'])
def get_statement_transactions(statement_id):
    """
    Get transactions for a specific statement

    Query params:
        - page (default: 1)
        - limit (default: 50)
        - type: 'credit' | 'debit'
        - date_from: YYYY-MM-DD
        - date_to: YYYY-MM-DD
        - min_amount: number
        - max_amount: number
        - search: text search in transaction data
        - sort_by: 'date' | 'amount' (default: date)
        - sort_order: 'asc' | 'desc' (default: asc)
    """
    # Verify statement exists
    BankStatement.query.get_or_404(statement_id)

    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    txn_type = request.args.get('type')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    min_amount = request.args.get('min_amount', type=float)
    max_amount = request.args.get('max_amount', type=float)
    search = request.args.get('search')
    sort_by = request.args.get('sort_by', 'date')
    sort_order = request.args.get('sort_order', 'asc')

    query = Transaction.query.filter_by(statement_id=statement_id)

    # Apply filters
    if txn_type:
        query = query.filter(Transaction.transaction_type == txn_type)

    if date_from:
        query = query.filter(Transaction.transaction_date >= date_from)

    if date_to:
        query = query.filter(Transaction.transaction_date <= date_to)

    if min_amount is not None:
        query = query.filter(Transaction.amount >= Decimal(str(min_amount)))

    if max_amount is not None:
        query = query.filter(Transaction.amount <= Decimal(str(max_amount)))

    if search:
        # Search in JSON data - convert JSON to string for search
        query = query.filter(cast(Transaction.data, String).ilike(f"%{search}%"))

    # Apply sorting
    if sort_by == 'amount':
        order_col = Transaction.amount
    else:
        order_col = Transaction.transaction_date

    if sort_order == 'desc':
        query = query.order_by(desc(order_col), desc(Transaction.row_index))
    else:
        query = query.order_by(asc(order_col), asc(Transaction.row_index))

    pagination = query.paginate(page=page, per_page=limit, error_out=False)

    transactions = [txn.to_dict() for txn in pagination.items]

    # Calculate summary
    total_credit = db.session.query(db.func.sum(Transaction.amount)).filter(
        Transaction.statement_id == statement_id,
        Transaction.transaction_type == 'credit'
    ).scalar() or 0

    total_debit = db.session.query(db.func.sum(Transaction.amount)).filter(
        Transaction.statement_id == statement_id,
        Transaction.transaction_type == 'debit'
    ).scalar() or 0

    return jsonify({
        "transactions": transactions,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": pagination.total,
            "pages": pagination.pages
        },
        "summary": {
            "total_credit": str(total_credit),
            "total_debit": str(total_debit),
            "net": str(total_credit - total_debit)
        }
    })


@transactions_bp.route('/transactions', methods=['GET'])
def search_all_transactions():
    """
    Global transaction search across all statements

    Additional query param:
        - statement_ids: comma-separated list of statement IDs to filter
    """
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 50, type=int)
    txn_type = request.args.get('type')
    date_from = request.args.get('date_from')
    date_to = request.args.get('date_to')
    min_amount = request.args.get('min_amount', type=float)
    max_amount = request.args.get('max_amount', type=float)
    search = request.args.get('search')
    sort_by = request.args.get('sort_by', 'date')
    sort_order = request.args.get('sort_order', 'asc')
    statement_ids_param = request.args.get('statement_ids')

    query = Transaction.query

    # Filter by statement IDs if provided
    if statement_ids_param:
        try:
            statement_ids = [int(sid.strip()) for sid in statement_ids_param.split(',')]
            query = query.filter(Transaction.statement_id.in_(statement_ids))
        except ValueError:
            return jsonify({"error": True, "message": "Invalid statement_ids format"}), 400

    # Apply filters
    if txn_type:
        query = query.filter(Transaction.transaction_type == txn_type)

    if date_from:
        query = query.filter(Transaction.transaction_date >= date_from)

    if date_to:
        query = query.filter(Transaction.transaction_date <= date_to)

    if min_amount is not None:
        query = query.filter(Transaction.amount >= Decimal(str(min_amount)))

    if max_amount is not None:
        query = query.filter(Transaction.amount <= Decimal(str(max_amount)))

    if search:
        query = query.filter(cast(Transaction.data, String).ilike(f"%{search}%"))

    # Apply sorting
    if sort_by == 'amount':
        order_col = Transaction.amount
    else:
        order_col = Transaction.transaction_date

    if sort_order == 'desc':
        query = query.order_by(desc(order_col))
    else:
        query = query.order_by(asc(order_col))

    pagination = query.paginate(page=page, per_page=limit, error_out=False)

    transactions = [txn.to_dict() for txn in pagination.items]

    return jsonify({
        "transactions": transactions,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": pagination.total,
            "pages": pagination.pages
        }
    })
