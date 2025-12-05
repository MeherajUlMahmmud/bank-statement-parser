import os
import logging
from flask import Blueprint, request, jsonify, send_file, current_app
from sqlalchemy import desc, or_
from app.models import db, BankStatement, CustomerDetails, BankDetails, Transaction, TransactionSchema

logger = logging.getLogger(__name__)

statements_bp = Blueprint('statements', __name__)


@statements_bp.route('/statements', methods=['GET'])
def list_statements():
    """
    List all statements with pagination

    Query params:
        - page (default: 1)
        - limit (default: 20)
        - status (filter by processing status)
        - search (search in filename, bank name, account number)
    """
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 20, type=int)
    status = request.args.get('status')
    search = request.args.get('search')

    query = BankStatement.query

    if status:
        query = query.filter(BankStatement.processing_status == status)

    if search:
        search_term = f"%{search}%"
        query = query.outerjoin(BankDetails).outerjoin(CustomerDetails).filter(
            or_(
                BankStatement.original_filename.ilike(search_term),
                BankDetails.bank_name.ilike(search_term),
                CustomerDetails.account_number.ilike(search_term)
            )
        )

    query = query.order_by(desc(BankStatement.upload_date))
    pagination = query.paginate(page=page, per_page=limit, error_out=False)

    statements = [stmt.to_dict() for stmt in pagination.items]

    return jsonify({
        "statements": statements,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": pagination.total,
            "pages": pagination.pages
        }
    })


@statements_bp.route('/statements/<int:statement_id>', methods=['GET'])
def get_statement(statement_id):
    """
    Get full statement details including customer and bank info
    """
    statement = BankStatement.query.get_or_404(statement_id)

    customer = statement.customer_details
    bank = statement.bank_details

    return jsonify({
        "id": statement.id,
        "filename": statement.original_filename,
        "upload_date": statement.upload_date.isoformat(),
        "status": statement.processing_status,
        "page_count": statement.page_count,
        "customer_details": customer.to_dict() if customer else None,
        "bank_details": bank.to_dict() if bank else None,
        "transaction_count": len(statement.transactions)
    })


@statements_bp.route('/statements/<int:statement_id>', methods=['DELETE'])
def delete_statement(statement_id):
    """
    Delete a statement and all related data
    """
    statement = BankStatement.query.get_or_404(statement_id)

    try:
        # Delete related records (cascade should handle this)
        Transaction.query.filter_by(statement_id=statement_id).delete()
        TransactionSchema.query.filter_by(statement_id=statement_id).delete()
        CustomerDetails.query.filter_by(statement_id=statement_id).delete()
        BankDetails.query.filter_by(statement_id=statement_id).delete()

        # Delete the statement
        db.session.delete(statement)
        db.session.commit()

        logger.info(f"Deleted statement {statement_id}")

        return jsonify({"success": True, "message": "Statement deleted"})

    except Exception as e:
        logger.exception(f"Error deleting statement {statement_id}: {e}")
        db.session.rollback()
        return jsonify({"error": True, "message": str(e)}), 500


@statements_bp.route('/statements/<int:statement_id>/schema', methods=['GET'])
def get_statement_schema(statement_id):
    """
    Get transaction column schema for dynamic rendering
    """
    # Verify statement exists
    BankStatement.query.get_or_404(statement_id)

    schema = TransactionSchema.query.filter_by(statement_id=statement_id).first()

    if not schema:
        return jsonify({
            "columns": [],
            "column_metadata": {},
            "detected_bank_format": None
        })

    return jsonify(schema.to_dict())


@statements_bp.route('/statements/<int:statement_id>/summary', methods=['GET'])
def get_statement_summary(statement_id):
    """
    Get summary statistics for a statement
    """
    statement = BankStatement.query.get_or_404(statement_id)

    # Calculate transaction summaries
    total_transactions = Transaction.query.filter_by(statement_id=statement_id).count()

    total_credit = db.session.query(db.func.sum(Transaction.amount)).filter(
        Transaction.statement_id == statement_id,
        Transaction.transaction_type == 'credit'
    ).scalar() or 0

    total_debit = db.session.query(db.func.sum(Transaction.amount)).filter(
        Transaction.statement_id == statement_id,
        Transaction.transaction_type == 'debit'
    ).scalar() or 0

    credit_count = Transaction.query.filter_by(
        statement_id=statement_id,
        transaction_type='credit'
    ).count()

    debit_count = Transaction.query.filter_by(
        statement_id=statement_id,
        transaction_type='debit'
    ).count()

    return jsonify({
        "statement_id": statement_id,
        "total_transactions": total_transactions,
        "total_credit": str(total_credit),
        "total_debit": str(total_debit),
        "net": str(total_credit - total_debit),
        "credit_count": credit_count,
        "debit_count": debit_count,
        "opening_balance": str(statement.bank_details.opening_balance) if statement.bank_details and statement.bank_details.opening_balance else None,
        "closing_balance": str(statement.bank_details.closing_balance) if statement.bank_details and statement.bank_details.closing_balance else None
    })


@statements_bp.route('/statements/<int:statement_id>/metadata', methods=['GET'])
def get_statement_metadata(statement_id):
    """
    Get detailed extraction metadata including raw data, logs, token usage, and confidence scores
    """
    statement = BankStatement.query.get_or_404(statement_id)

    # Get transaction schema if available
    schema = TransactionSchema.query.filter_by(statement_id=statement_id).first()
    schema_data = schema.to_dict() if schema else None

    return jsonify({
        "statement_id": statement_id,
        "raw_extraction_data": statement.raw_extraction_data or [],
        "token_usage": statement.token_usage or {},
        "confidence_scores": statement.confidence_scores or {},
        "processing_logs": statement.processing_logs or [],
        "validation_errors": statement.validation_errors or {},
        "transaction_schema": schema_data,
        "processing_times": {
            "started_at": statement.processing_started_at.isoformat() if statement.processing_started_at else None,
            "completed_at": statement.processing_completed_at.isoformat() if statement.processing_completed_at else None,
            "duration_seconds": (statement.processing_completed_at - statement.processing_started_at).total_seconds()
                if statement.processing_started_at and statement.processing_completed_at else None
        },
        "page_count": statement.page_count,
        "status": statement.processing_status,
        "error_message": statement.error_message
    })


@statements_bp.route('/statements/<int:statement_id>/file', methods=['GET'])
def get_statement_file(statement_id):
    """
    Serve the original PDF file for a statement
    """
    statement = BankStatement.query.get_or_404(statement_id)

    # Build full file path
    upload_folder = current_app.config['UPLOAD_FOLDER']
    file_path = os.path.join(upload_folder, statement.filename)

    if not os.path.exists(file_path):
        logger.error(f"PDF file not found for statement {statement_id}: {file_path}")
        return jsonify({"error": True, "message": "File not found"}), 404

    try:
        return send_file(
            file_path,
            mimetype='application/pdf',
            as_attachment=False,  # Display in browser instead of downloading
            download_name=statement.original_filename
        )
    except Exception as e:
        logger.exception(f"Error serving file for statement {statement_id}: {e}")
        return jsonify({"error": True, "message": str(e)}), 500


@statements_bp.route('/statements/<int:statement_id>/schema', methods=['PUT'])
def update_statement_schema(statement_id):
    """
    Update transaction column metadata (types and display names)

    Request body:
    {
        "column_metadata": {
            "ColumnName": {
                "type": "date|currency|text|number",
                "display_name": "ColumnName"
            }
        }
    }
    """
    # Verify statement exists
    BankStatement.query.get_or_404(statement_id)

    schema = TransactionSchema.query.filter_by(statement_id=statement_id).first()

    if not schema:
        return jsonify({"error": True, "message": "Schema not found for this statement"}), 404

    data = request.get_json()

    if not data or 'column_metadata' not in data:
        return jsonify({"error": True, "message": "column_metadata is required"}), 400

    column_metadata = data['column_metadata']

    # Validate the structure
    valid_types = ['date', 'currency', 'text', 'number']
    for column, meta in column_metadata.items():
        if not isinstance(meta, dict):
            return jsonify({"error": True, "message": f"Invalid metadata for column '{column}'"}), 400

        if 'type' not in meta or 'display_name' not in meta:
            return jsonify({"error": True, "message": f"Column '{column}' must have 'type' and 'display_name'"}), 400

        if meta['type'] not in valid_types:
            return jsonify({"error": True, "message": f"Invalid type '{meta['type']}' for column '{column}'. Must be one of: {', '.join(valid_types)}"}), 400

    try:
        # Update the column metadata
        schema.column_metadata = column_metadata
        db.session.commit()

        logger.info(f"Updated schema for statement {statement_id}")

        return jsonify({
            "success": True,
            "message": "Schema updated successfully",
            "schema": schema.to_dict()
        })

    except Exception as e:
        logger.exception(f"Error updating schema for statement {statement_id}: {e}")
        db.session.rollback()
        return jsonify({"error": True, "message": str(e)}), 500
