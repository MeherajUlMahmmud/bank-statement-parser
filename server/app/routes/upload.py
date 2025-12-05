import os
import hashlib
import logging
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.models import db, BankStatement
from app.services.processor import process_statement_sync

logger = logging.getLogger(__name__)

upload_bp = Blueprint('upload', __name__)


def get_date_wise_path(base_folder: str, upload_date: datetime = None) -> str:
    """
    Generate a date-wise folder path for file storage

    Args:
        base_folder: Base upload folder path
        upload_date: Date to use for folder structure (defaults to current date)

    Returns:
        Path in format: base_folder/YYYY/MM/DD
    """
    if upload_date is None:
        upload_date = datetime.utcnow()

    year = upload_date.strftime('%Y')
    month = upload_date.strftime('%m')
    day = upload_date.strftime('%d')

    return os.path.join(base_folder, year, month, day)


@upload_bp.route('/upload', methods=['POST'])
def upload_statement():
    """
    Upload a PDF bank statement for processing

    Request: multipart/form-data with 'file' field
    Response: {
        "success": true,
        "statement_id": 123,
        "message": "File uploaded and processing started"
    }
    """
    if 'file' not in request.files:
        return jsonify({"error": True, "message": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": True, "message": "No file selected"}), 400

    if not file.filename.lower().endswith('.pdf'):
        return jsonify({"error": True, "message": "Only PDF files are allowed"}), 400

    try:
        # Calculate file hash to prevent duplicates
        file_content = file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        file.seek(0)

        # Check for duplicate
        existing = BankStatement.query.filter_by(file_hash=file_hash).first()
        if existing:
            return jsonify({
                "error": True,
                "message": "This file has already been uploaded",
                "existing_statement_id": existing.id
            }), 409

        # Save file in date-wise folder structure
        original_filename = secure_filename(file.filename)
        upload_date = datetime.utcnow()

        # Create date-wise folder path (YYYY/MM/DD)
        date_folder = get_date_wise_path(current_app.config['UPLOAD_FOLDER'], upload_date)

        # Ensure date-wise folder exists
        os.makedirs(date_folder, exist_ok=True)

        # Generate unique filename with hash
        filename = f"{file_hash[:16]}_{original_filename}"
        filepath = os.path.join(date_folder, filename)

        # Calculate relative path from base upload folder for database storage
        relative_path = os.path.relpath(filepath, current_app.config['UPLOAD_FOLDER'])

        file.save(filepath)
        logger.info(f"Saved file to {filepath}")

        # Create database record with relative path
        statement = BankStatement(
            filename=relative_path,
            original_filename=original_filename,
            file_hash=file_hash,
            processing_status='pending'
        )
        db.session.add(statement)
        db.session.commit()
        logger.info(f"Created statement record {statement.id}")

        # Process synchronously
        success = process_statement_sync(current_app._get_current_object(), statement.id)

        if success:
            return jsonify({
                "success": True,
                "statement_id": statement.id,
                "message": "File uploaded and processed successfully"
            }), 200
        else:
            return jsonify({
                "success": False,
                "statement_id": statement.id,
                "message": "File uploaded but processing failed"
            }), 500

    except Exception as e:
        logger.exception(f"Error uploading file: {e}")
        return jsonify({"error": True, "message": str(e)}), 500
