import hashlib
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, BinaryIO
from fastapi import UploadFile

from ..core.config import settings

logger = logging.getLogger(__name__)


class StorageService:
    """
    Service for handling file uploads with date-wise organization.
    Supports hash-based deduplication and organized file storage.
    """

    def __init__(self, base_path: Optional[str] = None):
        """
        Initialize storage service.

        Args:
            base_path: Base directory for file storage (defaults to settings.UPLOAD_DIR)
        """
        self.base_path = Path(base_path or settings.UPLOAD_DIR)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"StorageService initialized with base_path: {self.base_path}")

    def _calculate_file_hash(self, file_content: bytes) -> str:
        """
        Calculate SHA256 hash of file content.

        Args:
            file_content: File content as bytes

        Returns:
            str: SHA256 hash in hexadecimal
        """
        return hashlib.sha256(file_content).hexdigest()

    def _get_date_organized_path(self, filename: str, upload_date: Optional[datetime] = None) -> Path:
        """
        Get date-organized path for file storage (YYYY/MM/DD/filename).

        Args:
            filename: Original filename
            upload_date: Upload date (defaults to current date)

        Returns:
            Path: Organized file path
        """
        date = upload_date or datetime.now()
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')

        dir_path = self.base_path / year / month / day
        dir_path.mkdir(parents=True, exist_ok=True)

        return dir_path / filename

    async def save_upload_file(
            self,
            upload_file: UploadFile,
            use_hash_name: bool = False,
            check_duplicate: bool = True
    ) -> dict:
        """
        Save an uploaded file with optional deduplication.

        Args:
            upload_file: FastAPI UploadFile object
            use_hash_name: Whether to rename file using its hash
            check_duplicate: Whether to check for duplicates

        Returns:
            dict: {
                'path': str,  # Saved file path
                'hash': str,  # File hash
                'size': int,  # File size in bytes
                'duplicate': bool,  # Whether file already existed
                'original_filename': str  # Original filename
            }
        """
        # Read file content
        content = await upload_file.read()
        file_hash = self._calculate_file_hash(content)
        file_size = len(content)

        # Check for duplicates
        is_duplicate = False
        if check_duplicate:
            existing_file = self._find_file_by_hash(file_hash)
            if existing_file:
                logger.info(f"Duplicate file detected: {upload_file.filename} (hash: {file_hash[:8]}...)")
                return {
                    'path': str(existing_file),
                    'hash': file_hash,
                    'size': file_size,
                    'duplicate': True,
                    'original_filename': upload_file.filename or 'unknown'
                }

        # Determine filename
        if use_hash_name:
            # Use hash as filename, preserve extension
            ext = Path(upload_file.filename or '').suffix
            filename = f"{file_hash}{ext}"
        else:
            # Use original filename
            filename = upload_file.filename or f"{file_hash}.bin"

        # Get organized path
        file_path = self._get_date_organized_path(filename)

        # Handle filename conflicts
        if file_path.exists():
            base = file_path.stem
            ext = file_path.suffix
            counter = 1
            while file_path.exists():
                filename = f"{base}_{counter}{ext}"
                file_path = file_path.parent / filename
                counter += 1

        # Save file
        try:
            with open(file_path, 'wb') as f:
                f.write(content)

            logger.info(f"File saved: {file_path} ({file_size} bytes)")

            return {
                'path': str(file_path),
                'hash': file_hash,
                'size': file_size,
                'duplicate': is_duplicate,
                'original_filename': upload_file.filename or 'unknown'
            }

        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise

    def save_file(self, file_content: bytes, filename: str, use_hash_name: bool = False) -> dict:
        """
        Save file content directly.

        Args:
            file_content: File content as bytes
            filename: Filename
            use_hash_name: Whether to use hash as filename

        Returns:
            dict: Save result with path, hash, size
        """
        file_hash = self._calculate_file_hash(file_content)
        file_size = len(file_content)

        # Determine filename
        if use_hash_name:
            ext = Path(filename).suffix
            filename = f"{file_hash}{ext}"

        # Get organized path
        file_path = self._get_date_organized_path(filename)

        # Handle conflicts
        if file_path.exists():
            base = file_path.stem
            ext = file_path.suffix
            counter = 1
            while file_path.exists():
                filename = f"{base}_{counter}{ext}"
                file_path = file_path.parent / filename
                counter += 1

        # Save file
        try:
            with open(file_path, 'wb') as f:
                f.write(file_content)

            logger.info(f"File saved: {file_path} ({file_size} bytes)")

            return {
                'path': str(file_path),
                'hash': file_hash,
                'size': file_size,
                'original_filename': filename
            }

        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise

    def _find_file_by_hash(self, file_hash: str) -> Optional[Path]:
        """
        Find a file by its hash by searching the storage directory.

        Args:
            file_hash: File hash to search for

        Returns:
            Path: File path if found, None otherwise
        """
        # Search through year/month/day structure
        for year_dir in self.base_path.glob('*'):
            if not year_dir.is_dir():
                continue
            for month_dir in year_dir.glob('*'):
                if not month_dir.is_dir():
                    continue
                for day_dir in month_dir.glob('*'):
                    if not day_dir.is_dir():
                        continue
                    for file_path in day_dir.glob('*'):
                        if not file_path.is_file():
                            continue
                        # Check if filename contains hash or calculate hash
                        if file_hash in file_path.name:
                            return file_path
                        # Optionally calculate hash (expensive)
                        try:
                            with open(file_path, 'rb') as f:
                                content = f.read()
                                if self._calculate_file_hash(content) == file_hash:
                                    return file_path
                        except Exception:
                            continue
        return None

    def get_file_path(self, path_str: str) -> Path:
        """Get Path object from string path."""
        return Path(path_str)

    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists.

        Args:
            path: File path

        Returns:
            bool: True if file exists
        """
        try:
            return Path(path).exists()
        except Exception as e:
            logger.error(f"Error checking file existence: {str(e)}")
            return False

    def delete_file(self, path: str) -> bool:
        """
        Delete a file.

        Args:
            path: File path to delete

        Returns:
            bool: True if deleted successfully
        """
        try:
            file_path = Path(path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"File deleted: {path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return False

    def get_file_size(self, path: str) -> int:
        """
        Get file size in bytes.

        Args:
            path: File path

        Returns:
            int: File size in bytes
        """
        try:
            return Path(path).stat().st_size
        except Exception as e:
            logger.error(f"Error getting file size: {str(e)}")
            return 0

    def get_file_url(self, path: str) -> str:
        """
        Get URL for accessing a file (for local storage, returns relative path).

        Args:
            path: File path

        Returns:
            str: URL or path to access file
        """
        try:
            # For local storage, return path relative to base
            file_path = Path(path)
            if file_path.is_absolute():
                try:
                    relative = file_path.relative_to(self.base_path)
                    return f"/uploads/{relative}"
                except ValueError:
                    return str(file_path)
            return str(file_path)
        except Exception as e:
            logger.error(f"Error getting file URL: {str(e)}")
            return ""
