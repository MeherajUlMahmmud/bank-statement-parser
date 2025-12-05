import logging
import os
from typing import Optional
from django.core.files.storage import default_storage
from django.conf import settings

logger = logging.getLogger(__name__)


class StorageService:
    """
    Abstract storage service for handling file uploads.
    Supports both local storage (dev) and S3 (production).
    """

    def __init__(self, storage_type: Optional[str] = None):
        """
        Initialize storage service.
        
        Args:
            storage_type: 'local' or 's3' (defaults to settings.STORAGE_TYPE)
        """
        self.storage_type = storage_type or getattr(settings, 'STORAGE_TYPE', 'local')
        self.storage = default_storage

        # Check if S3 is configured
        if self.storage_type == 's3':
            try:
                from storages.backends.s3boto3 import S3Boto3Storage
                self.storage = S3Boto3Storage()
                logger.info("S3 storage initialized")
            except ImportError:
                logger.warning("django-storages not installed, falling back to local storage")
                self.storage_type = 'local'
            except Exception as e:
                logger.error(f"Failed to initialize S3 storage: {str(e)}, falling back to local")
                self.storage_type = 'local'

        logger.info(f"StorageService initialized with type: {self.storage_type}")

    def save_file(self, file, path: str) -> str:
        """
        Save a file to storage.
        
        Args:
            file: File object or file-like object
            path: Path where file should be saved
        
        Returns:
            str: URL or path to saved file
        """
        try:
            saved_path = self.storage.save(path, file)
            logger.info(f"File saved to {saved_path}")
            return saved_path
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            raise

    def get_file_url(self, path: str) -> str:
        """
        Get URL for a file.
        
        Args:
            path: Path to file
        
        Returns:
            str: URL to access the file
        """
        try:
            if self.storage_type == 's3':
                return self.storage.url(path)
            else:
                # Local storage - return media URL
                return os.path.join(settings.MEDIA_URL, path)
        except Exception as e:
            logger.error(f"Error getting file URL: {str(e)}")
            return ""

    def delete_file(self, path: str) -> bool:
        """
        Delete a file from storage.
        
        Args:
            path: Path to file to delete
        
        Returns:
            bool: True if deleted successfully
        """
        try:
            if self.storage.exists(path):
                self.storage.delete(path)
                logger.info(f"File deleted: {path}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting file: {str(e)}")
            return False

    def file_exists(self, path: str) -> bool:
        """
        Check if a file exists.
        
        Args:
            path: Path to file
        
        Returns:
            bool: True if file exists
        """
        try:
            return self.storage.exists(path)
        except Exception as e:
            logger.error(f"Error checking file existence: {str(e)}")
            return False

    def get_file_size(self, path: str) -> int:
        """
        Get file size in bytes.
        
        Args:
            path: Path to file
        
        Returns:
            int: File size in bytes
        """
        try:
            return self.storage.size(path)
        except Exception as e:
            logger.error(f"Error getting file size: {str(e)}")
            return 0
