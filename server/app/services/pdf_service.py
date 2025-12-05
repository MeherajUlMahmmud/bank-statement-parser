import gc
import logging
import mimetypes
import os
from contextlib import contextmanager
from typing import List, Optional

import fitz  # PyMuPDF

from base import settings

logger = logging.getLogger(__name__)


class PdfValidationError(Exception):
    """
    Custom exception for PDF validation errors.
    Raised when a PDF file fails validation checks.
    """
    pass


class PdfService:
    """
    Service class for validating, inspecting, and converting PDF files to images.

    Provides methods for comprehensive PDF validation, extracting file/page info,
    and converting PDF pages to images with memory/resource management.
    """

    def __init__(
            self,
            pdf_file_path: str,
            dpi: int = settings.PDF_DPI,
            max_file_size_mb: Optional[int] = None,
            max_pages: Optional[int] = None,
    ):
        """
        Initialize the PdfService and validate the PDF file.

        Args:
            pdf_file_path (str): Path to the PDF file.
            dpi (int): Resolution for exporting images (default: from settings).
            max_file_size_mb (Optional[int]): Maximum allowed file size in MB.
            max_pages (Optional[int]): Maximum allowed number of pages.

        Raises:
            PdfValidationError: If the PDF file fails validation.
        """
        logger.info(f"Initializing PdfProcessor with pdf_file_path='{pdf_file_path}', dpi={dpi}")
        self.pdf_file_path = pdf_file_path
        self.dpi = dpi
        self.max_file_size_mb = max_file_size_mb or getattr(settings, 'MAX_PDF_SIZE_MB', 100)
        self.max_pages = max_pages or getattr(settings, 'MAX_ALLOWED_PAGE_COUNT', 100)
        self.pdf_document = None

        # Validate the PDF file before proceeding
        self._validate_pdf_file()

    def __enter__(self):
        """
        Context manager entry point. Opens the PDF document.

        Returns:
            PdfService: The current instance with the PDF document opened.
        """
        self.pdf_document = self._open_pdf()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit point. Closes the PDF document and releases resources.
        """
        self.close()

    def close(self):
        """
        Explicitly close the PDF document and free resources.
        """
        if self.pdf_document:
            try:
                self.pdf_document.close()
                self.pdf_document = None
                gc.collect()  # Force garbage collection
            except Exception as e:
                logger.error(f"Error closing PDF document: {e}")

    def _validate_pdf_file(self):
        """
        Perform comprehensive PDF file validation including existence, type, size, and structure.

        Raises:
            PdfValidationError: If validation fails at any step.
        """
        logger.info(f"Starting validation for PDF file: {self.pdf_file_path}")

        # Check if file exists
        if not os.path.isfile(self.pdf_file_path):
            error_msg = f"PDF file does not exist: {self.pdf_file_path}"
            logger.error(error_msg)
            raise PdfValidationError(error_msg)

        # Check if file is readable
        if not os.access(self.pdf_file_path, os.R_OK):
            error_msg = f"PDF file is not readable: {self.pdf_file_path}"
            logger.error(error_msg)
            raise PdfValidationError(error_msg)

        # Validate file size
        self._validate_file_size()

        # Validate file type/extension
        self._validate_file_type()

        # Validate PDF structure and content
        self._validate_pdf_structure()

        logger.info(f"PDF file validation completed successfully: {self.pdf_file_path}")

    def _validate_file_size(self):
        """
        Validate the size of the PDF file.

        Raises:
            PdfValidationError: If file size is zero or exceeds the maximum allowed size.
        """
        try:
            file_size_bytes = os.path.getsize(self.pdf_file_path)
            file_size_mb = file_size_bytes / (1024 * 1024)

            logger.info(f"PDF file size: {file_size_mb:.2f} MB ({file_size_bytes} bytes)")

            # Check if file is empty
            if file_size_bytes == 0:
                error_msg = f"PDF file is empty: {self.pdf_file_path}"
                logger.error(error_msg)
                raise PdfValidationError(error_msg)

            # Check maximum file size
            if file_size_mb > self.max_file_size_mb:
                error_msg = (f"PDF file size ({file_size_mb:.2f} MB) exceeds maximum allowed size "
                             f"({self.max_file_size_mb} MB): {self.pdf_file_path}")
                logger.error(error_msg)
                raise PdfValidationError(error_msg)

            logger.info(f"File size validation passed: {file_size_mb:.2f} MB <= {self.max_file_size_mb} MB")

        except OSError as e:
            error_msg = f"Error getting file size for {self.pdf_file_path}: {e}"
            logger.error(error_msg)
            raise PdfValidationError(error_msg)

    def _validate_file_type(self):
        """
        Validate the file type using extension, MIME type, and PDF signature.

        Raises:
            PdfValidationError: If the file is not a valid PDF.
        """
        # Check file extension
        file_ext = os.path.splitext(self.pdf_file_path)[1].lower()
        if file_ext != '.pdf':
            error_msg = f"Invalid file extension '{file_ext}'. Expected '.pdf': {self.pdf_file_path}"
            logger.error(error_msg)
            raise PdfValidationError(error_msg)

        # Check MIME type
        mime_type, _ = mimetypes.guess_type(self.pdf_file_path)
        if mime_type and mime_type != 'application/pdf':
            error_msg = f"Invalid MIME type '{mime_type}'. Expected 'application/pdf': {self.pdf_file_path}"
            logger.error(error_msg)
            raise PdfValidationError(error_msg)

        # Check PDF magic bytes (file signature)
        try:
            with open(self.pdf_file_path, 'rb') as f:
                header = f.read(8)
                if not header.startswith(b'%PDF-'):
                    error_msg = f"Invalid PDF file signature. File may be corrupted: {self.pdf_file_path}"
                    logger.error(error_msg)
                    raise PdfValidationError(error_msg)

                # Extract PDF version
                pdf_version = header[5:8].decode('ascii', errors='ignore')
                logger.info(f"PDF version detected: {pdf_version}")

        except (IOError, UnicodeDecodeError) as e:
            error_msg = f"Error reading PDF file header: {e}"
            logger.error(error_msg)
            raise PdfValidationError(error_msg)

        logger.info("File type validation passed: Valid PDF file")

    def _validate_pdf_structure(self):
        """
        Validate the structure and content of the PDF using PyMuPDF.

        Raises:
            PdfValidationError: If the PDF is corrupted, empty, or has invalid structure.
        """
        temp_document = None
        try:
            # Try to open the PDF with PyMuPDF
            temp_document = fitz.open(self.pdf_file_path)

            # Check if PDF is encrypted/password protected
            if temp_document.needs_pass:
                error_msg = f"PDF file is password protected: {self.pdf_file_path}"
                logger.error(error_msg)
                raise PdfValidationError(error_msg)

            # Check page count
            page_count = len(temp_document)
            logger.info(f"PDF contains {page_count} pages")

            if page_count == 0:
                error_msg = f"PDF file contains no pages: {self.pdf_file_path}"
                logger.error(error_msg)
                raise PdfValidationError(error_msg)

            if page_count > self.max_pages:
                error_msg = (f"PDF file has too many pages ({page_count}). Maximum allowed: {self.max_pages}")
                logger.error(error_msg)
                raise PdfValidationError(error_msg)

            # Try to load the first page to ensure basic functionality
            try:
                first_page = temp_document.load_page(0)
                # Try to get page dimensions
                rect = first_page.rect
                logger.info(f"First page dimensions: {rect.width} x {rect.height}")

                # Check for reasonable page dimensions
                if rect.width <= 0 or rect.height <= 0:
                    error_msg = f"Invalid page dimensions in PDF: {self.pdf_file_path}"
                    logger.error(error_msg)
                    raise PdfValidationError(error_msg)

            except Exception as e:
                error_msg = f"Error loading first page of PDF: {e}"
                logger.error(error_msg)
                raise PdfValidationError(error_msg)

            # Check PDF metadata for additional validation
            metadata = temp_document.metadata
            if metadata:
                logger.info(f"PDF metadata - Title: {metadata.get('title', 'N/A')}, "
                            f"Author: {metadata.get('author', 'N/A')}, "
                            f"Creator: {metadata.get('creator', 'N/A')}")

            logger.info("PDF structure validation passed")

        except fitz.FileDataError as e:
            error_msg = f"Corrupted or invalid PDF file: {e}"
            logger.error(error_msg)
            raise PdfValidationError(error_msg)
        finally:
            # Clean up temporary document
            if temp_document:
                try:
                    temp_document.close()
                    temp_document = None
                    gc.collect()
                except Exception as e:
                    logger.warning(f"Error closing temporary PDF document: {e}")

    def _open_pdf(self):
        """
        Open the PDF file using PyMuPDF.

        Returns:
            fitz.Document: The opened PDF document.

        Raises:
            FileNotFoundError: If the file does not exist.
            fitz.FileDataError: If the file is invalid or corrupted.
            Exception: For other errors during opening.
        """
        logger.info(f"Attempting to open PDF file: {self.pdf_file_path}")
        try:
            pdf_document = fitz.open(self.pdf_file_path)
            logger.info(f"PDF file opened successfully: {self.pdf_file_path}")
            return pdf_document
        except FileNotFoundError:
            logger.error(f"PDF file not found: {self.pdf_file_path}")
            raise
        except fitz.FileDataError:
            logger.error(f"Invalid or corrupted PDF file: {self.pdf_file_path}")
            raise
        except Exception as e:
            logger.error(f"Error opening PDF file: {e}")
            raise

    def get_page_count(self) -> int:
        """
        Get the number of pages in the PDF.

        Returns:
            int: Number of pages in the PDF.
        """
        should_close = False
        if not self.pdf_document:
            self.pdf_document = self._open_pdf()
            should_close = True

        try:
            page_count = len(self.pdf_document)
            logger.info(f"Retrieved page count: {page_count} pages in '{self.pdf_file_path}'")
            return page_count
        finally:
            if should_close:
                self.close()

    def get_file_info(self) -> dict:
        """
        Get comprehensive information about the PDF file, including size, page count, metadata, and dimensions.

        Returns:
            dict: Dictionary containing file information and PDF metadata.
        """
        info = {
            'file_path': self.pdf_file_path,
            'file_size_bytes': os.path.getsize(self.pdf_file_path),
        }

        # Basic file info
        info['file_size_mb'] = info['file_size_bytes'] / (1024 * 1024)

        # PDF specific info
        should_close = False
        if not self.pdf_document:
            self.pdf_document = self._open_pdf()
            should_close = True

        try:
            info['page_count'] = len(self.pdf_document)
            info['metadata'] = self.pdf_document.metadata
            info['is_encrypted'] = self.pdf_document.needs_pass

            # Get first page dimensions
            if info['page_count'] > 0:
                first_page = self.pdf_document.load_page(0)
                rect = first_page.rect
                info['page_width'] = rect.width
                info['page_height'] = rect.height

        finally:
            if should_close:
                self.close()

        return info

    @contextmanager
    def _page_context(self, page_num: int):
        """
        Context manager for handling page resources.

        Args:
            page_num (int): The page number to load.

        Yields:
            fitz.Page: The loaded page object.
        """
        page = None
        try:
            if self.pdf_document is None:
                raise RuntimeError("PDF document is not open. Call _open_pdf() before using _page_context.")
            page = self.pdf_document.load_page(page_num)
            yield page
        finally:
            if page:
                # Clear page resources
                page = None
                gc.collect()

    @contextmanager
    def _pixmap_context(self, page, matrix):
        """
        Context manager for handling pixmap resources.

        Args:
            page (fitz.Page): The page to render.
            matrix (fitz.Matrix): The transformation matrix for rendering.

        Yields:
            fitz.Pixmap: The rendered pixmap object.
        """
        pix = None
        try:
            pix = page.get_pixmap(matrix=matrix)
            yield pix
        finally:
            if pix:
                # Clear pixmap resources
                pix = None
                gc.collect()

    def convert_to_images(self, output_path: str) -> List[str]:
        """
        Convert each page of the PDF to an image with the specified DPI.

        Args:
            output_path (str): Directory where the images will be saved.

        Returns:
            List[str]: List of paths to the saved images.
        """
        if not self.pdf_document:
            self.pdf_document = self._open_pdf()

        logger.info(f"Starting PDF to image conversion for '{self.pdf_file_path}' with output_path='{output_path}' "
                    f"and dpi={self.dpi}")
        os.makedirs(output_path, exist_ok=True)
        logger.info(f"Ensured output directory exists: '{output_path}'")

        image_paths = []
        page_count = self.get_page_count()
        logger.info(f"Converting {page_count} pages to images")

        batch_size: int = 5

        # Process pages in batches to manage memory
        for batch_start in range(0, page_count, batch_size):
            batch_end = min(batch_start + batch_size, page_count)

            # Close and reopen document for each batch to prevent memory accumulation
            if self.pdf_document:
                self.close()

            self.pdf_document = self._open_pdf()

            try:
                for page_num in range(batch_start, batch_end):
                    logger.info(f"Converting page {page_num + 1}/{page_count} to image")

                    with self._page_context(page_num) as page:
                        # Define the resolution (500 DPI)
                        zoom_x = self.dpi / 72
                        zoom_y = self.dpi / 72
                        matrix = fitz.Matrix(zoom_x, zoom_y)
                        logger.debug(f"Using matrix zoom_x={zoom_x}, zoom_y={zoom_y}")

                        # Render page to a Pixmap at 500 DPI
                        with self._pixmap_context(page, matrix) as pix:
                            image_path = os.path.join(output_path, f"Page_{page_num + 1}.jpg")
                            pix.save(image_path)
                            image_paths.append(image_path)
                            logger.info(f"Saved page {page_num + 1} as image: {image_path}")
            except Exception as e:
                logger.error(f"Error in batch processing pages {batch_start + 1} to {batch_end}: {e}")
                raise
            finally:
                # Ensure cleanup
                self.close()
                image_paths_batch = None  # Clear references
                gc.collect()

        logger.info("PDF to image conversion completed")
        return image_paths
