import os
import logging
from typing import List, Dict
import fitz  # PyMuPDF
from PIL import Image
import io

logger = logging.getLogger(__name__)


class PDFProcessor:
    """
    Responsible for extracting pages from PDF as high-quality images
    """

    def __init__(self, config):
        self.dpi = config['PDF_DPI']
        self.output_format = config['IMAGE_FORMAT']
        self.max_pages = config['MAX_PAGES']
        self.output_folder = config['EXTRACTED_IMAGES_FOLDER']

        # Ensure output folder exists
        os.makedirs(self.output_folder, exist_ok=True)

    def extract_images(self, pdf_path: str, statement_id: int) -> List[Dict[str, any]]:
        """
        Extract all pages as images

        Args:
            pdf_path: Path to the PDF file
            statement_id: ID of the statement for folder organization

        Returns:
            List of dicts: [
                {"page_number": 1, "image_path": "/path/to/image_1.png"},
                {"page_number": 2, "image_path": "/path/to/image_2.png"},
                ...
            ]
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        # Create statement-specific folder
        statement_folder = os.path.join(self.output_folder, str(statement_id))
        os.makedirs(statement_folder, exist_ok=True)

        images = []

        try:
            # Open PDF
            pdf_document = fitz.open(pdf_path)
            page_count = len(pdf_document)

            if page_count > self.max_pages:
                raise ValueError(f"PDF has {page_count} pages, exceeding maximum of {self.max_pages}")

            logger.info(f"Extracting {page_count} pages from {pdf_path}")

            # Extract each page
            for page_num in range(page_count):
                page = pdf_document[page_num]

                # Calculate zoom factor from DPI
                # PyMuPDF default is 72 DPI, so we scale up
                zoom = self.dpi / 72.0
                mat = fitz.Matrix(zoom, zoom)

                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # Convert to PIL Image
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))

                # Save image
                image_filename = f"page_{page_num + 1}.{self.output_format.lower()}"
                image_path = os.path.join(statement_folder, image_filename)
                img.save(image_path, self.output_format)

                images.append({
                    "page_number": page_num + 1,
                    "image_path": image_path
                })

                logger.debug(f"Extracted page {page_num + 1}/{page_count} to {image_path}")

            pdf_document.close()
            logger.info(f"Successfully extracted {len(images)} pages")

            return images

        except Exception as e:
            logger.exception(f"Error extracting images from PDF: {e}")
            raise

    def get_page_count(self, pdf_path: str) -> int:
        """
        Get total page count without extracting

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Number of pages
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        try:
            pdf_document = fitz.open(pdf_path)
            page_count = len(pdf_document)
            pdf_document.close()
            return page_count
        except Exception as e:
            logger.exception(f"Error getting page count: {e}")
            raise

    def cleanup_images(self, statement_id: int) -> None:
        """
        Remove all extracted images for a statement

        Args:
            statement_id: ID of the statement
        """
        statement_folder = os.path.join(self.output_folder, str(statement_id))

        if os.path.exists(statement_folder):
            try:
                # Remove all files in the folder
                for filename in os.listdir(statement_folder):
                    file_path = os.path.join(statement_folder, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)

                # Remove the folder
                os.rmdir(statement_folder)
                logger.info(f"Cleaned up images for statement {statement_id}")
            except Exception as e:
                logger.exception(f"Error cleaning up images: {e}")
