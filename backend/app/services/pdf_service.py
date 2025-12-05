import logging
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Tuple
from PIL import Image

from ..core.config import settings

logger = logging.getLogger(__name__)


class PDFProcessingService:
    """
    Service for processing PDF files and converting them to images for OCR.

    Handles:
    - PDF to image conversion
    - Multi-page processing
    - Image optimization for OCR
    """

    def __init__(self, dpi: int = None, image_format: str = None):
        """
        Initialize PDF Processing Service.

        Args:
            dpi: DPI for image conversion (default from settings)
            image_format: Output image format (PNG, JPEG, etc.)
        """
        self.dpi = dpi or settings.PDF_DPI
        self.image_format = image_format or settings.PDF_IMAGE_FORMAT
        logger.info(f"PDFProcessingService initialized: DPI={self.dpi}, format={self.image_format}")

    def pdf_to_images(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """
        Convert PDF pages to images.

        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save images (default: temp directory)

        Returns:
            List[str]: Paths to generated images
        """
        logger.info(f"Converting PDF to images: {pdf_path}")

        try:
            # Open PDF
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            logger.info(f"PDF has {page_count} pages")

            # Setup output directory
            if output_dir is None:
                output_dir = Path(pdf_path).parent / "temp_images"
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Convert each page
            image_paths = []
            for page_num in range(page_count):
                page = doc.load_page(page_num)

                # Render page to image
                mat = fitz.Matrix(self.dpi / 72, self.dpi / 72)
                pix = page.get_pixmap(matrix=mat)

                # Save image
                image_filename = f"page_{page_num + 1:03d}.{self.image_format.lower()}"
                image_path = output_path / image_filename

                if self.image_format.upper() == "PNG":
                    pix.save(str(image_path))
                else:
                    # Convert to PIL Image for other formats
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    img.save(str(image_path), self.image_format.upper())

                image_paths.append(str(image_path))
                logger.debug(f"Converted page {page_num + 1}/{page_count}: {image_path}")

            doc.close()
            logger.info(f"Successfully converted {len(image_paths)} pages to images")
            return image_paths

        except Exception as e:
            logger.error(f"Error converting PDF to images: {str(e)}")
            raise

    def get_pdf_metadata(self, pdf_path: str) -> dict:
        """
        Extract metadata from PDF.

        Args:
            pdf_path: Path to PDF file

        Returns:
            dict: PDF metadata
        """
        try:
            doc = fitz.open(pdf_path)
            metadata = {
                'page_count': len(doc),
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'keywords': doc.metadata.get('keywords', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'mod_date': doc.metadata.get('modDate', ''),
            }
            doc.close()
            return metadata
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
            return {}

    def cleanup_temp_images(self, image_paths: List[str]):
        """
        Clean up temporary image files.

        Args:
            image_paths: List of image paths to delete
        """
        if not settings.CLEANUP_TEMP_FILES:
            logger.info("Temp file cleanup disabled, skipping")
            return

        logger.info(f"Cleaning up {len(image_paths)} temporary images")
        for image_path in image_paths:
            try:
                Path(image_path).unlink()
            except Exception as e:
                logger.warning(f"Failed to delete {image_path}: {str(e)}")

        # Try to remove parent directory if empty
        try:
            parent_dir = Path(image_paths[0]).parent
            if parent_dir.name == "temp_images" and not list(parent_dir.iterdir()):
                parent_dir.rmdir()
                logger.debug(f"Removed empty temp directory: {parent_dir}")
        except Exception:
            pass
