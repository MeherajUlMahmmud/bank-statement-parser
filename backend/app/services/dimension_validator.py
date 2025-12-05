"""
Dimension validation utility - Simple implementation for FastAPI.
Provides basic dimension checking for images and PDFs.
"""
import logging
from typing import Tuple
from PIL import Image
import fitz  # PyMuPDF

logger = logging.getLogger(__name__)


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get dimensions of an image file.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Tuple[int, int]: (width, height) in pixels.

    Raises:
        Exception: If the image cannot be read.
    """
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            logger.info(f"Image dimensions: {width}x{height} pixels for {image_path}")
            return width, height
    except Exception as e:
        logger.error(f"Error reading image dimensions from {image_path}: {e}")
        raise


def get_pdf_page_dimensions(pdf_path: str, page_num: int = 0) -> Tuple[float, float]:
    """
    Get dimensions of a PDF page.

    Args:
        pdf_path (str): Path to the PDF file.
        page_num (int): Page number (0-indexed). Default is 0 (first page).

    Returns:
        Tuple[float, float]: (width, height) in points.

    Raises:
        Exception: If the PDF cannot be read.
    """
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            page_num = 0

        page = doc.load_page(page_num)
        rect = page.rect
        width = rect.width
        height = rect.height

        doc.close()
        logger.info(f"PDF page {page_num} dimensions: {width}x{height} points for {pdf_path}")
        return width, height
    except Exception as e:
        logger.error(f"Error reading PDF dimensions from {pdf_path}: {e}")
        raise
