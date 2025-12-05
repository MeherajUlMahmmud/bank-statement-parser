"""
Dimension validation utility - DEPRECATED.
Page size validation has been completely removed.
This file is kept for backward compatibility but contains no validation logic.
"""
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """
    Get dimensions of an image file.
    No validation is performed.

    Args:
        image_path (str): Path to the image file.

    Returns:
        Tuple[int, int]: (width, height) in pixels.

    Raises:
        Exception: If the image cannot be read.
    """
    try:
        from PIL import Image
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
    No validation is performed.

    Args:
        pdf_path (str): Path to the PDF file.
        page_num (int): Page number (0-indexed). Default is 0 (first page).

    Returns:
        Tuple[float, float]: (width, height) in points.

    Raises:
        Exception: If the PDF cannot be read.
    """
    try:
        import fitz  # PyMuPDF
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


# Stub functions that do nothing (for backward compatibility)
def validate_image_dimensions(image_path: str, config=None) -> Tuple[int, int]:
    """No validation performed - returns dimensions only."""
    return get_image_dimensions(image_path)


def validate_pdf_dimensions(pdf_path: str, config=None) -> Tuple[float, float]:
    """No validation performed - returns dimensions only."""
    return get_pdf_page_dimensions(pdf_path, page_num=0)


# Additional stub functions for backward compatibility
def validate_a4_strict(image_path: str) -> Tuple[int, int]:
    """No validation performed."""
    return get_image_dimensions(image_path)


def validate_a4_flexible(image_path: str, tolerance_percent: float = 5.0) -> Tuple[int, int]:
    """No validation performed."""
    return get_image_dimensions(image_path)


def validate_pdf_a4_strict(pdf_path: str) -> Tuple[float, float]:
    """No validation performed."""
    return get_pdf_page_dimensions(pdf_path, page_num=0)


def validate_pdf_a4_flexible(pdf_path: str, tolerance_percent: float = 5.0) -> Tuple[float, float]:
    """No validation performed."""
    return get_pdf_page_dimensions(pdf_path, page_num=0)


def validate_a4_size_points(width_points: float, height_points: float) -> bool:
    """No validation performed - always returns True."""
    return True


def validate_a4_size_pixels(width_pixels: int, height_pixels: int, dpi: int = 300) -> bool:
    """No validation performed - always returns True."""
    return True


def validate_image_a4_dimensions(image_path: str) -> Tuple[int, int]:
    """No validation performed."""
    return get_image_dimensions(image_path)


def validate_pdf_a4_dimensions(pdf_path: str) -> Tuple[float, float]:
    """No validation performed."""
    return get_pdf_page_dimensions(pdf_path, page_num=0)
