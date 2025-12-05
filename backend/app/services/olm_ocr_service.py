import logging
import time
import base64
from typing import Optional
from pathlib import Path

import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)


class OlmOCRService:
    """
    Service for OCR extraction using OlmOCR.

    OlmOCR provides high-quality OCR extraction from images with
    layout preservation and table detection capabilities.
    """

    def __init__(
            self,
            base_url: Optional[str] = None,
            request_timeout: Optional[int] = None,
            max_retries: Optional[int] = None,
            retry_delay: Optional[float] = None,
    ):
        """
        Initialize OlmOCR service.

        Args:
            base_url: Base URL for OlmOCR API
            request_timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.base_url = (base_url or settings.OLMOCR_BASE_URL).rstrip('/')
        self.request_timeout = request_timeout if request_timeout is not None else settings.OLMOCR_REQUEST_TIMEOUT
        self.max_retries = max_retries if max_retries is not None else settings.OLMOCR_MAX_RETRIES
        self.retry_delay = retry_delay if retry_delay is not None else settings.OLMOCR_RETRY_DELAY

        logger.info(
            f"OlmOCRService initialized: base_url={self.base_url}, "
            f"timeout={self.request_timeout}s, max_retries={self.max_retries}"
        )

    async def extract_text_from_image(self, image_path: str) -> str:
        """
        Extract text from image using OlmOCR.

        Args:
            image_path: Path to image file

        Returns:
            str: Extracted raw OCR text

        Raises:
            Exception: If OCR extraction fails
        """
        logger.info(f"Starting OCR extraction for: {image_path}")
        start_time = time.time()

        try:
            # Read and encode image
            with open(image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            # Call OlmOCR API
            async with httpx.AsyncClient(timeout=self.request_timeout) as client:
                response = await client.post(
                    f"{self.base_url}/ocr",
                    json={
                        "image": image_data,
                        "preserve_layout": True,
                        "detect_tables": True,
                        "language": "en"
                    }
                )

                if response.status_code != 200:
                    response.raise_for_status()

                result = response.json()
                extracted_text = result.get("text", "")

                duration = time.time() - start_time
                logger.info(
                    f"OCR extraction completed in {duration:.2f}s, "
                    f"extracted {len(extracted_text)} characters"
                )

                return extracted_text

        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {str(e)}")
            raise

    async def extract_text_from_images(self, image_paths: list[str]) -> list[str]:
        """
        Extract text from multiple images.

        Args:
            image_paths: List of image file paths

        Returns:
            list[str]: List of extracted texts for each image
        """
        logger.info(f"Processing {len(image_paths)} images for OCR")

        results = []
        for idx, image_path in enumerate(image_paths, 1):
            logger.info(f"Processing image {idx}/{len(image_paths)}: {Path(image_path).name}")
            try:
                text = await self.extract_text_from_image(image_path)
                results.append(text)
            except Exception as e:
                logger.error(f"Failed to process image {idx}: {str(e)}")
                results.append("")  # Add empty string for failed extractions

        logger.info(f"OCR extraction completed for {len(results)} images")
        return results

    async def check_service_availability(self) -> bool:
        """
        Check if OlmOCR service is available.

        Returns:
            bool: True if service is available
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/health")
                is_available = response.status_code == 200
                logger.info(f"OlmOCR service {'available' if is_available else 'unavailable'}")
                return is_available
        except Exception as e:
            logger.error(f"Failed to check OlmOCR service: {str(e)}")
            return False
