import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional

import httpx

from ..core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class OllamaConfig:
    """
    Configuration dataclass for OllamaService.

    Attributes:
        model_name (str): Name of the Ollama model to use.
        base_url (str): Base URL for the Ollama API.
        api_url (str): Full API endpoint for generating completions.
        request_timeout (int): Timeout for API requests in seconds.
        max_retries (int): Maximum number of retry attempts for API calls.
        retry_delay (float): Delay between retry attempts in seconds.
    """
    model_name: str
    base_url: str
    api_url: str
    request_timeout: int
    max_retries: int
    retry_delay: float


class OllamaService:
    """
    Async service class for interacting with the Ollama API to process document data.

    Provides methods for extracting structured data from OCR text or images using an Ollama language model,
    with error correction and validation, and includes retry and availability checks.
    Supports multiple document types through canonical schema extraction.
    """

    def __init__(
            self,
            model_name: Optional[str] = None,
            base_url: Optional[str] = None,
            request_timeout: Optional[int] = None,
            max_retries: Optional[int] = None,
            retry_delay: Optional[float] = None,
    ):
        """
        Initialize the OllamaService with configuration and perform validation.

        Args:
            model_name (str): Name of the Ollama model (default from settings).
            base_url (str): Base URL for the Ollama API (default from settings).
            request_timeout (int): Timeout for API requests in seconds.
            max_retries (int): Maximum number of retry attempts for API calls.
            retry_delay (float): Delay between retry attempts in seconds.
        """
        # Get values from settings if not provided
        model_name = model_name or settings.OLLAMA_MODEL_NAME
        base_url = base_url or settings.OLLAMA_BASE_URL
        request_timeout = request_timeout if request_timeout is not None else settings.OLLAMA_REQUEST_TIMEOUT
        max_retries = max_retries if max_retries is not None else settings.OLLAMA_MAX_RETRIES
        retry_delay = retry_delay if retry_delay is not None else settings.OLLAMA_RETRY_DELAY

        # Validate required parameters
        if not model_name or not model_name.strip():
            raise ValueError("OLLAMA_MODEL_NAME must be provided and cannot be empty")
        if not base_url or not base_url.strip():
            raise ValueError("OLLAMA_BASE_URL must be provided and cannot be empty")

        self.config = OllamaConfig(
            model_name=model_name.strip(),
            base_url=base_url.rstrip('/'),  # Ensure no trailing slash
            api_url=f"{base_url.rstrip('/')}/api/generate",
            request_timeout=max(1, request_timeout),  # Ensure positive timeout
            max_retries=max(0, max_retries),  # Ensure non-negative retries
            retry_delay=max(0.1, retry_delay),  # Ensure minimum delay
        )

        logger.info(
            f"Initialized OllamaService with model: {self.config.model_name}, "
            f"base_url: {self.config.base_url}, "
            f"timeout: {self.config.request_timeout}s, "
            f"max_retries: {self.config.max_retries}, "
            f"retry_delay: {self.config.retry_delay}s"
        )

    async def process_ocr_with_image(self, image_path: str) -> str:
        """
        Performs OCR on an image using the Ollama vision model.

        Args:
            image_path (str): Path or data URL of the image to process.

        Returns:
            str: Raw OCR text extracted from the image.

        Raises:
            Exception: If the API call fails.
        """
        logger.info(f"Processing image for OCR extraction: {image_path}")
        start_time = time.time()

        prompt = """Extract all text from this image using OCR.
        Preserve the original layout, structure, and formatting as much as possible.
        Include all visible text, numbers, and characters.
        Do not interpret or modify the content - just extract the raw text exactly as it appears.
        """

        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                # Read image and convert to base64 if it's a file path
                import base64
                if not image_path.startswith('data:'):
                    with open(image_path, 'rb') as f:
                        image_data = base64.b64encode(f.read()).decode('utf-8')
                else:
                    # Extract base64 data from data URL
                    image_data = image_path.split(',')[1] if ',' in image_path else image_path

                payload = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False,
                    "options": {
                        "temperature": 0.0,  # Deterministic for OCR
                        "top_p": 0.9,
                        "num_predict": 16384,  # Large context for full documents
                        "num_ctx": 8192,
                    }
                }

                response = await client.post(
                    self.config.api_url,
                    json=payload,
                )
                logger.debug(f"OCR API response received in {time.time() - start_time:.2f}s")

                if response.status_code != 200:
                    response.raise_for_status()

                result = response.json()
                response_text = result.get("response", "")
                logger.info(f"OCR extracted {len(response_text)} characters in {time.time() - start_time:.2f}s")
                return response_text
        except Exception as e:
            logger.error(f"Ollama OCR API call failed: {str(e)}")
            raise

    async def process_transactions_with_image(self, prompt: str, image_url: str) -> str:
        """
        Extracts transaction/item data from a document image using the Ollama model.

        Args:
            prompt (str): The prompt to send to the Ollama model.
            image_url (str): URL or data URL of the image containing the document.

        Returns:
            str: Raw response string from the Ollama API containing extracted transactions/items in JSON format.

        Raises:
            Exception: If the API call fails.
        """
        logger.info("Processing image for transaction extraction")
        start_time = time.time()
        logger.debug(f"Sending API request, prompt length: {len(prompt)}")

        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                payload = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "images": [image_url],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.8,
                        "top_k": 20,
                        "num_predict": 12288,
                        "num_ctx": 8192,
                        "repeat_penalty": 1.05,
                        "stop": ["\n\n---", "```", "Note:", "Summary:"],
                        "tfs_z": 0.9,
                        "typical_p": 0.95,
                    }
                }

                response = await client.post(
                    self.config.api_url,
                    json=payload,
                )
                logger.debug(f"API response received in {time.time() - start_time:.2f}s")

                if response.status_code != 200:
                    response.raise_for_status()

                result = response.json()
                response_text = result.get("response", "")
                logger.debug(f"Response length: {len(response_text)} characters")
                return response_text
        except Exception as e:
            logger.error(f"Ollama API call with image failed: {str(e)}")
            raise

    async def process_with_image(self, prompt: str, image_url: str, num_predict: int = 8192) -> str:
        """
        Generic method to process an image with a custom prompt using the Ollama model.

        Args:
            prompt (str): The prompt to send to the Ollama model.
            image_url (str): URL or data URL of the image to process.
            num_predict (int): Maximum tokens to predict (default: 8192).

        Returns:
            str: Raw response string from the Ollama API containing the extraction results.

        Raises:
            Exception: If the API call fails.
        """
        logger.info("Processing image with custom prompt")
        start_time = time.time()
        logger.debug(f"Sending API request, prompt length: {len(prompt)}")

        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                payload = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "images": [image_url],
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.8,
                        "top_k": 20,
                        "num_predict": num_predict,
                        "num_ctx": 4096,
                        "repeat_penalty": 1.05,
                        "stop": ["\n\n---", "```", "Note:", "Summary:"],
                        "tfs_z": 0.9,
                        "typical_p": 0.95,
                    }
                }

                response = await client.post(
                    self.config.api_url,
                    json=payload,
                )
                logger.debug(f"API response received in {time.time() - start_time:.2f}s")

                if response.status_code != 200:
                    response.raise_for_status()

                result = response.json()
                response_text = result.get("response", "")
                logger.debug(f"Response length: {len(response_text)} characters")
                return response_text
        except Exception as e:
            logger.error(f"Ollama API call with image failed: {str(e)}")
            raise

    async def process_metadata_with_image(self, prompt: str, image_url: str) -> str:
        """
        Extracts metadata from a document image using the Ollama model.

        Args:
            prompt (str): The prompt to send to the Ollama model.
            image_url (str): URL or data URL of the image containing the document.

        Returns:
            str: Raw response string from the Ollama API containing extracted metadata in JSON format.

        Raises:
            Exception: If the API call fails.
        """
        logger.info("Processing image for metadata extraction")
        # Use the generic method with default settings for metadata
        return await self.process_with_image(prompt, image_url, num_predict=6144)

    async def check_llm_availability(self) -> bool:
        """
        Checks if the Ollama server is available by making a version request.

        Returns:
            bool: True if the server is available, False otherwise.
        """
        logger.debug("Checking Ollama server availability")
        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                response = await client.get(
                    f"{self.config.base_url}/api/version",
                )
                is_available = response.status_code == 200
                logger.info(
                    f"Ollama server {'available' if is_available else f'unavailable (status: {response.status_code})'}")
                return is_available
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to Ollama server: {str(e)}")
            return False

    async def is_service_ready(self) -> bool:
        """
        Comprehensive check to verify if the Ollama service is ready for use.

        Returns:
            bool: True if the service is properly configured and available, False otherwise.
        """
        try:
            # Check if configuration is valid
            if not self.config.model_name or not self.config.base_url:
                logger.error("Ollama service not properly configured")
                return False

            # Check if server is available
            if not await self.check_llm_availability():
                logger.error("Ollama server is not available")
                return False

            return True
        except Exception as e:
            logger.error(f"Error checking Ollama service readiness: {str(e)}")
            return False

    async def call_llm_api_with_retry(self, prompt: str) -> str:
        """
        Calls the Ollama API with a retry mechanism and exponential backoff.

        Args:
            prompt (str): The prompt to send to the API.

        Returns:
            str: API response string, or an empty string if all retries fail.
        """
        import asyncio

        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._call_llm_api(prompt)
            except httpx.RequestError as e:
                logger.warning(f"API call attempt {attempt + 1}/{self.config.max_retries + 1} failed: {str(e)}")
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.debug(f"Retrying after {delay:.2f}s")
                    await asyncio.sleep(delay)
        logger.error("All API retries failed")
        return ""

    async def _call_llm_api(self, prompt: str) -> str:
        """
        Makes a single API call to the Ollama service.

        Args:
            prompt (str): The prompt to send to the Ollama model.

        Returns:
            str: API response string.

        Raises:
            httpx.RequestError: If the API call fails.
        """
        start_time = time.time()
        logger.debug(f"Sending API request, prompt length: {len(prompt)}")

        try:
            async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
                payload = {
                    "model": self.config.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.8,
                        "top_k": 20,
                        "num_predict": 12288,
                        "num_ctx": 8192,
                        "repeat_penalty": 1.05,
                        "stop": ["\n\n---", "```", "Note:", "Summary:"],
                        "tfs_z": 0.9,
                        "typical_p": 0.95,
                    }
                }

                response = await client.post(
                    self.config.api_url,
                    json=payload,
                )
                logger.debug(f"API response received in {time.time() - start_time:.2f}s, status: {response.status_code}")

                if response.status_code != 200:
                    response.raise_for_status()

                result = response.json()
                response_text = result.get("response", "")
                logger.debug(f"Response length: {len(response_text)} characters")
                return response_text

        except Exception as e:
            logger.error(f"Ollama API call failed: {str(e)}")
            raise

    @staticmethod
    def parse_llm_response(response: str, response_type: str) -> Dict[str, Any]:
        """
        Parses the Ollama API response string into structured data.

        Args:
            response (str): Raw response string from Ollama.
            response_type (str): Expected response type ("metadata" or "transactions").

        Returns:
            dict: Structured dictionary with 'metadata' or 'transactions' key, or default empty structure on error.
        """
        default_returns = {
            "metadata": {"metadata": {}},
            "transactions": {"transactions": []}
        }
        if not response:
            logger.warning("Empty response from Ollama")
            return default_returns.get(response_type, {"data": {}})

        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start < 0 or json_end <= json_start:
            logger.warning("No valid JSON found in response")
            logger.info(f"Response: {response}")
            return default_returns[response_type]

        try:
            data = json.loads(response[json_start:json_end])
            if not isinstance(data, dict):
                logger.warning("Response is not a dictionary")
                return default_returns.get(response_type, {"data": {}})

            if response_type == "metadata":
                structured_data = {
                    "metadata": data.get("metadata", {}),
                }
            elif response_type == "transactions":
                structured_data = {
                    "transactions": data.get("transactions", []),
                }
            else:
                # Generic fallback for other response types
                structured_data = {"data": data}

            return structured_data
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}, response: {response}")
            return default_returns[response_type]
