import json
import logging
import time
from dataclasses import dataclass
from typing import Dict, Any

from groq import Groq

from base import settings
from .prompt_service import PromptService

logger = logging.getLogger(__name__)


@dataclass
class GroqConfig:
    """
    Configuration dataclass for GroqService.

    Attributes:
        model_name (str): Name of the Groq model to use.
        api_key (str): API key for authenticating with the Groq service.
        request_timeout (int): Timeout for API requests in seconds.
        max_retries (int): Maximum number of retry attempts for API calls.
        retry_delay (float): Delay between retry attempts in seconds.
    """
    model_name: str
    api_key: str
    request_timeout: int
    max_retries: int
    retry_delay: float


class GroqService:
    """
    Service class for interacting with the Groq API to process document data.

    Provides methods for extracting structured data from OCR text or images
    using a Groq language model, with error correction and validation.
    Supports multiple document types through canonical schema extraction.
    """

    def __init__(
            self,
            model_name: str = str(settings.GROQ_MODEL_NAME),
            api_key: str = str(settings.GROQ_API_KEY),
            request_timeout: int = settings.GROQ_REQUEST_TIMEOUT,
            max_retries: int = settings.GROQ_MAX_RETRIES,
            retry_delay: float = settings.GROQ_RETRY_DELAY,
    ):
        """
        Initialize the GroqService with configuration and perform validation.

        Args:
            model_name (str): Name of the Groq model (default: meta-llama/llama-4-scout-17b-16e-instruct).
            api_key (str): API key for Groq service.
            request_timeout (int): Timeout for API requests in seconds.
            max_retries (int): Maximum number of retry attempts for API calls.
            retry_delay (float): Delay between retry attempts in seconds.

        Raises:
            ValueError: If the API key is not provided.
        """
        self.config = GroqConfig(
            model_name=model_name,
            api_key=api_key,
            request_timeout=max(1, request_timeout),  # Ensure positive timeout
            max_retries=max(0, max_retries),  # Ensure non-negative retries
            retry_delay=max(0.1, retry_delay),  # Ensure minimum delay
        )

        # Initialize Groq client
        if not self.config.api_key:
            logger.error("Groq API key is required but not provided")
            raise ValueError("Groq API key is required")

        self.client = Groq(api_key=self.config.api_key)

        # Initialize prompt service
        self.prompt_service = PromptService()

        logger.info(
            f"Initialized GroqService with model: {self.config.model_name}, "
            f"timeout: {self.config.request_timeout}s, "
            f"max_retries: {self.config.max_retries}, "
            f"retry_delay: {self.config.retry_delay}s"
        )

    def process_transactions_with_image(self, prompt: str, image_url: str) -> str:
        """
        Extracts transaction/item data from a document image using the Groq model.

        Args:
            prompt (str): The prompt to send to the Groq model.
            image_url (str): URL or data URL of the image containing the document.

        Returns:
            str: Raw response string from the Groq API containing extracted transactions/items in JSON format.

        Raises:
            Exception: If the API call fails.
        """
        logger.info("Processing image for transaction extraction")
        start_time = time.time()
        logger.debug(f"Sending API request, prompt length: {len(prompt)}")

        try:
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt.strip()},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=8192,
                top_p=0.8,
                stream=False,
                response_format={"type": "json_object"},
                stop=["\n\n---", "```", "Note:", "Summary:"],
            )

            logger.debug(f"API response received in {time.time() - start_time:.2f}s")

            response_text = completion.choices[0].message.content or ""
            logger.debug(f"Response length: {len(response_text)} characters")
            return response_text
        except Exception as e:
            logger.error(f"Groq API call with image failed: {str(e)}")
            raise

    def process_with_image(self, prompt: str, image_url: str, max_tokens: int = 8192) -> str:
        """
        Generic method to process an image with a custom prompt using the Groq model.

        Args:
            prompt (str): The prompt to send to the Groq model.
            image_url (str): URL or data URL of the image to process.
            max_tokens (int): Maximum tokens for the response (default: 8192).

        Returns:
            str: Raw response string from the Groq API containing the extraction results.

        Raises:
            Exception: If the API call fails.
        """
        logger.info("Processing image with custom prompt")
        start_time = time.time()
        logger.debug(f"Sending API request, prompt length: {len(prompt)}")

        try:
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt.strip()},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                temperature=0.1,
                max_tokens=max_tokens,
                top_p=0.8,
                stream=False,
                response_format={"type": "json_object"},
                stop=["\n\n---", "```", "Note:", "Summary:"],
            )

            logger.debug(f"API response received in {time.time() - start_time:.2f}s")

            response_text = completion.choices[0].message.content or ""
            logger.debug(f"Response length: {len(response_text)} characters")
            return response_text
        except Exception as e:
            logger.error(f"Groq API call with image failed: {str(e)}")
            raise

    def process_metadata_with_image(self, prompt: str, image_url: str) -> str:
        """
        Extracts metadata from a document image using the Groq model.

        Args:
            prompt (str): The prompt to send to the Groq model.
            image_url (str): URL or data URL of the image containing the document.

        Returns:
            str: Raw response string from the Groq API containing extracted metadata in JSON format.

        Raises:
            Exception: If the API call fails.
        """
        logger.info("Processing image for metadata extraction")
        # Use the generic method with default settings for metadata
        return self.process_with_image(prompt, image_url, max_tokens=4096)

    def check_llm_availability(self) -> bool:
        """
        Checks if the Groq API is available by making a simple test request.

        Returns:
            bool: True if the API is available, False otherwise.
        """
        logger.debug("Checking Groq API availability")
        try:
            # Try a simple completion to test API connectivity
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0,
            )
            is_available = response is not None
            logger.info(f"Groq API {'available' if is_available else 'unavailable'}")
            return is_available

        except Exception as e:
            logger.error(f"Failed to connect to Groq API: {str(e)}")
            return False

    def is_service_ready(self) -> bool:
        """
        Comprehensive check to verify if the Groq service is ready for use.
        
        Returns:
            bool: True if the service is properly configured and available, False otherwise.
        """
        try:
            # Check if configuration is valid
            if not self.config.model_name or not self.config.api_key:
                logger.error("Groq service not properly configured")
                return False

            # Check if API is available
            if not self.check_llm_availability():
                logger.error("Groq API is not available")
                return False

            return True
        except Exception as e:
            logger.error(f"Error checking Groq service readiness: {str(e)}")
            return False

    def call_llm_api_with_retry(self, prompt: str) -> str:
        """
        Calls the Groq API with a retry mechanism and exponential backoff.

        Args:
            prompt (str): The prompt to send to the API.

        Returns:
            str: API response string, or an empty string if all retries fail.
        """
        for attempt in range(self.config.max_retries + 1):
            try:
                return self._call_llm_api(prompt)
            except Exception as e:
                logger.warning(f"API call attempt {attempt + 1}/{self.config.max_retries + 1} failed: {str(e)}")
                if attempt < self.config.max_retries:
                    delay = self.config.retry_delay * (2 ** attempt)
                    logger.debug(f"Retrying after {delay:.2f}s")
                    time.sleep(delay)
        logger.error("All API retries failed")
        return ""

    def _call_llm_api(self, prompt: str) -> str:
        """
        Makes a single API call to the Groq service.

        Args:
            prompt (str): The prompt to send to the Groq model.

        Returns:
            str: API response string.

        Raises:
            Exception: If the API call fails.
        """
        start_time = time.time()
        logger.debug(f"Sending API request, prompt length: {len(prompt)}")

        try:
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Balanced for JSON consistency
                max_tokens=8192,
                top_p=0.8,
                stream=False,
                response_format={"type": "json_object"},
                stop=["\n\n---", "```", "Note:", "Summary:"],
            )

            logger.debug(f"API response received in {time.time() - start_time:.2f}s")

            response_text = completion.choices[0].message.content or ""
            logger.debug(f"Response length: {len(response_text)} characters")
            return response_text

        except Exception as e:
            logger.error(f"Groq API call failed: {str(e)}")
            raise

    @staticmethod
    def parse_llm_response(response: str, response_type: str) -> Dict[str, Any]:
        """
        Parses the Groq API response string into structured data.

        Args:
            response (str): Raw response string from Groq.
            response_type (str): Expected response type ("metadata" or "transactions").

        Returns:
            dict: Structured dictionary with 'metadata' or 'transactions' key, or default empty structure on error.
        """
        default_returns = {
            "metadata": {"metadata": {}},
            "transactions": {"transactions": []}
        }
        if not response:
            logger.warning("Empty response from Groq")
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
