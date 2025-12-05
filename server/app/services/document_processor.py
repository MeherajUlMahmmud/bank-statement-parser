import logging
import os
import traceback
from typing import List, Dict, Any, Optional
from datetime import datetime

import cv2
import numpy as np
from django.db import transaction as db_transaction
from django.utils import timezone

from document_control.helper.helper import Helper
from document_control.services.groq_service import GroqService
from document_control.services.ollama_service import OllamaService
from document_control.services.pdf_service import PdfService
from document_control.services.data_processor import DataProcessor
from document_control.models import (
    DocumentModel, ExtractionModel, ExtractionMetadataModel,
    FieldExtractionModel, ValidationResultModel
)
from base import settings
from common.constants.error_messages import ErrorMessage

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Generic document processing service that handles any document type (bank statements,
    invoices, receipts, etc.). Uses canonical JSON schema for extraction and storage.
    """

    def __init__(self, document: DocumentModel):
        """
        Initialize the DocumentProcessor for a given document.

        Args:
            document (DocumentModel): The document object containing file and metadata information.
        """
        self.document = document
        self.is_success = False

        # Setup working directories
        parent_folder = os.path.dirname(self.document.file.file_path.path)
        self.image_save_path = os.path.join(parent_folder, 'extracted_images')
        self.processed_image_save_path = os.path.join(parent_folder, 'processed_images')

        # Create directories if they do not exist
        os.makedirs(self.image_save_path, exist_ok=True)
        os.makedirs(self.processed_image_save_path, exist_ok=True)

        # Initialize LLM service
        self.llm_service = self._initialize_llm_service()

    def _initialize_llm_service(self):
        """Initialize LLM service (Ollama or Groq)."""
        use_ollama = getattr(settings, 'USE_OLLAMA', False)

        try:
            if use_ollama:
                llm_service = OllamaService()
                if hasattr(llm_service, 'is_service_ready') and not llm_service.is_service_ready():
                    logger.warning("Ollama service not ready, falling back to GroqService")
                    llm_service = GroqService()
            else:
                llm_service = GroqService()

            logger.info(f"Using LLM service: {type(llm_service).__name__}")
            return llm_service

        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            logger.info("Falling back to GroqService")
            return GroqService()

    def process_document(self) -> bool:
        """
        Main processing method that handles document extraction using canonical schema.

        Returns:
            bool: True if processing is successful, False otherwise.
        """
        try:
            # Update document status
            self.document.process_status = 'processing'
            self.document.processing_started_at = timezone.now()
            self.document.save(update_fields=['process_status', 'processing_started_at'])

            # Convert PDF to images or prepare image file
            image_paths = self._prepare_images()
            if not image_paths:
                return self._handle_failure("Failed to prepare images")

            # Load images
            loaded_images = self._load_images(image_paths)
            if not loaded_images:
                return self._handle_failure("Failed to load images")

            # Update page count
            self.document.page_count = len(loaded_images)
            self.document.save(update_fields=['page_count'])

            # Process each page and extract data using canonical schema
            canonical_data = self._extract_canonical_data(loaded_images)
            if not canonical_data:
                return self._handle_failure("Failed to extract data from document")

            # Process and clean extracted data (DataProcessor)
            canonical_data = self._process_extracted_data(canonical_data)

            # Save extraction results
            self._save_extraction_results(canonical_data)

            # Mark as success
            self._mark_success()
            return True

        except Exception as e:
            error_info = traceback.format_exc()
            logger.error("Caught an exception during processing:")
            logger.error(error_info)
            return self._handle_failure(str(e))

    def _prepare_images(self) -> List[str]:
        """
        Prepare images from uploaded file (PDF or image).
        For PDFs: converts to images.
        For images: copies to the image save path.

        Returns:
            List[str]: List of image paths.
        """
        file_path = self.document.file.file_path.path
        file_ext = os.path.splitext(file_path)[1].lower()
        is_pdf = file_ext == '.pdf'
        is_image = file_ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']

        if is_pdf:
            return self._convert_pdf_to_images()
        elif is_image:
            return self._prepare_image_file()
        else:
            logger.error(f"Unsupported file type: {file_ext}")
            return []

    def _convert_pdf_to_images(self) -> List[str]:
        """
        Convert PDF to images.

        Returns:
            List[str]: List of image paths.
        """
        try:
            with PdfService(self.document.file.file_path.path, dpi=getattr(settings, 'PDF_DPI', 200)) as pdf_processor:
                page_count = pdf_processor.get_page_count()
                logger.info(f"PDF contains {page_count} pages")
                image_paths = pdf_processor.convert_to_images(self.image_save_path)
                return image_paths
        except Exception as e:
            logger.error(ErrorMessage.PDF_TO_IMAGE_CONVERSION_ERROR_MESSAGE)
            logger.error(f"Error occurred: {str(e)}")
            return []

    def _prepare_image_file(self) -> List[str]:
        """
        Copy uploaded image file to the image save path.

        Returns:
            List[str]: List containing the single image path.
        """
        try:
            import shutil
            file_path = self.document.file.file_path.path
            file_name = os.path.basename(file_path)

            # Copy image to the image save path
            destination_path = os.path.join(self.image_save_path, f"Page_1_{file_name}")
            shutil.copy2(file_path, destination_path)

            logger.info(f"Image file copied to: {destination_path}")
            return [destination_path]
        except Exception as e:
            logger.error(f"Error preparing image file: {str(e)}")
            return []

    def _load_images(self, image_paths: List[str]) -> List[np.ndarray]:
        """
        Load images from paths.

        Args:
            image_paths (List[str]): List of image paths.

        Returns:
            List[np.ndarray]: List of loaded images.
        """
        loaded_images = []
        for image_path in image_paths:
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    loaded_images.append(img)
                else:
                    logger.warning(f"Failed to load image: {image_path}")
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {e}")
        return loaded_images

    def _extract_canonical_data(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Extract data from document images using canonical JSON schema.

        Args:
            images (List[np.ndarray]): List of document page images.

        Returns:
            Dict[str, Any]: Canonical JSON data structure.
        """
        try:
            # Initialize canonical data structure
            canonical_data = {
                "document_id": str(self.document.id),
                "source_filename": self.document.file.file_name,
                "document_type": self.document.document_type,
                "pages": [],
                "extractions": {}
            }

            # Use full-page extraction for all documents
            canonical_data = self._extract_full_pages(images, canonical_data)

            return canonical_data

        except Exception as e:
            logger.error(f"Error during canonical data extraction: {e}")
            logger.error(traceback.format_exc())
            return {}

    def _extract_raw_text_from_image(self, image: np.ndarray) -> str:
        """
        Extract raw text from an image using the LLM service.
        
        Args:
            image (np.ndarray): Image array to extract text from.
            
        Returns:
            str: Extracted raw text from the image.
        """
        try:
            image_data_url = Helper.image_to_data_url(image)
            prompt = "Extract all the text from this document image. Return only the raw text content, preserving line breaks and spacing. Do not format or structure the text, just return it as-is."
            
            # Handle different service parameter names
            if hasattr(self.llm_service, 'process_with_image'):
                # Check if it's Ollama (uses num_predict) or Groq (uses max_tokens)
                import inspect
                sig = inspect.signature(self.llm_service.process_with_image)
                params = list(sig.parameters.keys())
                
                if 'num_predict' in params:
                    # Ollama service
                    raw_text = self.llm_service.process_with_image(
                        prompt=prompt,
                        image_url=image_data_url,
                        num_predict=4096
                    )
                else:
                    # Groq service
                    raw_text = self.llm_service.process_with_image(
                        prompt=prompt,
                        image_url=image_data_url,
                        max_tokens=4096
                    )
            else:
                logger.warning("LLM service does not support process_with_image, skipping raw text extraction")
                return ""
            
            # Clean up the response (remove markdown formatting if present)
            if raw_text:
                # Remove markdown code blocks if present
                raw_text = raw_text.strip()
                if raw_text.startswith('```'):
                    lines = raw_text.split('\n')
                    raw_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else raw_text
                raw_text = raw_text.strip()
            
            logger.info(f"Extracted {len(raw_text)} characters of raw text")
            return raw_text
        except Exception as e:
            logger.error(f"Error extracting raw text: {str(e)}")
            logger.error(traceback.format_exc())
            return ""

    def _extract_full_pages(
            self,
            images: List[np.ndarray],
            canonical_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Extract data from full document pages (for invoices, receipts, generic docs).

        Args:
            images (List[np.ndarray]): List of document images.
            canonical_data (Dict[str, Any]): Canonical data structure to populate.

        Returns:
            Dict[str, Any]: Populated canonical data.
        """
        logger.info("Using full-page extraction")

        # For multi-page documents, we can process each page or combine them
        # For simplicity, we'll process all pages together

        # Convert first page (or all pages) to data URL
        if len(images) == 1:
            # Single page - process directly
            image_data_url = Helper.image_to_data_url(images[0])
            extraction = self._extract_with_canonical_prompt(image_data_url)

            if extraction and 'extractions' in extraction:
                canonical_data['extractions'] = extraction['extractions']

            # Extract raw text from the page
            raw_text = self._extract_raw_text_from_image(images[0])
            
            canonical_data['pages'].append({
                "page_number": 1,
                "raw_text": raw_text
            })
        else:
            # Multi-page - process each page and merge results
            for page_idx, image in enumerate(images):
                page_number = page_idx + 1
                logger.info(f"Processing page {page_number}/{len(images)}")

                image_data_url = Helper.image_to_data_url(image)
                extraction = self._extract_with_canonical_prompt(image_data_url)

                if extraction and 'extractions' in extraction:
                    # Merge extractions (simple merge - can be improved)
                    self._merge_extractions(canonical_data['extractions'], extraction['extractions'])

                # Extract raw text from the page
                raw_text = self._extract_raw_text_from_image(image)

                canonical_data['pages'].append({
                    "page_number": page_number,
                    "raw_text": raw_text
                })

        return canonical_data

    def _extract_with_canonical_prompt(self, image_data_url: str) -> Dict[str, Any]:
        """
        Extract data using canonical schema prompt.

        Args:
            image_data_url (str): Base64 encoded image data URL.

        Returns:
            Dict[str, Any]: Extracted data in canonical format.
        """
        try:
            # Get canonical extraction prompt based on document type
            prompt = self.llm_service.prompt_service.create_canonical_extraction_prompt(
                document_type=self.document.document_type,
                include_confidence=True,
                include_bbox=True
            )

            # Send to LLM for extraction
            if hasattr(self.llm_service, 'process_with_image'):
                response = self.llm_service.process_with_image(prompt, image_data_url)
            else:
                # Fallback for services that don't have generic process_with_image
                response = self.llm_service.process_metadata_with_image(prompt, image_data_url)

            # Parse response
            if isinstance(response, str):
                import json
                extraction = json.loads(response)
            else:
                extraction = response

            return extraction

        except Exception as e:
            logger.error(f"Error extracting with canonical prompt: {e}")
            logger.error(traceback.format_exc())
            return {}

    def _merge_extractions(self, target: Dict[str, Any], source: Dict[str, Any]):
        """
        Merge source extractions into target extractions.

        Args:
            target (Dict[str, Any]): Target extraction dictionary.
            source (Dict[str, Any]): Source extraction dictionary.
        """
        for key, value in source.items():
            if key not in target:
                target[key] = value
            elif isinstance(value, list) and isinstance(target[key], list):
                target[key].extend(value)
            elif isinstance(value, dict) and isinstance(target[key], dict):
                target[key].update(value)

    def _process_extracted_data(self, canonical_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and clean extracted data using DataProcessor.

        Args:
            canonical_data (Dict[str, Any]): Raw canonical data from extraction.

        Returns:
            Dict[str, Any]: Processed and cleaned canonical data.
        """
        try:
            # Check if data processing is enabled
            use_data_processing = getattr(settings, 'USE_DATA_PROCESSING', True)

            if not use_data_processing:
                logger.info("Data processing is disabled, skipping DataProcessor")
                return canonical_data

            logger.info("Processing extracted data with DataProcessor")

            # Initialize DataProcessor
            data_processor = DataProcessor(
                auto_correct=getattr(settings, 'DATA_PROCESSOR_AUTO_CORRECT', True),
                remove_duplicates=getattr(settings, 'DATA_PROCESSOR_REMOVE_DUPLICATES', True)
            )

            # Process data
            processed_data = data_processor.resolve_issues(canonical_data)

            # Log processing results
            validation = processed_data.get('validation', {})
            logger.info(
                f"Data processing completed. "
                f"Issues found: {len(validation.get('issues_found', []))}, "
                f"Resolved: {len(validation.get('issues_resolved', []))}, "
                f"Valid: {validation.get('is_valid', True)}"
            )

            return processed_data

        except Exception as e:
            logger.error(f"Error during data processing: {e}", exc_info=True)
            # Return original data if processing fails
            logger.warning("Returning original data due to processing error")
            return canonical_data

    def _save_extraction_results(self, canonical_data: Dict[str, Any]):
        """
        Save extraction results to the database using generic models.

        Args:
            canonical_data (Dict[str, Any]): Canonical JSON data to save.
        """
        try:
            with db_transaction.atomic():
                # Create or update ExtractionModel
                extraction, created = ExtractionModel.objects.update_or_create(
                    document=self.document,
                    defaults={
                        'canonical_data': canonical_data,
                        'raw_extraction': canonical_data  # Can store different raw format if needed
                    }
                )

                logger.info(f"{'Created' if created else 'Updated'} ExtractionModel for document {self.document.id}")

                # Create or update ExtractionMetadataModel
                ExtractionMetadataModel.objects.update_or_create(
                    document=self.document,
                    defaults={
                        'vlm_model': type(self.llm_service).__name__,
                        'prompt_version': '1.0.0',
                        'prompts_used': ['canonical_extraction'],
                        'extraction_version': '2.0.0',  # New generic version
                        'processing_config': {
                            'dpi': getattr(settings, 'PDF_DPI', 200),
                            'document_type': self.document.document_type
                        }
                    }
                )

                # Create FieldExtractionModel records for each field
                self._create_field_extractions(extraction, canonical_data)

                # Create ValidationResultModel
                self._create_validation_result(extraction, canonical_data)

        except Exception as e:
            logger.error(f"Error saving extraction results: {e}")
            logger.error(traceback.format_exc())
            raise

    def _create_field_extractions(self, extraction: ExtractionModel, canonical_data: Dict[str, Any]):
        """
        Create FieldExtractionModel records for each extracted field.

        Args:
            extraction (ExtractionModel): The extraction model instance.
            canonical_data (Dict[str, Any]): Canonical JSON data.
        """
        try:
            # Delete existing field extractions
            FieldExtractionModel.objects.filter(extraction=extraction).delete()

            field_extractions = []
            extractions = canonical_data.get('extractions', {})

            # Recursively extract fields
            self._extract_fields_recursive(extractions, '', field_extractions, extraction)

            # Bulk create field extractions
            if field_extractions:
                FieldExtractionModel.objects.bulk_create(field_extractions, ignore_conflicts=True)
                logger.info(f"Created {len(field_extractions)} field extraction records")

        except Exception as e:
            logger.error(f"Error creating field extractions: {e}")

    def _extract_fields_recursive(
            self,
            data: Any,
            path: str,
            field_list: List[FieldExtractionModel],
            extraction: ExtractionModel
    ):
        """
        Recursively extract fields from canonical data structure.

        Args:
            data (Any): Current data node.
            path (str): Current field path.
            field_list (List[FieldExtractionModel]): List to append field extraction models to.
            extraction (ExtractionModel): Parent extraction model.
        """
        if isinstance(data, dict):
            # Check if this is a field with value/confidence structure
            if 'value' in data:
                field_name = path.split('.')[-1] if path else 'unknown'
                field_extraction = FieldExtractionModel(
                    extraction=extraction,
                    field_path=path,
                    field_name=field_name,
                    value=str(data.get('value', '')),
                    confidence=float(data.get('confidence', 0.0)),
                    page_number=data.get('page'),
                    bbox=data.get('bbox', []),
                    raw_text=str(data.get('value', '')),
                    extraction_method='vlm',
                    needs_review=float(data.get('confidence', 0.0)) < 0.7
                )
                field_list.append(field_extraction)
            else:
                # Recurse into nested dictionaries
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    self._extract_fields_recursive(value, new_path, field_list, extraction)

        elif isinstance(data, list):
            # Handle arrays (transactions, line_items, etc.)
            for idx, item in enumerate(data):
                new_path = f"{path}[{idx}]"
                self._extract_fields_recursive(item, new_path, field_list, extraction)

    def _create_validation_result(self, extraction: ExtractionModel, canonical_data: Dict[str, Any]):
        """
        Create validation result for the extraction.

        Args:
            extraction (ExtractionModel): The extraction model instance.
            canonical_data (Dict[str, Any]): Canonical JSON data.
        """
        try:
            # Calculate overall confidence
            field_confidences = []
            self._collect_confidences(canonical_data.get('extractions', {}), field_confidences)

            confidence_overall = sum(field_confidences) / len(field_confidences) if field_confidences else 0.0

            # Check for low confidence fields
            fields_below_threshold = []
            self._collect_low_confidence_fields(canonical_data.get('extractions', {}), '', fields_below_threshold, 0.7)

            # Create validation result
            ValidationResultModel.objects.update_or_create(
                extraction=extraction,
                defaults={
                    'passed': confidence_overall >= 0.7 and len(fields_below_threshold) == 0,
                    'confidence_overall': confidence_overall,
                    'validation_errors': [],
                    'validation_warnings': [f"Low confidence field: {f}" for f in fields_below_threshold],
                    'fields_below_threshold': fields_below_threshold,
                    'schema_compliant': True  # Assume schema compliance for now
                }
            )

        except Exception as e:
            logger.error(f"Error creating validation result: {e}")

    def _collect_confidences(self, data: Any, confidences: List[float]):
        """Recursively collect confidence scores from canonical data."""
        if isinstance(data, dict):
            if 'confidence' in data:
                confidences.append(float(data['confidence']))
            for value in data.values():
                self._collect_confidences(value, confidences)
        elif isinstance(data, list):
            for item in data:
                self._collect_confidences(item, confidences)

    def _collect_low_confidence_fields(
            self,
            data: Any,
            path: str,
            low_fields: List[str],
            threshold: float
    ):
        """Recursively collect field paths with confidence below threshold."""
        if isinstance(data, dict):
            if 'value' in data and 'confidence' in data:
                if float(data['confidence']) < threshold:
                    low_fields.append(path)
            else:
                for key, value in data.items():
                    new_path = f"{path}.{key}" if path else key
                    self._collect_low_confidence_fields(value, new_path, low_fields, threshold)
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                new_path = f"{path}[{idx}]"
                self._collect_low_confidence_fields(item, new_path, low_fields, threshold)

    def _mark_success(self):
        """Mark processing as successfully completed."""
        self.document.process_status = "completed"
        self.document.processing_completed_at = timezone.now()
        self.document.save(update_fields=['process_status', 'processing_completed_at'])
        self.is_success = True
        logger.info(f"Document {self.document.id} processing completed successfully")

    def _handle_failure(self, error_message: str) -> bool:
        """
        Handle processing failure.

        Args:
            error_message (str): The error message.

        Returns:
            bool: Always returns False.
        """
        self.document.process_status = "failed"
        self.document.processing_completed_at = timezone.now()
        self.document.save(update_fields=['process_status', 'processing_completed_at'])
        logger.error(f"Document processing failed: {error_message}")
        return False
