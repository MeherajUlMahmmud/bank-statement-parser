import logging
import time
from typing import Dict, Any
from pathlib import Path

from ..agents.ocr_cleanup_agent import OCRCleanupAgent
from ..agents.extraction_agent import ExtractionAgent
from ..agents.normalization_agent import NormalizationAgent
from ..services.olm_ocr_service import OlmOCRService
from ..services.pdf_service import PDFProcessingService

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrator for the multi-agent OCR pipeline.

    Pipeline stages:
    1. PDF → Images (PDFProcessingService)
    2. Images → Raw OCR Text (OlmOCRService)
    3. Raw OCR → Cleaned Text (Agent 1: OCRCleanupAgent)
    4. Cleaned Text → Structured Data (Agent 2: ExtractionAgent)
    5. Structured Data → Normalized & Validated (Agent 3: NormalizationAgent)
    """

    def __init__(self):
        """Initialize the pipeline orchestrator."""
        self.pdf_service = PDFProcessingService()
        self.ocr_service = OlmOCRService()
        self.agent1 = OCRCleanupAgent()
        self.agent2 = ExtractionAgent()
        self.agent3 = NormalizationAgent()
        logger.info("PipelineOrchestrator initialized with all agents")

    async def process_bank_statement(self, pdf_path: str) -> Dict[str, Any]:
        """
        Process a bank statement PDF through the complete pipeline.

        Args:
            pdf_path: Path to bank statement PDF

        Returns:
            dict: Complete processing results with all stage outputs
        """
        logger.info(f"=== Starting pipeline processing for: {pdf_path} ===")
        start_time = time.time()

        result = {
            'success': False,
            'pdf_path': pdf_path,
            'stages': {},
            'final_data': {},
            'validation_results': {},
            'metadata': {},
            'errors': []
        }

        try:
            # Stage 1: PDF to Images
            logger.info("Stage 1: Converting PDF to images")
            stage1_start = time.time()
            image_paths = self.pdf_service.pdf_to_images(pdf_path)
            pdf_metadata = self.pdf_service.get_pdf_metadata(pdf_path)

            result['stages']['pdf_conversion'] = {
                'success': True,
                'page_count': len(image_paths),
                'duration': time.time() - stage1_start,
                'metadata': pdf_metadata
            }
            logger.info(f"Stage 1: Complete - {len(image_paths)} pages converted")

            # Stage 2: OCR Extraction
            logger.info("Stage 2: Extracting text via OlmOCR")
            stage2_start = time.time()
            ocr_texts = await self.ocr_service.extract_text_from_images(image_paths)
            combined_ocr_text = "\n\n--- PAGE BREAK ---\n\n".join(ocr_texts)

            result['stages']['ocr_extraction'] = {
                'success': True,
                'pages_processed': len(ocr_texts),
                'total_characters': len(combined_ocr_text),
                'duration': time.time() - stage2_start
            }
            logger.info(f"Stage 2: Complete - {len(combined_ocr_text)} chars extracted")

            # Stage 3: Agent 1 - OCR Cleanup
            logger.info("Stage 3: Agent 1 - Cleaning OCR text")
            stage3_start = time.time()
            cleanup_result = await self.agent1.process(combined_ocr_text)

            if not cleanup_result['success']:
                result['errors'].append(f"Agent 1 failed: {cleanup_result['error']}")
                result['stages']['ocr_cleanup'] = cleanup_result
                return result

            cleaned_text = cleanup_result['cleaned_text']
            result['stages']['ocr_cleanup'] = {
                **cleanup_result,
                'duration': time.time() - stage3_start
            }
            logger.info(f"Stage 3: Complete - {len(cleaned_text)} chars cleaned")

            # Stage 4: Agent 2 - Data Extraction
            logger.info("Stage 4: Agent 2 - Extracting structured data")
            stage4_start = time.time()
            extraction_result = await self.agent2.process(cleaned_text)

            if not extraction_result['success']:
                result['errors'].append(f"Agent 2 failed: {extraction_result['error']}")
                result['stages']['data_extraction'] = extraction_result
                return result

            extracted_data = extraction_result['extracted_data']
            result['stages']['data_extraction'] = {
                **extraction_result,
                'duration': time.time() - stage4_start
            }
            logger.info(
                f"Stage 4: Complete - "
                f"{len(extracted_data.get('transactions', []))} transactions extracted"
            )

            # Stage 5: Agent 3 - Normalization & Validation
            logger.info("Stage 5: Agent 3 - Normalizing and validating")
            stage5_start = time.time()
            normalization_result = await self.agent3.process(extracted_data)

            if not normalization_result['success']:
                result['errors'].append(f"Agent 3 failed: {normalization_result['error']}")
                result['stages']['normalization'] = normalization_result
                return result

            result['stages']['normalization'] = {
                **normalization_result,
                'duration': time.time() - stage5_start
            }

            # Final results
            result['final_data'] = normalization_result['normalized_data']
            result['validation_results'] = normalization_result['validation_results']
            result['success'] = True

            # Metadata
            total_duration = time.time() - start_time
            result['metadata'] = {
                'total_duration': total_duration,
                'pdf_pages': len(image_paths),
                'transactions_found': len(result['final_data'].get('transactions', [])),
                'overall_confidence': result['validation_results'].get('overall_confidence', 0),
                'issues_count': len(result['validation_results'].get('issues', []))
            }

            logger.info(
                f"=== Pipeline Complete in {total_duration:.2f}s - "
                f"{result['metadata']['transactions_found']} transactions, "
                f"confidence: {result['metadata']['overall_confidence']:.2%} ==="
            )

            # Cleanup temp files
            self.pdf_service.cleanup_temp_images(image_paths)

            return result

        except Exception as e:
            logger.error(f"Pipeline error: {str(e)}", exc_info=True)
            result['errors'].append(str(e))
            result['metadata']['total_duration'] = time.time() - start_time
            return result
