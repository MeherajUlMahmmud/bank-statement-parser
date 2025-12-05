# import gc
# import logging
# from typing import List, Tuple, Optional

# import numpy as np
# from paddleocr import PaddleOCR

# from base import settings

# logger = logging.getLogger(__name__)


# class PaddleService:
#     """
#     Service class for performing OCR on images using PaddleOCR with memory optimization.

#     Provides methods to initialize PaddleOCR, crop images, and run OCR with resource management.
#     """

#     @classmethod
#     def _get_paddle_ocr(cls):
#         """
#         Create a new PaddleOCR instance for each call to avoid holding memory.

#         Returns:
#             PaddleOCR: A new PaddleOCR instance configured with settings.
#         """
#         logger.info("Initializing PaddleOCR instance...")
#         return PaddleOCR(
#             use_gpu=settings.USE_GPU,
#             lang='en',
#             det_model_dir=settings.DET_MODEL_DIR,
#             rec_model_dir=settings.REC_MODEL_DIR,
#             cls_model_dir=settings.CLS_MODEL_DIR,
#             use_angle_cls=settings.USE_ANGLE_CLS,
#             drop_score=settings.DROP_SCORE,
#         )

#     @staticmethod
#     def crop_image(image: np.ndarray, box: tuple) -> Optional[np.ndarray]:
#         """
#         Crop a region from the image based on the provided bounding box.

#         Args:
#             image (np.ndarray): The input image array.
#             box (tuple): A tuple of ((x_min, y_min), (x_max, y_max)), where values can be integers or 'min'/'max'.

#         Returns:
#             Optional[np.ndarray]: The cropped image region, or None if cropping fails.
#         """
#         try:
#             logger.info(f"Cropping image with box: {box}")

#             x_min, y_min = box[0]
#             x_max, y_max = box[1]

#             x_min = 0 if x_min == "min" else x_min
#             y_min = 0 if y_min == "min" else y_min
#             x_max = image.shape[1] if x_max == "max" else x_max
#             y_max = image.shape[0] if y_max == "max" else y_max

#             return image[y_min:y_max, x_min:x_max]

#         except Exception as e:
#             logger.error(f"Error cropping image: {e}")
#             return None

#     @classmethod
#     def run_paddle_ocr(cls, img: np.ndarray) -> Tuple[List, List[str], List[float]]:
#         """
#         Perform OCR on an image using PaddleOCR with memory optimization.

#         Args:
#             img (np.ndarray): Input image array.

#         Returns:
#             Tuple[List, List[str], List[float]]: A tuple containing:
#                 - List: Bounding boxes for detected text regions.
#                 - List[str]: Recognized text strings.
#                 - List[float]: Probabilities/confidence scores for each text.
#         """
#         logger.info("Entering method run_paddle_ocr...")

#         try:
#             # Get the OCR model
#             paddle_ocr = cls._get_paddle_ocr()

#             # Perform OCR
#             output = paddle_ocr.ocr(
#                 img,
#                 cls=False,  # Disable classifier to save memory
#                 bin=False,
#             )[0]

#             # Extract results
#             if output:
#                 boxes = [line[0] for line in output]
#                 texts = [line[1][0] for line in output]
#                 probabilities = [line[1][1] for line in output]
#             else:
#                 boxes, texts, probabilities = [], [], []

#             # Force garbage collection
#             import paddle
#             paddle.device.cuda.empty_cache()
#             gc.collect()

#             logger.info(
#                 f"OCR completed successfully, found {len(texts)} text elements")
#             return boxes, texts, probabilities

#         except Exception as e:
#             logger.error(f"Error in OCR processing: {str(e)}")
#             return [], [], []
#         finally:
#             # Ensure memory is freed
#             gc.collect()
#             logger.info("Exiting method run_paddle_ocr...")
