# import logging
# import os
# from typing import List, Dict, Union

# import numpy as np
# from base import settings
# from ultralytics import YOLO

# logger = logging.getLogger(__name__)


# class YoloService:
#     def __init__(
#             self,
#             model_path: str,
#             confidence_threshold: float = 0.5,
#     ):
#         """
#         Initializes the YoloService with a YOLO model.

#         Parameters:
#         model_path (str): Path to the trained YOLO model file (e.g., .pt).
#         confidence_threshold (float): Minimum confidence score for a detection to be considered valid.
#         """
#         logger.info(f"Initializing YoloService with model: {model_path}")
#         if not os.path.exists(model_path):
#             logger.error(f"YOLO model file not found at {model_path}")
#             raise FileNotFoundError(f"YOLO model file not found at {model_path}")

#         try:
#             self.model = YOLO(model_path, task='detect')
#             self.confidence_threshold = confidence_threshold

#             # Get class names from the model
#             self.class_names = self.model.names
#             self.classes_list = self.class_names.values()

#             logger.info(f"YOLO model loaded successfully with classes: {self.class_names}")
#         except ImportError:
#             logger.error("The 'ultralytics' library is not installed. Please install it: pip install ultralytics")
#             raise
#         except Exception as e:
#             logger.error(f"Error loading YOLO model from {model_path}: {e}")
#             raise

#     def get_image_sections(self, image: np.ndarray):
#         coordinates, confidences, classes = self.__detect_sections(image)

#         logger.info("Formatting the results...")
#         coordinates_dict = self._format_results(coordinates, confidences, classes)

#         image_section_dict = self.__get_images_from_coordinates(coordinates_dict, image)

#         return image_section_dict

#     def __detect_sections(self, image: np.ndarray) -> tuple:
#         """
#         Detects all the sections in the image using the loaded YOLO model.

#         Parameters:
#         image (np.ndarray): The input image (OpenCV format, BGR).

#         Returns:
#         Dict[str, List[Tuple[int, int, int, int]]]: A dictionary where keys are section names
#         and values are lists of bounding box coordinates (x_min, y_min, x_max, y_max).
#         """
#         logger.info("Attempting to detect sections in image.")

#         results = self.model.predict(
#             source=image,
#             conf=self.confidence_threshold,
#             save=settings.YOLO_SAVE_ANNOTATED_IMAGE,
#             save_crop=settings.YOLO_SAVE_CROPPED_IMAGE,
#             verbose=True,
#             project=settings.YOLO_PROJECT_PATH,
#         )

#         # Convert tensors to lists (following the working pattern from back_ocr_engine.py)
#         logger.info("Contours detected successfully")
#         logger.info("Converting tensors to list...")

#         coordinates = results[0].boxes.xyxy.tolist()  # convert tensor to list
#         confidences = results[0].boxes.conf.tolist()  # convert tensor to list
#         classes = results[0].boxes.cls.tolist()  # convert tensor to list

#         logger.info(f"Successfully converted tensors - {len(coordinates)} detections")

#         return coordinates, confidences, classes

#     # @staticmethod
#     def _format_results(self, coordinates: dict, confidences: dict, classes: dict) -> dict:
#         coordinates_dict = {class_name: [] for class_name in self.classes_list}
#         confidences_dict = {class_name: [] for class_name in self.classes_list}

#         for i, class_index in enumerate(classes):
#             # Append the coordinates and confidences to the corresponding class dictionary
#             class_name = self.class_names[int(class_index)]
#             coordinates_dict[class_name].append(coordinates[i])
#             confidences_dict[class_name].append(round(confidences[i], 2))

#         # Keep only the maximum confidence for each class
#         for class_name in self.classes_list:
#             confidences_list = confidences_dict[class_name]
#             if len(confidences_list) > 1:
#                 max_index = confidences_list.index(max(confidences_list))
#                 coordinates_dict[class_name] = [
#                     coordinates_dict[class_name][max_index]]
#                 confidences_dict[class_name] = [
#                     confidences_dict[class_name][max_index]]

#         return coordinates_dict

#     def __get_images_from_coordinates(
#             self,
#             coordinates_dict: Dict[str, List[float]],
#             image: np.ndarray,
#     ) -> Dict[str, Union[np.ndarray, None]]:
#         """
#         Extract images from the main image based on the provided coordinates.
#         """

#         images = {key: None for key in self.classes_list}

#         for key, value in coordinates_dict.items():
#             if not value or len(value[0]) < 4:
#                 logger.info(f"No valid coordinates found for {key}")
#                 continue

#             logger.info(f"Coordinates for {key}: {value}")
#             x1, y1, x2, y2 = map(int, value[0])
#             images[key] = image[y1:y2, x1:x2]

#         return images
