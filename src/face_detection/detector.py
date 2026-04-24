"""
Core detection engines for the Niyya Face Detector package.

This module provides two primary detector classes:
- :class:`FaceDetector`: A lightweight detector supporting Haar Cascades and DNN models,
  optimized for speed and basic use cases.
- :class:`AdvancedFaceDetector`: A high-accuracy detector using RetinaFace via InsightFace,
  featuring robust fallback mechanisms for partially obscured faces (e.g., Niqab).

Both classes return structured :class:`~niyya_face_detector.types.DetectionResult` objects
and support automatic face blurring capabilities.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from .auto_blur import auto_blur_faces
from .exceptions import ImageProcessingError, ModelLoadingError, InvalidImageError
from .types import DetectionResult, BlurDataResult, FaceBox

# Configure logger for the module
logger = logging.getLogger(__name__)


class FaceDetector:
    """
    A versatile facial detection engine supporting multiple backends.

    This class offers a balance between performance and accuracy by supporting
    both Haar Cascades (fast, CPU-based) and DNN models (slower, more accurate).
    It includes a built-in fallback mechanism to detect eyes when full faces
    are obscured, enhancing robustness for diverse demographic use cases.

    :param model_type: The detection backend to use. Options are ``'haar'`` or ``'dnn'``.
        Defaults to ``'haar'``.
    :type model_type: str
    :param confidence_threshold: Minimum confidence score required to consider a detection
        valid (applies only to DNN model). Defaults to ``0.5``.
    :type confidence_threshold: float
    :param model_path: Required path to the DNN model files (`.prototxt` and `.caffemodel`)
        if ``model_type='dnn'``. Ignored for Haar cascades.
    :type model_path: pathlib.Path, optional
    :param use_eye_fallback: If ``True``, enables the secondary eye-detection mechanism
        when no full face is found. Defaults to ``True``.
    :type use_eye_fallback: bool

    :raises ValueError: If an unsupported ``model_type`` is provided.
    :raises ModelLoadingError: If the specified model files cannot be loaded or initialized.

    Example::

        # Fast Haar detection with eye fallback
        detector = FaceDetector(model_type="haar", use_eye_fallback=True)
        result = detector.analyze("photo.jpg")
        print(f"Faces found: {result.face_count}")
    """

    def __init__(
        self,
        model_type: str = "haar",
        confidence_threshold: float = 0.5,
        model_path: Optional[Path] = None,
        use_eye_fallback: bool = True,
    ):
        """
        Initialize the FaceDetector with the specified configuration.

        Loads the selected computer vision model into memory immediately upon instantiation.
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self._model_path = model_path
        self.use_eye_fallback = use_eye_fallback

        if model_type == "haar":
            self._init_haar_model()
        elif model_type == "dnn":
            self._init_dnn_model()
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. Choose 'haar' or 'dnn'."
            )

        # Initialize eye detectors if fallback is enabled
        if self.use_eye_fallback:
            self._init_eye_models()

    ########################
    ##   Initialization   ##
    ########################

    def _init_haar_model(self) -> None:
        """
        Load the Haar Cascade classifier for frontal face detection.

        Uses the pre-trained model included in OpenCV's data repository.

        :raises ModelLoadingError: If the Haar cascade XML file cannot be loaded.
        """
        try:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except Exception as e:
            raise ModelLoadingError(f"Failed to load Haar cascade: {str(e)}") from e

    def _init_dnn_model(self) -> None:
        """
        Load the Deep Neural Network (DNN) model for face detection.

        Requires a Caffe model (`.prototxt` and `.caffemodel`) located at ``model_path``.

        :raises ModelLoadingError: If ``model_path`` is missing or model files cannot be read.
        """
        if not self._model_path:
            raise ModelLoadingError("Model path is required for DNN initialization.")

        prototxt = self._model_path / "deploy.prototxt"
        caffemodel = self._model_path / "res10_300x300_ssd_iter_140000.caffemodel"

        try:
            self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        except Exception as e:
            raise ModelLoadingError(f"Failed to load DNN model: {str(e)}") from e

    def _init_eye_models(self) -> None:
        """
        Load Haar Cascades specifically trained for eye detection.

        Loads both standard eye cascades and cascades for eyes with glasses to improve
        detection rates across different subjects. Failure to load these models is non-critical;
        a warning is logged, and the fallback feature is silently disabled.
        """
        try:
            self._eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )
            self._eye_glasses_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
            )
        except Exception as e:
            logger.warning(f"Eye fallback disabled due to loading error: {e}")
            self._eye_cascade = None
            self._eye_glasses_cascade = None

    #########################
    ##      Analysis       ##
    #########################

    def _detect_eyes_fallback(
        self, img: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], float]:
        """
        Perform eye detection as a fallback when face detection fails.

        Scans the image for eye patterns using Haar Cascades. If eyes are found,
        their coordinates are returned to be treated as regions of interest.

        :param img: The input image as a NumPy array (BGR format).
        :type img: numpy.ndarray

        :return: A tuple containing a list of eye bounding boxes ``(x, y, w, h)``
            and a fixed confidence score (``0.85`` if eyes found, ``0.0`` otherwise).
        :rtype: tuple(List[tuple], float)
        """
        if not hasattr(self, "_eye_cascade") or self._eye_cascade is None:
            return [], 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect standard eyes
        eyes = self._eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10)
        )

        # Detect eyes with glasses
        eyes_glasses = []
        if self._eye_glasses_cascade:
            eyes_glasses = self._eye_glasses_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10)
            )

        all_eyes = list(eyes) + list(eyes_glasses)
        confidence = 0.85 if len(all_eyes) > 0 else 0.0

        return [
            (int(x), int(y), int(w), int(h)) for (x, y, w, h) in all_eyes
        ], confidence

    def _analyze_haar(self, img: np.ndarray) -> tuple:
        """
        Detect faces using the Haar Cascade method.

        :param img: Input image in BGR format.
        :type img: numpy.ndarray
        :return: Tuple of (list of face rectangles, confidence score).
        :rtype: tuple
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        # Haar does not provide confidence scores; default to 1.0 if detected
        confidence = 1.0 if len(faces) > 0 else 0.0
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces], confidence

    def _analyze_dnn(self, img: np.ndarray) -> tuple:
        """
        Detect faces using the Deep Neural Network (DNN) method.

        :param img: Input image in BGR format.
        :type img: numpy.ndarray
        :return: Tuple of (list of face rectangles, average confidence score).
        :rtype: tuple
        """
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        self._net.setInput(blob)
        detections = self._net.forward()
        faces = []
        confidences = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, w_box, h_box = box.astype("int")
                faces.append((int(x), int(y), int(w_box), int(h_box)))
                confidences.append(float(confidence))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return faces, avg_confidence

    def _analyze_bytes(self, image_path: str, image_content: bytes) -> DetectionResult:
        """
        Internal method to analyze image content from bytes.

        Decodes the image, runs the selected detection model, applies eye fallback
        if necessary, and formats the result into a :class:`DetectionResult` object.

        :param image_path: Identifier or path for the image (used in metadata).
        :type image_path: str
        :param image_content: Raw binary content of the image.
        :type image_content: bytes

        :return: Structured detection result.
        :rtype: DetectionResult

        :raises ImageProcessingError: If decoding fails or an unexpected error occurs.
        """
        try:
            nparr = np.frombuffer(image_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise InvalidImageError(
                    "Failed to decode image: content may be corrupted or unsupported."
                )

            # 1. Primary Detection
            if self.model_type == "haar":
                rects, raw_confidence = self._analyze_haar(img)
            else:
                rects, raw_confidence = self._analyze_dnn(img)

            # 2. Robustness: Eye Fallback
            detected_rects = rects
            final_confidence = raw_confidence
            detection_source = self.model_type

            if len(rects) == 0 and self.use_eye_fallback:
                eye_rects, eye_conf = self._detect_eyes_fallback(img)
                if len(eye_rects) > 0:
                    detected_rects = eye_rects
                    final_confidence = eye_conf
                    detection_source = "eyes_fallback"
                    logger.info(
                        f"Eye fallback triggered for {image_path} ({len(eye_rects)} eyes found)."
                    )

            # 3. Format Results
            faces_data = []
            if isinstance(final_confidence, list):
                conf_list = final_confidence
            elif isinstance(final_confidence, (int, float)):
                conf_list = [final_confidence] * len(detected_rects)
            else:
                conf_list = [0.0] * len(detected_rects)

            for i, rect in enumerate(detected_rects):
                conf_val = conf_list[i] if i < len(conf_list) else 0.0
                bbox = rect.tolist() if hasattr(rect, "tolist") else list(rect)

                faces_data.append(
                    FaceBox(
                        bbox=bbox,
                        confidence=float(conf_val),
                        source=detection_source,
                    )
                )

            global_confidence = (
                max([f.confidence for f in faces_data]) if faces_data else 0.0
            )

            return DetectionResult(
                image_path=image_path,
                has_face=len(faces_data) > 0,
                face_count=len(faces_data),
                confidence=global_confidence,
                model_type=detection_source,
                faces=faces_data,
            )

        except InvalidImageError:
            raise
        except Exception as e:
            logger.error(f"Unexpected processing error: {e}", exc_info=True)
            raise ImageProcessingError(f"Image processing failed: {str(e)}") from e

    def analyze(self, image_path: str) -> DetectionResult:
        """
        Analyze an image file to detect faces or eyes.

        Reads the image from the specified path, validates its existence and content,
        and processes it using the configured detection model.

        :param image_path: Absolute or relative path to the image file.
        :type image_path: str

        :return: Structured detection result containing face count, confidence, and coordinates.
        :rtype: DetectionResult

        :raises FileNotFoundError: If the specified file does not exist.
        :raises ImageProcessingError: If the file is empty, unreadable, or processing fails.
        :raises PermissionError: If read permissions are denied.
        """
        image_file = Path(image_path)

        if not image_file.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        if not image_file.is_file():
            raise ImageProcessingError(f"Path is not a file: {image_path}")

        try:
            with open(image_file, "rb") as f:
                image_bytes = f.read()

            if not image_bytes:
                raise InvalidImageError(f"Image file is empty: {image_path}")

            return self._analyze_bytes(image_path, image_bytes)

        except PermissionError:
            raise ImageProcessingError(f"Permission denied reading file: {image_path}")
        except IOError as e:
            raise ImageProcessingError(f"IO error reading file: {str(e)}") from e

    def blur_faces(
        self, image_path: str, filename: str = "result.jpg"
    ) -> Optional[BlurDataResult]:
        """
        Detect faces in an image and apply a Gaussian blur to obscure them.

        This method combines detection and privacy masking. If faces (or eyes) are found,
        they are blurred and saved to a timestamped subdirectory.

        :param image_path: Path to the source image.
        :type image_path: str
        :param filename: Desired name for the output blurred image. Defaults to ``'result.jpg'``.
        :type filename: str, optional

        :return: A :class:`BlurDataResult` object with paths to original and blurred images,
            or ``None`` if no faces were detected to blur.
        :rtype: BlurDataResult or None

        :raises Exception: If saving the output image fails.
        """
        result_analysis = self.analyze(image_path)

        if not result_analysis.has_face:
            return None

        blur_result = auto_blur_faces(str(image_path), result_analysis.faces)

        if not blur_result.was_blurred:
            return None

        # Prepare output path
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / date_str
        output_dir.mkdir(parents=True, exist_ok=True)

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filename += ".jpg"

        final_path = output_dir / filename

        try:
            blur_result.image.save(str(final_path))
        except Exception as e:
            logger.error(f"Failed to save blurred image: {e}")
            raise ImageProcessingError(f"Failed to save output image: {str(e)}") from e

        return BlurDataResult(
            real_image=str(image_path),
            blurred_image=str(final_path.resolve()),
            done=True,
            faces_detected=blur_result.faces_detected,
            detection_method=result_analysis.model_type,
        )


class AdvancedFaceDetector:
    """
    High-accuracy facial detection engine using RetinaFace.

    This class leverages the InsightFace library and the RetinaFace model to provide
    state-of-the-art detection performance, particularly effective on small faces,
    varied angles, and partially occluded faces. It includes a robust Haar-based
    eye detection fallback for extreme occlusion cases (e.g., Niqab).

    :param verbose: If ``True``, enables verbose logging from the underlying InsightFace model.
        Defaults to ``False``.
    :type verbose: bool, optional
    :param use_eye_fallback: If ``True``, enables the secondary eye-detection mechanism
        when RetinaFace finds no faces. Defaults to ``True``.
    :type use_eye_fallback: bool, optional

    :raises Exception: If the RetinaFace model fails to initialize or if eye fallback
        loading fails critically.

    Example::

        detector = AdvancedFaceDetector(use_eye_fallback=True)
        result = detector.analyze("crowd.jpg")
        print(f"Detected {result.face_count} faces with max confidence {result.confidence:.2f}")
    """

    def __init__(self, verbose: bool = False, use_eye_fallback: bool = True):
        """
        Initialize the AdvancedFaceDetector with RetinaFace.

        Prepares the model for inference on CPU (ctx_id=-1) with a balanced
        detection size.
        """
        self.app = FaceAnalysis(allowed_modules=["detection"], verbose=verbose)
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.model_type = "RetinaFace"
        self.use_eye_fallback = use_eye_fallback

        # Load Haar cascades for eye fallback
        if self.use_eye_fallback:
            try:
                self._eye_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_eye.xml"
                )
            except Exception as e:
                self._eye_cascade = None
                # Critical failure for fallback mechanism
                raise RuntimeError(
                    f"Failed to initialize eye fallback cascade: {e}"
                ) from e

    def _detect_eyes_fallback(self, img: np.ndarray) -> List[FaceBox]:
        """
        Detect eyes using Haar Cascades as a fallback when RetinaFace fails.

        Implements logic to expand bounding boxes around detected eyes to cover
        the surrounding orbital region, and merges multiple detections into a
        single region if necessary. Ensures all coordinates are clamped within
        image boundaries.

        :param img: Input image in BGR format.
        :type img: numpy.ndarray

        :return: List of :class:`FaceBox` objects representing detected eye regions.
        :rtype: List[FaceBox]
        """
        if self._eye_cascade is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape[:2]

        # Sensitive detection settings
        eyes = self._eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10)
        )

        faces_data: List[FaceBox] = []
        expansion_factor = 2.5

        for x, y, w_eye, h_eye in eyes:
            # Calculate expanded box centered on the eye
            center_x = x + w_eye / 2
            center_y = y + h_eye / 2
            new_w = int(w_eye * expansion_factor)
            new_h = int(h_eye * expansion_factor)
            new_x = int(center_x - new_w / 2)
            new_y = int(center_y - new_h / 2)

            # Clamp coordinates to image boundaries
            new_x = max(0, new_x)
            new_y = max(0, new_y)
            new_w = min(new_w, w - new_x)
            new_h = min(new_h, h - new_y)

            # Skip invalid boxes
            if new_w <= 0 or new_h <= 0:
                continue

            faces_data.append(
                FaceBox(
                    bbox=[new_x, new_y, new_x + new_w, new_y + new_h],
                    confidence=0.85,
                    source="eyes_fallback",
                )
            )

        # Merge multiple detections into a single bounding box
        if len(faces_data) > 1:
            min_x = min(f.bbox[0] for f in faces_data)
            min_y = min(f.bbox[1] for f in faces_data)
            max_x = max(f.bbox[2] for f in faces_data)
            max_y = max(f.bbox[3] for f in faces_data)

            faces_data = [
                FaceBox(
                    bbox=[min_x, min_y, max_x, max_y],
                    confidence=0.85,
                    source="eyes_fallback_merged",
                )
            ]

        return faces_data

    def analyze(self, image_path: str) -> DetectionResult:
        """
        Analyze an image using RetinaFace with eye fallback support.

        Attempts high-accuracy face detection first. If no faces are found and
        fallback is enabled, attempts eye detection. Returns a unified result object.

        :param image_path: Path to the image file.
        :type image_path: str

        :return: Structured detection result.
        :rtype: DetectionResult
        """
        img = cv2.imread(image_path)

        # Handle unreadable images gracefully
        if img is None:
            logger.warning(f"Could not read image: {image_path}")
            return DetectionResult(
                image_path=image_path,
                has_face=False,
                face_count=0,
                confidence=0.0,
                model_type=self.model_type,
                faces=[],
            )

        faces = self.app.get(img)
        faces_data = []
        max_confidence = 0.0

        for face in faces:
            bbox = face["bbox"].astype(int).tolist()
            score = float(face["det_score"])
            if score > max_confidence:
                max_confidence = score

            faces_data.append(
                FaceBox(
                    bbox=bbox,
                    confidence=score,
                    source="RetinaFace",
                )
            )

        # Robustness: Fallback to eyes if no face found
        if len(faces_data) == 0 and self.use_eye_fallback:
            eye_data = self._detect_eyes_fallback(img)
            if eye_data:
                faces_data = eye_data
                max_confidence = 0.85
                logger.info(f"Eye fallback triggered for {image_path}.")

        return DetectionResult(
            image_path=image_path,
            has_face=len(faces_data) > 0,
            face_count=len(faces_data),
            confidence=max_confidence,
            model_type=(
                "eyes_fallback"
                if (len(faces_data) > 0 and faces_data[0].source == "eyes_fallback")
                else self.model_type
            ),
            faces=faces_data,
        )

    def blur_faces(
        self, image_path: str, filename: str = "result.jpg"
    ) -> Optional[BlurDataResult]:
        """
        Detect and blur faces (or eyes) in an image.

        Uses the advanced detection pipeline and applies privacy blurring to
        all detected regions.

        :param image_path: Path to the source image.
        :type image_path: str
        :param filename: Output filename. Defaults to ``'result.jpg'``.
        :type filename: str, optional

        :return: Result metadata if successful, ``None`` if no faces detected.
        :rtype: BlurDataResult or None
        """
        result_analysis = self.analyze(image_path)

        if not result_analysis.has_face:
            return None

        blur_result = auto_blur_faces(str(image_path), result_analysis.faces)

        if not blur_result.was_blurred:
            return None

        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / date_str
        output_dir.mkdir(parents=True, exist_ok=True)

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            filename += ".jpg"

        final_path = output_dir / filename

        try:
            blur_result.image.save(str(final_path))
        except Exception as e:
            logger.error(f"Failed to save blurred image: {e}")
            return None

        return BlurDataResult(
            real_image=str(image_path),
            blurred_image=str(final_path.resolve()),
            done=True,
            faces_detected=blur_result.faces_detected,
            detection_method=result_analysis.model_type,
        )
