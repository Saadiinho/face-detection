"""
Data models for the Niyya Face Detector package.

This module defines structured, immutable data classes using Python's ``dataclasses``.
These classes ensure type safety, data validation, and consistent serialization
for detection results and blurring operations.

The use of ``frozen=True`` guarantees immutability, preventing accidental
modification of detection results after creation.
"""

import json
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass(frozen=True)
class FaceBox:
    """
    Represents a detected facial region or feature (e.g., eyes) within an image.

    This immutable data class encapsulates the bounding box coordinates,
    detection confidence, and the source model responsible for the detection.
    It includes automatic validation to ensure data integrity upon instantiation.

    :param bbox: A list of four integers representing the bounding box coordinates
        in the format ``[x1, y1, x2, y2]``, where ``(x1, y1)`` is the top-left
        corner and ``(x2, y2)`` is the bottom-right corner.
    :type bbox: List[int]
    :param confidence: The detection confidence score, normalized between ``0.0``
        (no confidence) and ``1.0`` (absolute certainty).
    :type confidence: float
    :param source: Identifier of the detection model or method used (e.g.,
        ``'RetinaFace'``, ``'Haar'``, ``'eyes_fallback'``). Defaults to ``'unknown'``.
    :type source: str, optional

    :ivar bbox: Validated bounding box coordinates.
    :vartype bbox: List[int]
    :ivar confidence: Validated confidence score.
    :vartype confidence: float
    :ivar source: Detection source identifier.
    :vartype source: str

    :raises ValueError: If ``bbox`` does not contain exactly 4 elements,
        if any coordinate cannot be converted to an integer, or if
        ``confidence`` is outside the valid range ``[0.0, 1.0]``.

    Example::

        box = FaceBox(bbox=[100, 100, 200, 200], confidence=0.98, source="RetinaFace")
        print(box.to_dict())
    """

    bbox: List[int]
    confidence: float
    source: str = "unknown"

    def __post_init__(self):
        """
        Validate and normalize data immediately after initialization.

        This method ensures that:
        1. The bounding box contains exactly 4 coordinates.
        2. All coordinates are converted to standard Python integers.
        3. The confidence score lies within the valid range [0.0, 1.0].

        Since the dataclass is frozen, attribute modification uses
        ``object.__setattr__``.

        :raises ValueError: If validation fails for bbox structure or confidence range.
        """
        # Validate BBox structure
        if len(self.bbox) != 4:
            raise ValueError(
                f"bbox must contain exactly 4 integers [x1, y1, x2, y2], got {len(self.bbox)}"
            )

        # Normalize coordinates to int (handles numpy types, floats, etc.)
        if not all(isinstance(x, int) for x in self.bbox):
            try:
                object.__setattr__(self, "bbox", [int(x) for x in self.bbox])
            except (TypeError, ValueError):
                raise ValueError("All bbox coordinates must be convertible to integers")

        # Validate Confidence range
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )

    def to_dict(self) -> dict:
        """
        Convert the FaceBox instance into a standard Python dictionary.

        This method is useful for serialization processes, such as converting
        the object to JSON or integrating with frameworks that expect dictionaries.

        :return: A dictionary representation of the FaceBox with keys
            ``'bbox'``, ``'confidence'``, and ``'source'``.
        :rtype: dict
        """
        return asdict(self)


@dataclass(frozen=True)
class DetectionResult:
    """
    Comprehensive result container for a facial detection analysis.

    This class aggregates all metadata resulting from an image analysis,
    including the presence of faces, total count, maximum confidence score,
    the model used, and detailed information for each detected face.

    It serves as the primary return type for the ``analyze`` method in
    detector classes.

    :param image_path: The file path or identifier of the analyzed image.
    :type image_path: str
    :param has_face: Boolean flag indicating whether at least one face or
        feature was successfully detected.
    :type has_face: bool
    :param face_count: The total number of distinct faces or features detected.
    :type face_count: int
    :param confidence: The highest confidence score among all detected items.
        Will be ``0.0`` if no faces were detected.
    :type confidence: float
    :param model_type: The identifier of the AI model that performed the
        successful detection (e.g., ``'RetinaFace'``, ``'eyes_fallback'``).
    :type model_type: str
    :param faces: A list of :class:`FaceBox` objects, each detailing a specific
        detection. Defaults to an empty list.
    :type faces: List[FaceBox], optional

    :ivar image_path: Path to the source image.
    :vartype image_path: str
    :ivar has_face: Detection status flag.
    :vartype has_face: bool
    :ivar face_count: Number of detected items.
    :vartype face_count: int
    :ivar confidence: Maximum confidence score.
    :vartype confidence: float
    :ivar model_type: Model identifier.
    :vartype model_type: str
    :ivar faces: List of detailed detection boxes.
    :vartype faces: List[FaceBox]
    """

    image_path: str
    has_face: bool
    face_count: int
    confidence: float
    model_type: str
    faces: List[FaceBox] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        Convert the entire detection result into a nested dictionary.

        Recursively converts the instance and all contained :class:`FaceBox`
        objects into a standard dictionary structure suitable for JSON serialization.

        :return: A dictionary containing all result metadata and a list of
            face dictionaries under the ``'faces'`` key.
        :rtype: dict
        """
        data = asdict(self)
        return data

    def to_json(self, indent: int = 2) -> str:
        """
        Serialize the detection result to a formatted JSON string.

        This convenience method wraps :meth:`to_dict` and directly returns
        a JSON-formatted string, ideal for API responses or logging.

        :param indent: The number of spaces to use for indentation in the
            output JSON string. Defaults to ``2``.
        :type indent: int, optional

        :return: A JSON string representation of the detection result.
        :rtype: str

        Example::

            result = detector.analyze("image.jpg")
            print(result.to_json(indent=4))
        """
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class BlurDataResult:
    """
    Result container for a face blurring operation.

    This class holds references to the original and processed images,
    along with operational metadata such as success status, the number
    of faces obscured, and the detection method utilized.

    It is typically returned by the ``blur_faces`` method after a
    successful privacy masking operation.

    :param real_image: The absolute file path to the original, unmodified image.
    :type real_image: str
    :param blurred_image: The absolute file path to the newly generated
        blurred image saved on disk.
    :type blurred_image: str
    :param done: Boolean flag indicating whether the blurring process
        completed successfully.
    :type done: bool
    :param faces_detected: The number of faces that were identified and
        subsequently blurred.
    :type faces_detected: int
    :param detection_method: The specific algorithm or fallback mechanism
        used to locate the faces before blurring.
    :type detection_method: str

    :ivar real_image: Path to the source image.
    :vartype real_image: str
    :ivar blurred_image: Path to the output blurred image.
    :vartype blurred_image: str
    :ivar done: Operation success status.
    :vartype done: bool
    :ivar faces_detected: Count of blurred faces.
    :vartype faces_detected: int
    :ivar detection_method: Identifier of the detection strategy used.
    :vartype detection_method: str
    """

    real_image: str
    blurred_image: str
    done: bool
    faces_detected: int
    detection_method: str
