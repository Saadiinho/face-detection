"""
Automatic face blurring utilities for privacy preservation.

This module provides functionality to apply Gaussian blur to specific regions
of an image, typically faces detected by the :mod:`niyya_face_detector.detector` module.
It supports flexible configuration for blur intensity, padding, and minimum detection size,
ensuring robust privacy masking while preserving image quality in non-sensitive areas.

The core function :func:`auto_blur_faces` handles image loading, detection validation,
coordinate clamping, and the actual blurring process using the Pillow library.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

from PIL import Image, ImageFilter

# Ensure correct import path relative to your package structure
from .types import FaceBox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlurConfig:
    """
    Configuration container for the face blurring process.

    This immutable data class allows users to customize the behavior of the
    blurring algorithm, controlling the intensity of the blur, the margin around
    detected faces, and filtering criteria for small detections.

    :param blur_radius: The radius of the Gaussian blur kernel in pixels.
        Higher values result in stronger obscuration. Defaults to ``15``.
    :type blur_radius: int
    :param bbox_padding: Number of pixels to add around each detected bounding box
        before applying blur. This ensures the edges of the face are fully covered.
        Defaults to ``10``.
    :type bbox_padding: int
    :param min_face_size: Minimum width or height (in pixels) for a detection to be
        considered valid. Detections smaller than this are ignored as potential noise.
        Defaults to ``20``.
    :type min_face_size: int
    :param preserve_original_if_empty: If ``True``, returns the original image unchanged
        when no valid faces are detected. If ``False``, behavior is the same but explicitly
        indicates no operation was performed. Defaults to ``True``.
    :type preserve_original_if_empty: bool

    :ivar blur_radius: Configured blur radius.
    :vartype blur_radius: int
    :ivar bbox_padding: Configured padding size.
    :vartype bbox_padding: int
    :ivar min_face_size: Configured minimum detection size.
    :vartype min_face_size: int
    :ivar preserve_original_if_empty: Configured preservation flag.
    :vartype preserve_original_if_empty: bool

    Example::

        # Strong blur with large padding
        config = BlurConfig(blur_radius=25, bbox_padding=20)
        result = auto_blur_faces("image.jpg", detections, config=config)
    """

    blur_radius: int = 15
    bbox_padding: int = 10
    min_face_size: int = 20
    preserve_original_if_empty: bool = True


@dataclass(frozen=True)
class BlurResult:
    """
    Result container for a face blurring operation.

    Holds the processed image along with metadata about the operation, including
    the number of faces blurred and the coordinates of the affected regions.

    :param image: The resulting PIL Image object with blur applied (or original if none).
    :type image: PIL.Image.Image
    :param faces_detected: The number of valid faces that were successfully blurred.
    :type faces_detected: int
    :param was_blurred: Boolean flag indicating whether any blurring operation was
        actually performed (``True``) or if the image was returned unchanged (``False``).
    :type was_blurred: bool
    :param bounding_boxes: List of tuples representing the final coordinates
        ``(x1, y1, x2, y2)`` of the blurred regions, including any applied padding.
    :type bounding_boxes: List[Tuple[int, int, int, int]]

    :ivar image: Processed image object.
    :vartype image: PIL.Image.Image
    :ivar faces_detected: Count of blurred faces.
    :vartype faces_detected: int
    :ivar was_blurred: Operation status flag.
    :vartype was_blurred: bool
    :ivar bounding_boxes: List of blurred region coordinates.
    :vartype bounding_boxes: List[Tuple[int, int, int, int]]
    """

    image: Image.Image
    faces_detected: int
    was_blurred: bool
    bounding_boxes: List[Tuple[int, int, int, int]]


def auto_blur_faces(
    image: Union[str, Path, Image.Image],
    detections: List[FaceBox],
    config: BlurConfig | None = None,
) -> BlurResult:
    """
    Automatically detect and blur faces in an image based on provided detections.

    This function takes an input image (either a file path or a PIL Image object)
    and a list of detected face regions. It validates the detections against size
    constraints, applies optional padding, and overlays a Gaussian blur on the
    specified regions.

    The process follows these steps:
    1. **Load**: Opens the image if a path is provided.
    2. **Filter**: Discards detections that are malformed or too small (noise).
    3. **Clamp**: Adjusts bounding box coordinates to ensure they stay within image boundaries.
    4. **Blur**: Crops, blurs, and pastes back the affected regions.

    :param image: The source image to process. Can be a file path (``str`` or ``Path``)
        or an already loaded ``PIL.Image`` object.
    :type image: str, pathlib.Path, or PIL.Image.Image
    :param detections: A list of :class:`FaceBox` objects containing bounding box
        coordinates for detected faces.
    :type detections: List[FaceBox]
    :param config: Optional :class:`BlurConfig` instance to customize blurring behavior.
        If ``None``, default settings are used.
    :type config: BlurConfig, optional

    :return: A :class:`BlurResult` object containing the processed image and operation metadata.
    :rtype: BlurResult

    :raises FileNotFoundError: If the provided image path does not exist.
    :raises IOError: If the image file cannot be opened or is corrupted.

    Example::

        from niyya_face_detector import auto_blur_faces, BlurConfig

        # Define custom config
        cfg = BlurConfig(blur_radius=20, bbox_padding=15)

        # Run blurring
        result = auto_blur_faces("photo.jpg", detected_faces, config=cfg)

        if result.was_blurred:
            print(f"Blurred {result.faces_detected} faces.")
            result.image.save("output_blurred.jpg")
    """
    cfg = config or BlurConfig()

    # 1. Load Image
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.copy()

    # 2. Filter Valid Detections
    valid_detections = []
    for det in detections:
        # Handle both dict-like access (if legacy) and attribute access (FaceBox)
        bbox = det.bbox if hasattr(det, "bbox") else det.get("bbox")

        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1

        # Ignore detections smaller than minimum threshold
        if width < cfg.min_face_size or height < cfg.min_face_size:
            continue

        valid_detections.append((x1, y1, x2, y2))

    # 3. Handle Empty Case
    if not valid_detections:
        logger.debug("No valid faces detected; returning original image.")
        return BlurResult(
            image=img, faces_detected=0, was_blurred=False, bounding_boxes=[]
        )

    # 4. Apply Blur to Each Region
    img_w, img_h = img.size
    processed_boxes = []

    for x1, y1, x2, y2 in valid_detections:
        # Apply padding and clamp to image boundaries
        x1_padded = max(0, x1 - cfg.bbox_padding)
        y1_padded = max(0, y1 - cfg.bbox_padding)
        x2_padded = min(img_w, x2 + cfg.bbox_padding)
        y2_padded = min(img_h, y2 + cfg.bbox_padding)

        # Skip if region becomes invalid after padding
        if x2_padded <= x1_padded or y2_padded <= y1_padded:
            continue

        # Crop → Blur → Paste
        region = img.crop((x1_padded, y1_padded, x2_padded, y2_padded))
        blurred_region = region.filter(ImageFilter.GaussianBlur(radius=cfg.blur_radius))
        img.paste(blurred_region, (x1_padded, y1_padded))

        processed_boxes.append((x1_padded, y1_padded, x2_padded, y2_padded))

    logger.debug(
        "Auto-blur applied: %d faces processed, radius=%d, padding=%d",
        len(processed_boxes),
        cfg.blur_radius,
        cfg.bbox_padding,
    )

    return BlurResult(
        image=img,
        faces_detected=len(processed_boxes),
        was_blurred=True,
        bounding_boxes=processed_boxes,
    )
