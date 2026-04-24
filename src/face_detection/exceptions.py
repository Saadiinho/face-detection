"""
Custom exceptions for the Niyya Face Detector package.

This module defines a hierarchy of custom exceptions designed to provide clear,
specific error handling for various failure scenarios within the detection pipeline.
All custom exceptions inherit from :class:`NiyyaDetectorError`, allowing users
to catch all package-specific errors with a single ``except`` block.

Example usage::

    from niyya_face_detector import FaceDetector
    from niyya_face_detector.exceptions import NiyyaDetectorError, ModelLoadingError

    try:
        detector = FaceDetector()
        result = detector.analyze("image.jpg")
    except ModelLoadingError as e:
        print(f"Critical AI model error: {e}")
    except NiyyaDetectorError as e:
        print(f"General detection error: {e}")
"""


class NiyyaDetectorError(Exception):
    """
    Base exception class for all Niyya Face Detector errors.

    This class serves as the root of the exception hierarchy for this package.
    Catching this exception will handle any custom error raised by the detector,
    including model loading failures, image processing issues, or invalid inputs.

    It is recommended to inherit from this class when creating new specific
    exceptions for future extensions of the package.

    :param message: A descriptive error message explaining the failure.
    :type message: str
    """

    pass


class ModelLoadingError(NiyyaDetectorError):
    """
    Exception raised when the AI model fails to initialize or load.

    This error occurs during the instantiation of a detector class (e.g.,
    :class:`~niyya_face_detector.detector.FaceDetector` or
    :class:`~niyya_face_detector.detector.AdvancedFaceDetector`).

    Common causes include:
    * Missing model weight files (e.g., `.caffemodel`, `.onnx`).
    * Corrupted model configuration files (e.g., `.prototxt`).
    * Incompatible hardware resources (e.g., insufficient RAM, missing GPU drivers).
    * Version conflicts in underlying dependencies (OpenCV, InsightFace, ONNX Runtime).

    If this exception is raised, the detector instance is not created, and
    the application should typically halt or fallback to a safe state, as
    no detection can proceed without a loaded model.

    Example::

        try:
            detector = AdvancedFaceDetector(model_path="/missing/path")
        except ModelLoadingError as e:
            logger.critical(f"System startup failed: {e}")
    """

    pass


class ImageProcessingError(NiyyaDetectorError):
    """
    Exception raised when an error occurs during image analysis or manipulation.

    This error is triggered during the execution of methods like ``analyze()``,
    ``analyze_bytes()``, or ``blur_faces()`` after the model has been successfully
    loaded.

    Common causes include:
    * The input image file is corrupted, unreadable, or in an unsupported format.
    * The image data is empty or malformed (e.g., zero bytes).
    * Memory allocation failures when processing extremely large images.
    * Internal errors within OpenCV or Pillow during resizing, decoding, or blurring.

    Unlike :class:`ModelLoadingError`, this indicates a runtime issue with specific
    data rather than a systemic configuration problem.

    Example::

        try:
            result = detector.analyze("corrupted_image.jpg")
        except ImageProcessingError as e:
            logger.warning(f"Failed to process user upload: {e}")
            return {"error": "Invalid image file"}
    """

    pass


class InvalidImageError(ImageProcessingError):
    """
    Exception raised specifically when the input image is invalid.

    This is a specialized subclass of :class:`ImageProcessingError` used to
    distinguish between general processing failures and issues with the input
    data itself.

    It is raised when:
    * The image cannot be decoded by OpenCV (returns ``None``).
    * The image dimensions are outside acceptable limits (too small or too large).
    * The file exists but contains no valid image data.

    Using this specific exception allows API endpoints to return a ``400 Bad Request``
    status code to the user, indicating that the fault lies with the uploaded file.
    """

    pass
