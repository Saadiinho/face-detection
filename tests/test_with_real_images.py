from src.face_detection.detector import FaceDetector, AdvancedFaceDetector
from tests.test_utils import get_all_images


class TestFaceDetectorRealImages:
    def test_face_detector(self):
        """Les tests doivent être supérieur à 90%"""
        image_file = get_all_images()

        detector = FaceDetector()
        nb_true = 0
        for image in image_file:
            result = detector.analyze(str(image))
            nb_true += 1 if result["has_face"] else 0
        assert nb_true >= 90

    def test_advanced_face_detector(self):
        image_file = get_all_images()

        detector = AdvancedFaceDetector()
        nb_true = 0
        for image in image_file:
            result = detector.analyze(str(image))
            nb_true += 1 if result["has_face"] else 0
        assert nb_true >= 90
