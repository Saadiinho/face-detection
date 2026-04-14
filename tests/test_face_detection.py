"""Tests unitaires pour le détecteur de visages."""

from unittest.mock import patch, MagicMock

import pytest

from src.face_detection.detector import FaceDetector
from src.face_detection.exceptions import ModelLoadingError, ImageProcessingError


class TestFaceDetector:
    """Tests pour la classe FaceDetector."""

    def test_init_haar_model(self):
        """Teste l'initialisation avec Haar."""
        detector = FaceDetector(model_type="haar")
        assert detector.model_type == "haar"
        assert hasattr(detector, "_cascade")

    def test_init_dnn_without_path_raises_error(self):
        """Teste que DNN sans chemin lève une erreur."""
        with pytest.raises(ModelLoadingError):
            FaceDetector(model_type="dnn")

    @patch("src.face_detection.detector.cv2.imdecode")
    @patch("src.face_detection.detector.cv2.cvtColor")
    @patch("src.face_detection.detector.cv2.CascadeClassifier.detectMultiScale")
    def test_analyze_bytes_returns_face_detected(
        self, mock_detect, mock_cvt, mock_decode
    ):
        """Teste l'analyse avec visage détecté."""
        mock_decode.return_value = MagicMock()
        mock_cvt.return_value = MagicMock()
        mock_detect.return_value = [(100, 100, 50, 50)]

        detector = FaceDetector(model_type="haar")
        result = detector._analyze_bytes(b"fake_image")

        assert result["has_face"] is True
        assert result["face_count"] == 1

    @patch("src.face_detection.detector.cv2.imdecode")
    def test_analyze_bytes_invalid_image_raises_error(self, mock_decode):
        """Teste la gestion d'image invalide."""
        mock_decode.return_value = None

        detector = FaceDetector(model_type="haar")

        with pytest.raises(ImageProcessingError):
            detector._analyze_bytes(b"invalid_image")

    def test_validate_for_upload_returns_true_when_no_face(self):
        """Teste la validation upload sans visage."""
        # ✅ CORRECTION: Patch _analyze_bytes, pas analyze
        with patch.object(FaceDetector, "_analyze_bytes") as mock_analyze_bytes:
            mock_analyze_bytes.return_value = {"has_face": False}

            detector = FaceDetector(model_type="haar")
            assert detector.validate_for_upload(b"img") is True

            # Bonus: vérifie que la méthode a été appelée correctement
            mock_analyze_bytes.assert_called_once_with(b"img")

    def test_validate_for_upload_returns_false_when_face(self):
        """Teste la validation upload avec visage."""
        # ✅ CORRECTION: Patch _analyze_bytes, pas analyze
        with patch.object(FaceDetector, "_analyze_bytes") as mock_analyze_bytes:
            mock_analyze_bytes.return_value = {"has_face": True}

            detector = FaceDetector(model_type="haar")
            assert detector.validate_for_upload(b"img") is False

            # Bonus: vérifie que la méthode a été appelée correctement
            mock_analyze_bytes.assert_called_once_with(b"img")
