"""
Niyya Face Detector - Module de détection faciale.

Ce module fournit une API simple pour détecter la présence de visages
dans des images, destiné à la modération de contenu.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

from .exceptions import ImageProcessingError, ModelLoadingError


class FaceDetector:
    """
    Détecteur de visages pour la modération de contenu.

    Ce classe analyse des images et retourne des informations sur la
    présence de visages humains.

    Example:
        >>> detector = FaceDetector()
        >>> result = detector.analyze(image_bytes)
        >>> print(result['has_face'])
        True
    """

    def __init__(
        self,
        model_type: str = "haar",
        confidence_threshold: float = 0.5,
        model_path: Optional[Path] = None
    ):
        """
        Initialise le détecteur de visages.

        Args:
            model_type: Type de modèle ("haar" ou "dnn")
            confidence_threshold: Seuil de confiance pour DNN (0.0 à 1.0)
            model_path: Chemin vers les modèles DNN (optionnel)
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self._model_path = model_path

        if model_type == "haar":
            self._init_haar_model()
        elif model_type == "dnn":
            self._init_dnn_model()
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")

    def _init_haar_model(self) -> None:
        """Initialise le modèle Haar Cascades."""
        try:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except Exception as e:
            raise ModelLoadingError(f"Échec chargement Haar: {str(e)}")

    def _init_dnn_model(self) -> None:
        """Initialise le modèle DNN (CNN)."""
        if not self._model_path:
            raise ModelLoadingError("Chemin modèle requis pour DNN")

        prototxt = self._model_path / "deploy.prototxt"
        caffemodel = self._model_path / "res10_300x300_ssd_iter_140000.caffemodel"

        try:
            self._net = cv2.dnn.readNetFromCaffe(
                str(prototxt),
                str(caffemodel)
            )
        except Exception as e:
            raise ModelLoadingError(f"Échec chargement DNN: {str(e)}")

    def analyze(self, image_content: bytes) -> Dict[str, Any]:
        """
        Analyse une image pour détecter des visages.

        Args:
            image_content: Contenu binaire de l'image

        Returns:
            Dict contenant:
                - has_face (bool): Présence d'au moins un visage
                - face_count (int): Nombre de visages détectés
                - faces (list): Coordonnées des visages [(x, y, w, h), ...]
                - confidence (float): Confiance moyenne des détections

        Raises:
            ImageProcessingError: Si l'image ne peut être traitée
        """
        try:
            # Décodage de l'image
            nparr = np.frombuffer(image_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ImageProcessingError("Impossible de décoder l'image")

            # Analyse selon le modèle
            if self.model_type == "haar":
                faces, confidence = self._analyze_haar(img)
            else:
                faces, confidence = self._analyze_dnn(img)

            return {
                "has_face": len(faces) > 0,
                "face_count": len(faces),
                "faces": faces,
                "confidence": confidence,
                "model_type": self.model_type
            }

        except ImageProcessingError:
            raise
        except Exception as e:
            raise ImageProcessingError(f"Erreur traitement: {str(e)}")

    def _analyze_haar(self, img: np.ndarray) -> tuple:
        """Analyse avec Haar Cascades."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        # Haar ne retourne pas de confiance, on met 1.0 par défaut
        confidence = 1.0 if len(faces) > 0 else 0.0

        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces], confidence

    def _analyze_dnn(self, img: np.ndarray) -> tuple:
        """Analyse avec DNN (CNN)."""
        h, w = img.shape[:2]

        # Prétraitement
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0)
        )

        self._net.setInput(blob)
        detections = self._net.forward()

        faces = []
        confidences = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, w_box, h_box) = box.astype("int")
                faces.append((int(x), int(y), int(w_box), int(h_box)))
                confidences.append(float(confidence))

        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        return faces, avg_confidence

    def validate_for_upload(self, image_content: bytes) -> bool:
        """
        Vérifie si une image est valide pour upload (aucun visage).

        Args:
            image_content: Contenu binaire de l'image

        Returns:
            True si l'image peut être uploadée (pas de visage),
            False sinon
        """
        result = self.analyze(image_content)
        return not result["has_face"]