"""
Niyya Face Detector - Module de détection faciale.

Ce module fournit une API simple pour détecter la présence de visages
dans des images, destiné à la modération de contenu.
"""

from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

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
        model_path: Optional[Path] = None,
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
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
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
            self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        except Exception as e:
            raise ModelLoadingError(f"Échec chargement DNN: {str(e)}")

    def _analyze_bytes(self, image_content: bytes) -> Dict[str, Any]:
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
                "confidence": confidence,
                "model_type": self.model_type,
            }

        except ImageProcessingError:
            raise
        except Exception as e:
            raise ImageProcessingError(f"Erreur traitement: {str(e)}")

    def _analyze_haar(self, img: np.ndarray) -> tuple:
        """Analyse avec Haar Cascades."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )

        # Haar ne retourne pas de confiance, on met 1.0 par défaut
        confidence = 1.0 if len(faces) > 0 else 0.0

        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces], confidence

    def _analyze_dnn(self, img: np.ndarray) -> tuple:
        """Analyse avec DNN (CNN)."""
        h, w = img.shape[:2]

        # Prétraitement
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

    def analyze(self, image_path: str) -> Dict[str, Any]:
        """
        Analyse une image depuis un chemin de fichier.

        Méthode principale d'entrée pour analyser une image stockée sur disque.
        Lit le fichier, le convertit en bytes, puis appelle analyze_bytes.

        Args:
            image_path: Chemin absolu ou relatif vers le fichier image

        Returns:
            Dict contenant:
                - has_face (bool): Présence d'au moins un visage
                - face_count (int): Nombre de visages détectés
                - faces (list): Coordonnées des visages [(x, y, w, h), ...]
                - confidence (float): Confiance moyenne des détections
                - model_type (str): Type de modèle utilisé

        Raises:
            FileNotFoundError: Si le fichier n'existe pas
            ImageProcessingError: Si l'image ne peut être traitée

        Example:
            >>> detector = FaceDetector(model_type="haar")
            >>> result = detector.analyze("tests/fixtures/face1.jpg")
            >>> print(result["has_face"])
            True
        """
        from pathlib import Path

        image_file = Path(image_path)

        # Vérification que le fichier existe
        if not image_file.exists():
            raise FileNotFoundError(f"Le fichier '{image_path}' n'existe pas")

        # Vérification que c'est bien un fichier
        if not image_file.is_file():
            raise ImageProcessingError(f"'{image_path}' n'est pas un fichier")

        try:
            # Lecture du fichier en mode binaire
            with open(image_file, "rb") as f:
                image_bytes = f.read()

            # Vérification que le fichier n'est pas vide
            if not image_bytes:
                raise ImageProcessingError(f"Le fichier '{image_path}' est vide")

            # Appel de la méthode d'analyse existante
            return self._analyze_bytes(image_bytes)

        except PermissionError:
            raise ImageProcessingError(f"Permission refusée pour lire '{image_path}'")
        except IOError as e:
            raise ImageProcessingError(f"Erreur de lecture: {str(e)}")


class AdvancedFaceDetector:
    """Détecteur avancé avec RetinaFace - Meilleure détection hijab."""

    def __init__(self, verbose: bool = False):
        # Initialisation de RetinaFace
        self.app = FaceAnalysis(allowed_modules=["detection"], verbose=verbose)
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.model_type = "RetinaFace"

    def analyze(self, image_path: str):
        """Analyse une image."""
        img = cv2.imread(image_path)
        faces = self.app.get(img)

        score = None
        for i, face in enumerate(faces):
            bbox = face["bbox"].astype(int)
            score = face["det_score"]
            print(f"   Visage {i + 1}: confiance={score:.2%}")
            print(
                f"   Coordonnées: x={bbox[0]}, y={bbox[1]}, w={bbox[2] - bbox[0]}, h={bbox[3] - bbox[1]}"
            )

        return {
            "has_face": len(faces) > 0,
            "face_count": len(faces),
            "confidence": f"{score:.2%}" if score is not None else "0.0%",
            "model_type": self.model_type,
        }
