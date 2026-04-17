import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from .auto_blur import auto_blur_faces
from .exceptions import ImageProcessingError, ModelLoadingError


class FaceDetector:
    def __init__(
        self,
        model_type: str = "haar",
        confidence_threshold: float = 0.5,
        model_path: Optional[Path] = None,
    ):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self._model_path = model_path

        if model_type == "haar":
            self._init_haar_model()
        elif model_type == "dnn":
            self._init_dnn_model()
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")

    ########################
    ##   Initialisation   ##
    ########################

    def _init_haar_model(self) -> None:
        try:
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        except Exception as e:
            raise ModelLoadingError(f"Échec chargement Haar: {str(e)}")

    def _init_dnn_model(self) -> None:
        if not self._model_path:
            raise ModelLoadingError("Chemin modèle requis pour DNN")

        prototxt = self._model_path / "deploy.prototxt"
        caffemodel = self._model_path / "res10_300x300_ssd_iter_140000.caffemodel"

        try:
            self._net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))
        except Exception as e:
            raise ModelLoadingError(f"Échec chargement DNN: {str(e)}")

    #########################
    ##       Analyse       ##
    #########################

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

    def _analyze_bytes(self, image_path: str, image_content: bytes) -> Dict[str, Any]:
        try:
            # Décodage de l'image
            nparr = np.frombuffer(image_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ImageProcessingError("Impossible de décoder l'image")

            # Analyse selon le modèle
            if self.model_type == "haar":
                faces_rects, raw_confidence = self._analyze_haar(img)
            else:
                faces_rects, raw_confidence = self._analyze_dnn(img)

            # --- CORRECTION ICI : Gestion flexible des confiances ---
            faces_data = []

            # Normaliser raw_confidence en liste si ce n'en est pas une
            if isinstance(raw_confidence, list):
                conf_list = raw_confidence
            elif isinstance(raw_confidence, (int, float)):
                # Si c'est un seul nombre, on crée une liste remplie de ce nombre
                # pour chaque visage détecté
                conf_list = [raw_confidence] * len(faces_rects)
            else:
                # Fallback si rien n'est retourné
                conf_list = [0.0] * len(faces_rects)

            # Construction de la liste détaillée
            for i, rect in enumerate(faces_rects):
                # Sécurisation de l'accès à la confiance
                conf_val = conf_list[i] if i < len(conf_list) else 0.0

                # Conversion du rectangle en liste standard [x1, y1, x2, y2]
                # Si rect est un tuple numpy (x,y,w,h), convertissez-le si nécessaire
                if hasattr(rect, 'tolist'):
                    bbox = rect.tolist()
                else:
                    bbox = list(rect)

                faces_data.append({
                    "bbox": bbox,
                    "confidence": float(conf_val)
                })

            # Calcul d'une confiance globale (le max des confiances trouvées)
            global_confidence = max([f['confidence'] for f in faces_data]) if faces_data else 0.0

            return {
                "image": image_path,
                "has_face": len(faces_data) > 0,
                "face_count": len(faces_data),
                "confidence": global_confidence,
                "model_type": self.model_type,
                "faces": faces_data  # C'est cette liste qui servira au floutage
            }

        except ImageProcessingError:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()  # Pour voir l'erreur exacte dans les logs
            raise ImageProcessingError(f"Erreur traitement: {str(e)}")

    def analyze(self, image_path: str) -> Dict[str, Any]:
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
            return self._analyze_bytes(image_path, image_bytes)

        except PermissionError:
            raise ImageProcessingError(f"Permission refusée pour lire '{image_path}'")
        except IOError as e:
            raise ImageProcessingError(f"Erreur de lecture: {str(e)}")


class AdvancedFaceDetector:
    def __init__(self, verbose: bool = False):
        """
        Initialise le détecteur RetinaFace via InsightFace.
        """
        # Initialisation de RetinaFace
        self.app = FaceAnalysis(allowed_modules=["detection"], verbose=verbose)
        # ctx_id=-1 pour CPU, 0 pour GPU si disponible
        # det_size=(640, 640) est un bon compromis précision/vitesse
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.model_type = "RetinaFace"

    def analyze(self, image_path: str) -> Dict[str, Any]:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "image": image_path,
                "has_face": False,
                "face_count": 0,
                "confidence": 0.0,
                "model_type": self.model_type,
                "faces": []
            }
        # Détection des visages
        faces = self.app.get(img)
        faces_data = []
        max_confidence = 0.0
        for i, face in enumerate(faces):
            # InsightFace retourne bbox sous forme [x1, y1, x2, y2] en float
            bbox = face["bbox"].astype(int).tolist()
            score = float(face["det_score"])
            if score > max_confidence:
                max_confidence = score
            faces_data.append({
                "bbox": bbox,  # [x1, y1, x2, y2]
                "confidence": score
            })
        return {
            "image": image_path,
            "has_face": len(faces_data) > 0,
            "face_count": len(faces_data),
            "confidence": max_confidence,
            "model_type": self.model_type,
            "faces": faces_data
        }

    def blur_faces(self, image_path: str, filename: str = "result.jpg") -> Optional[Dict[str, Any]]:
        # Analyze image
        result_analysis = self.analyze(image_path)
        if not result_analysis.get('has_face', False):
            return None # TODO Ajouter des logs

        # Floutage de l'image
        blur_result = auto_blur_faces(str(image_path), result_analysis['faces'])
        if not blur_result.was_blurred:
            return None # TODO Ajouter des logs

        # 3. Préparation de la sauvegarde
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / date_str
        output_dir.mkdir(parents=True, exist_ok=True)
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename += ".jpg"

        final_path = output_dir / filename

        # 4. Sauvegarde
        try:
            # blur_result.image est un objet PIL.Image
            blur_result.image.save(final_path)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde : {e}")
            return None # TODO Ajouter des logs

        # 5. Retour structuré
        return {
            "real_image": str(Path(image_path).resolve()),
            "blurred_image": str(final_path.resolve()),
            "done": True,
            "faces_detected": blur_result.faces_detected
        }
