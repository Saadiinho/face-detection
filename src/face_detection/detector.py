from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
from insightface.app import FaceAnalysis

from .auto_blur import auto_blur_faces
from .exceptions import ImageProcessingError, ModelLoadingError
from .types import DetectionResult


class FaceDetector:
    def __init__(
        self,
        model_type: str = "haar",
        confidence_threshold: float = 0.5,
        model_path: Optional[Path] = None,
        use_eye_fallback: bool = True,  # Nouveau paramètre
    ):
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self._model_path = model_path
        self.use_eye_fallback = use_eye_fallback

        if model_type == "haar":
            self._init_haar_model()
        elif model_type == "dnn":
            self._init_dnn_model()
        else:
            raise ValueError(f"Type de modèle non supporté: {model_type}")

        # Initialisation des détecteurs d'yeux (léger, toujours chargé si fallback activé)
        if self.use_eye_fallback:
            self._init_eye_models()

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

    def _init_eye_models(self) -> None:
        """Charge les cascades pour la détection d'yeux."""
        try:
            self._eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml"
            )
            # Optionnel : Cascade pour yeux avec lunettes
            self._eye_glasses_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml"
            )
        except Exception as e:
            print(f"⚠️ Avertissement: Échec chargement détecteurs d'yeux : {e}")
            self._eye_cascade = None
            self._eye_glasses_cascade = None

    #########################
    ##       Analyse       ##
    #########################

    def _detect_eyes_fallback(
        self, img: np.ndarray
    ) -> Tuple[List[Tuple[int, int, int, int]], float]:
        """
        Détecte les yeux si aucun visage n'est trouvé.
        Retourne une liste de bbox [x, y, w, h] et une confiance fixe (0.8 pour les yeux).
        """
        if not hasattr(self, "_eye_cascade") or self._eye_cascade is None:
            return [], 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détection yeux standards
        eyes = self._eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10)
        )

        # Détection yeux avec lunettes (pour renforcer)
        eyes_glasses = []
        if self._eye_glasses_cascade:
            eyes_glasses = self._eye_glasses_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=10, minSize=(10, 10)
            )

        # Fusionner et dédupliquer (simple approche : tout ajouter)
        all_eyes = list(eyes) + list(eyes_glasses)

        # Si on trouve au moins un œil, on considère que c'est un "succès"
        # On peut aussi grouper les deux yeux en une seule bbox si souhaité,
        # mais ici on retourne chaque œil détecté comme une zone à flouter.
        confidence = 0.85 if len(all_eyes) > 0 else 0.0

        return [
            (int(x), int(y), int(w), int(h)) for (x, y, w, h) in all_eyes
        ], confidence

    def _analyze_haar(self, img: np.ndarray) -> tuple:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        confidence = 1.0 if len(faces) > 0 else 0.0
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces], confidence

    def _analyze_dnn(self, img: np.ndarray) -> tuple:
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

    def _analyze_bytes(self, image_path: str, image_content: bytes) -> Dict[str, Any]:
        try:
            nparr = np.frombuffer(image_content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ImageProcessingError("Impossible de décoder l'image")

            # 1. Tentative de détection principale (Visage)
            if self.model_type == "haar":
                rects, raw_confidence = self._analyze_haar(img)
            else:
                rects, raw_confidence = self._analyze_dnn(img)

            # 2. LOGIQUE DE ROBUSTESSE : Fallback sur les yeux
            detected_rects = rects
            final_confidence = raw_confidence
            detection_source = self.model_type

            if len(rects) == 0 and self.use_eye_fallback:
                # Aucun visage trouvé, on tente les yeux
                eye_rects, eye_conf = self._detect_eyes_fallback(img)
                if len(eye_rects) > 0:
                    detected_rects = eye_rects
                    final_confidence = eye_conf
                    detection_source = "eyes_fallback"
                    print(
                        f"ℹ️ Repli sur détection d'yeux pour {image_path} ({len(eye_rects)} yeux trouvés)"
                    )

            # 3. Formatage des résultats
            faces_data = []
            if isinstance(final_confidence, list):
                conf_list = final_confidence
            elif isinstance(final_confidence, (int, float)):
                conf_list = [final_confidence] * len(detected_rects)
            else:
                conf_list = [0.0] * len(detected_rects)

            for i, rect in enumerate(detected_rects):
                conf_val = conf_list[i] if i < len(conf_list) else 0.0
                if hasattr(rect, "tolist"):
                    bbox = rect.tolist()
                else:
                    bbox = list(rect)

                faces_data.append(
                    {
                        "bbox": bbox,
                        "confidence": float(conf_val),
                        "source": detection_source,  # Utile pour le debug
                    }
                )

            global_confidence = (
                max([f["confidence"] for f in faces_data]) if faces_data else 0.0
            )

            return {
                "image": image_path,
                "has_face": len(faces_data)
                > 0,  # On garde has_face=True même si ce sont des yeux
                "face_count": len(faces_data),
                "confidence": global_confidence,
                "model_type": detection_source,  # Indique si c'est "haar", "dnn" ou "eyes_fallback"
                "faces": faces_data,
            }

        except ImageProcessingError:
            raise
        except Exception as e:
            import traceback

            traceback.print_exc()
            raise ImageProcessingError(f"Erreur traitement: {str(e)}")

    # ... (Les méthodes analyze() et blur_faces() restent identiques, elles utilisent le résultat formaté) ...

    def analyze(self, image_path: str) -> Dict[str, Any]:
        # (Code inchangé par rapport à ta version précédente)
        from pathlib import Path

        image_file = Path(image_path)
        if not image_file.exists():
            raise FileNotFoundError(f"Le fichier '{image_path}' n'existe pas")
        if not image_file.is_file():
            raise ImageProcessingError(f"'{image_path}' n'est pas un fichier")
        try:
            with open(image_file, "rb") as f:
                image_bytes = f.read()
            if not image_bytes:
                raise ImageProcessingError(f"Le fichier '{image_path}' est vide")
            return self._analyze_bytes(image_path, image_bytes)
        except PermissionError:
            raise ImageProcessingError(f"Permission refusée pour lire '{image_path}'")
        except IOError as e:
            raise ImageProcessingError(f"Erreur de lecture: {str(e)}")

    def blur_faces(
        self, image_path: str, filename: str = "result.jpg"
    ) -> Optional[Dict[str, Any]]:
        # (Code inchangé, il utilisera automatiquement les yeux si le fallback a marché)
        result_analysis = self.analyze(image_path)
        if not result_analysis.get("has_face", False):
            return None

        blur_result = auto_blur_faces(str(image_path), result_analysis["faces"])
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
            print(f"Erreur lors de la sauvegarde : {e}")
            return None

        return {
            "real_image": str(image_path),
            "blurred_image": str(final_path.resolve()),
            "done": True,
            "faces_detected": blur_result.faces_detected,
            "detection_method": result_analysis["model_type"],  # Ajout info méthode
        }


class AdvancedFaceDetector:
    def __init__(self, verbose: bool = False, use_eye_fallback: bool = True):
        self.app = FaceAnalysis(allowed_modules=["detection"], verbose=verbose)
        self.app.prepare(ctx_id=-1, det_size=(640, 640))
        self.model_type = "RetinaFace"
        self.use_eye_fallback = use_eye_fallback

        # Charger Haar pour les yeux en secours
        if self.use_eye_fallback:
            try:
                self._eye_cascade = cv2.CascadeClassifier(
                    cv2.data.haarcascades + "haarcascade_eye.xml"
                )
            except:
                self._eye_cascade = None

    def _detect_eyes_fallback(self, img: np.ndarray) -> List[Dict[str, Any]]:
        if self._eye_cascade is None:
            return []

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Détection sensible
        eyes = self._eye_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10)
        )

        faces_data = []
        expansion_factor = 2.5

        for x, y, w, h in eyes:
            # Agrandissement individuel (comme avant)
            center_x = x + w / 2
            center_y = y + h / 2
            new_w = int(w * expansion_factor)
            new_h = int(h * expansion_factor)
            new_x = int(center_x - new_w / 2)
            new_y = int(center_y - new_h / 2)

            new_x = max(0, new_x)
            new_y = max(0, new_y)

            faces_data.append(
                {
                    "bbox": [new_x, new_y, new_x + new_w, new_y + new_h],
                    "confidence": 0.85,
                    "source": "eyes_fallback",
                }
            )

        # ---------------------------------------------------------
        # 🟢 NOUVEAU : FUSION DES ZONES (La clé pour avoir 1 seul bloc)
        # ---------------------------------------------------------
        if len(faces_data) > 1:
            # On trouve les coordonnées extrêmes de toutes les boîtes détectées
            min_x = min(f["bbox"][0] for f in faces_data)
            min_y = min(f["bbox"][1] for f in faces_data)
            max_x = max(f["bbox"][2] for f in faces_data)
            max_y = max(f["bbox"][3] for f in faces_data)

            # On remplace TOUTES les détections par une SEULE grande boîte
            faces_data = [
                {
                    "bbox": [min_x, min_y, max_x, max_y],
                    "confidence": 0.85,
                    "source": "eyes_fallback_merged",  # Nouveau nom pour le debug
                }
            ]
        # ---------------------------------------------------------

        return faces_data

    def analyze(self, image_path: str) -> Dict[str, Any]:
        img = cv2.imread(image_path)
        if img is None:
            return DetectionResult().to_dict()
        faces = self.app.get(img)
        faces_data = []
        max_confidence = 0.0

        for face in faces:
            bbox = face["bbox"].astype(int).tolist()
            score = float(face["det_score"])
            if score > max_confidence:
                max_confidence = score
            faces_data.append(
                {"bbox": bbox, "confidence": score, "source": "retinaface"}
            )

        # LOGIQUE DE ROBUSTESSE : Fallback
        if len(faces_data) == 0 and self.use_eye_fallback:
            eye_data = self._detect_eyes_fallback(img)
            if eye_data:
                faces_data = eye_data
                max_confidence = 0.85
                print(f"ℹ️ Repli sur détection d'yeux pour {image_path}")
        final_result = DetectionResult(
            image_path=image_path,
            has_face=len(faces_data) > 0,
            face_count=len(faces_data),
            confidence=max_confidence,
            model_type=(
                "eyes_fallback"
                if (
                    len(faces_data) > 0
                    and faces_data[0].get("source") == "eyes_fallback"
                )
                else self.model_type
            ),
            faces=faces_data,
        )
        return final_result.to_dict()

    def blur_faces(
        self, image_path: str, filename: str = "result.jpg"
    ) -> Optional[Dict[str, Any]]:
        result_analysis = self.analyze(image_path)
        if not result_analysis.get("has_face", False):
            return None

        blur_result = auto_blur_faces(str(image_path), result_analysis["faces"])
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
            print(f"Erreur lors de la sauvegarde : {e}")
            return None

        return {
            "real_image": str(image_path),
            "blurred_image": str(final_path.resolve()),
            "done": True,
            "faces_detected": blur_result.faces_detected,
            "detection_method": result_analysis["model_type"],
        }
