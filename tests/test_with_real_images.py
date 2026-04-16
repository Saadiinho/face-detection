import pytest
from src.face_detection.detector import FaceDetector, AdvancedFaceDetector
from tests.test_utils import get_all_images

TEST_SCENARIOS = [
    ("haar_faces", 0, FaceDetector, 0.90),            # Dossier "faces", Modèle Haar,   Seuil 90%
    ("retina_faces", 0, AdvancedFaceDetector, 0.95),  # Dossier "faces", Modèle Retina, Seuil 95%
    ("retina_eyes", 1, AdvancedFaceDetector, 0.90),   # Dossier "eyes",  Modèle Retina, Seuil 90%
]


@pytest.mark.parametrize("test_id, folder_type, detector_class, threshold", TEST_SCENARIOS)
class TestFaceDetectorRealImages:
    """Tests de détection sur images réelles avec différents modèles."""
    def test_detection_accuracy(self, test_id, folder_type, detector_class, threshold):

        # 1. Récupérer les images selon le type de dossier
        image_files = get_all_images(folder_type)

        # Sécurité : si aucune image, on skip (géré aussi dans get_all_images mais double vérif)
        if not image_files:
            pytest.skip(f"Aucune image trouvée pour le scénario '{test_id}'")

        # 2. Initialiser le détecteur
        # Note: Pour AdvancedFaceDetector, pas d'args. Pour FaceDetector, on pourrait passer model_type
        if detector_class == FaceDetector:
            # On suppose que le défaut est 'haar' ou on le force ici si besoin
            detector = detector_class(model_type="haar")
        else:
            detector = detector_class()

        # 3. Boucle d'analyse
        nb_detected = 0
        errors = []

        for image_path in image_files:
            try:
                result = detector.analyze(str(image_path))
                if result["has_face"]:
                    nb_detected += 1
            except Exception as e:
                errors.append(f"{image_path.name}: {str(e)}")

        # 4. Calcul du taux de réussite
        total_images = len(image_files)
        accuracy = nb_detected / total_images

        # 5. Assertion dynamique
        assert accuracy >= threshold, (
            f"Échec du test [{test_id}]: "
            f"Taux de détection {accuracy:.2%} < seuil requis {threshold:.2%}. "
            f"({nb_detected}/{total_images} images détectées). "
            f"Erreurs: {errors}"
        )