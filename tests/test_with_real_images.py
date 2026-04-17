import pytest
from src.face_detection.detector import FaceDetector, AdvancedFaceDetector
from tests.test_utils import get_all_images

TEST_SCENARIOS = [
    (
        "haar_faces",
        0,
        FaceDetector,
        0.99,
        False,
    ),  # Dossier "faces", Modèle Haar,   Seuil 90%, Flouttage False
    (
        "haar_faces",
        1,
        FaceDetector,
        0.99,
        False,
    ),  # Dossier "eyes",  Modèle Haar,   Seuil 90%, Flouttage False
    (
        "retina_faces",
        0,
        AdvancedFaceDetector,
        0.99,
        False,
    ),  # Dossier "faces", Modèle Retina, Seuil 95%, Flouttage False
    (
        "retina_eyes",
        1,
        AdvancedFaceDetector,
        0.99,
        False,
    ),  # Dossier "eyes",  Modèle Retina, Seuil 90%, Flouttage False
]


@pytest.mark.parametrize(
    "test_id, folder_type, detector_class, threshold, blur", TEST_SCENARIOS
)
class TestFaceDetectorRealImages:
    """Tests de détection sur images réelles avec différents modèles."""

    def test_detection_accuracy(
        self, test_id, folder_type, detector_class, threshold, blur
    ):
        # 1. Récupérer les images selon le type de dossier
        image_files = get_all_images(folder_type)

        # Sécurité : si aucune image, on skip
        if not image_files:
            pytest.skip(f"Aucune image trouvée pour le scénario '{test_id}'")

        # 2. Initialiser le détecteur
        if detector_class == FaceDetector:
            detector = detector_class(model_type="haar")
        else:
            detector = detector_class()

        # 3. Boucle d'analyse
        nb_detected = 0
        errors = []
        failed_images = []

        for image_path in image_files:
            try:
                result = detector.analyze(str(image_path))
                if blur:
                    blurred_name = image_path.name.replace("image", "blurred_image")
                    blurred_image = detector.blur_faces(image_path, blurred_name)
                if result["has_face"]:
                    nb_detected += 1
                else:
                    # 👇 Si pas de visage détecté, on ajoute le nom du fichier à la liste
                    failed_images.append(str(image_path))

            except Exception as e:
                errors.append(f"{image_path}: {str(e)}")
                # On considère aussi une erreur comme un échec de détection
                failed_images.append(f"{image_path} (Erreur: {e})")

        # 4. Calcul du taux de réussite
        total_images = len(image_files)
        accuracy = nb_detected / total_images

        # Calcul du nombre d'échecs
        nb_failed = len(failed_images)

        # 5. Assertion dynamique avec détails complets
        assert accuracy >= threshold, (
            f"\n❌ ÉCHEC DU TEST [{test_id}]\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 Statistiques:\n"
            f"   • Taux de détection: {accuracy:.2%}\n"
            f"   • Seuil requis:      {threshold:.2%}\n"
            f"   • Images détectées:  {nb_detected} / {total_images}\n"
            f"   • Images ratées:     {nb_failed} / {total_images}\n"
            f"\n"
            f"🚫 Images non validées ({nb_failed}):\n"
            + "\n".join([f"   - {img}" for img in failed_images])
            + f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
