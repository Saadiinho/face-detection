#!/usr/bin/env python3
"""
Script de test manuel pour le module de détection faciale.

Usage:
    python test_manual.py --image /chemin/vers/ta/photo.jpg
"""

import argparse
from pathlib import Path
from src.face_detection.detector import FaceDetector, AdvancedFaceDetector
from src.face_detection.exceptions import ImageProcessingError


def test_image(image_path: str, model_type: str = "haar"):
    """
    Teste une image avec le détecteur de visages.

    Args:
        image_path: Chemin vers l'image à tester
        model_type: Type de modèle ("haar" ou "dnn")
    """
    # Vérification que le fichier existe
    image_file = Path(image_path)
    if not image_file.exists():
        print(f"❌ Erreur: Le fichier '{image_path}' n'existe pas.")
        return

    # Lecture de l'image
    try:
        with open(image_file, 'rb') as f:
            image_bytes = f.read()
    except Exception as e:
        print(f"❌ Erreur lecture fichier: {e}")
        return

    # Initialisation du détecteur
    print(f"🔍 Initialisation du détecteur ({model_type})...")
    detector = FaceDetector(model_type=model_type)

    # Analyse
    print(f"📸 Analyse de l'image: {image_file.name}")
    print("-" * 50)

    try:
        result = detector.analyze(image_bytes)

        # Affichage des résultats
        print(f"✅ Analyse terminée !")
        print(f"   • Visage détecté: {'OUI' if result['has_face'] else 'NON'}")
        print(f"   • Nombre de visages: {result['face_count']}")
        print(f"   • Confiance moyenne: {result['confidence']:.2%}")
        print(f"   • Modèle utilisé: {result['model_type']}")



    except ImageProcessingError as e:
        print(f"❌ Erreur de traitement: {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test de détection faciale pour Niyya Women"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Chemin vers l'image à tester"
    )
    parser.add_argument(
        "--model", "-m",
        choices=["haar", "dnn"],
        default="haar",
        help="Type de modèle à utiliser (défaut: haar)"
    )

    args = parser.parse_args()
    test_image(args.image, args.model)
    detector = AdvancedFaceDetector()
    print(detector.analyze(args.image))