"""Blur faces in one user-selected image for a quick visual check.

Usage from the repository root::

    python tests/blur_image.py
    python tests/blur_image.py /path/to/photo.jpg
    python tests/blur_image.py /path/to/photo.jpg --output /tmp/blurred.jpg
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this file directly without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from face_detection.detector import AdvancedFaceDetector, FaceDetector  # noqa: E402

# Change this path to select the image when running the script without arguments.
IMAGE_PATH = Path("/home/saadr/journals_20261105_inaya_22_IMG_3273.webp")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Détecte et floute les visages d’une image pour inspection."
    )
    parser.add_argument(
        "image",
        type=Path,
        nargs="?",
        default=IMAGE_PATH,
        help="Chemin de l’image à traiter (par défaut : IMAGE_PATH dans le fichier)",
    )
    parser.add_argument(
        "--model",
        choices=("haar", "dnn", "retinaface"),
        default="retinaface",
        help="Détecteur à utiliser (défaut : retinaface, le plus performant)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Chemin de sortie (défaut : results/blurred_<nom de l’image>.jpg)",
    )
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image introuvable : {args.image}")

    detector = (
        AdvancedFaceDetector()
        if args.model == "retinaface"
        else FaceDetector(model_type=args.model)
    )
    output = args.output or Path("results") / f"blurred_{args.image.stem}.jpg"

    try:
        result = detector.blur_faces(str(args.image), str(output))
    except Exception as exc:
        print(f"Erreur pendant le traitement : {exc}", file=sys.stderr)
        return 1

    if result is None:
        print("Aucun visage détecté : aucune image floutée n’a été créée.")
        return 2

    print(f"Visages floutés : {result.faces_detected}")
    print(f"Image enregistrée : {result.blurred_image}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
