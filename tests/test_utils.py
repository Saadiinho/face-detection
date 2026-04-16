import argparse
import os
from pathlib import Path
from typing import List

import pytest

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def get_all_images(type_folder: int = 0) -> List[Path]:
    test_file_dir = Path(__file__).parent
    folder = ""
    if type_folder == 0:
        folder = "faces"
    elif type_folder == 1:
        folder = "only-eyes"
    images_folder = os.path.join(test_file_dir, "images", folder)
    if not os.path.exists(images_folder):
        pytest.skip(f"{images_folder} n'existe pas")
    valid_extensions = "jpg"
    image_files = []
    for image in os.listdir(images_folder):
        if image.endswith(valid_extensions):
            image_files.append(os.path.join(images_folder, image))

    return image_files


def get_image_files(folder: Path) -> List[Path]:
    """"""
    if not folder.exists():
        raise FileNotFoundError(f"Le dossier '{folder}' n'existe pas")

    if not folder.is_dir():
        raise NotADirectoryError(f"'{folder}' n'est pas un dossier")

    # Filtrer les fichiers par extension
    image_files = [
        f
        for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    # Trier par nom pour un ordre cohérent
    return sorted(image_files, key=lambda x: x.name)


def rename_images(
    folder: Path, prefix: str, start_index: int = 1, dry_run: bool = False
) -> None:
    """
    """
    # Récupération des fichiers
    image_files = get_image_files(folder)

    if not image_files:
        print(f"⚠Aucune image trouvée dans '{folder}'")
        print(f"   Extensions supportées: {SUPPORTED_EXTENSIONS}")
        return

    print(f"Dossier: {folder}")
    print(f"{len(image_files)} image(s) trouvée(s)")
    print("-" * 60)

    # Compteur pour la numérotation
    counter = start_index

    # Suivi des opérations
    renamed_count = 0
    skipped_count = 0

    for image_path in image_files:
        # Déterminer l'extension (toujours .jpg pour uniformité)
        new_extension = ".jpg"
        new_name = f"{prefix}_{counter}{new_extension}"
        new_path = folder / new_name

        # Vérifier si le fichier a déjà le bon nom
        if image_path.name == new_name:
            print(f"SKIP: {image_path.name} (déjà correct)")
            skipped_count += 1
            counter += 1
            continue

        # Vérifier si le nouveau nom existe déjà (conflit)
        if new_path.exists() and new_path != image_path:
            print(f"⚠CONFLIT: {new_name} existe déjà, skipping...")
            skipped_count += 1
            counter += 1
            continue

        # Affichage de l'opération
        if dry_run:
            print(f"[DRY-RUN] {image_path.name} → {new_name}")
        else:
            try:
                image_path.rename(new_path)
                print(f"ENAMED: {image_path.name} → {new_name}")
                renamed_count += 1
            except Exception as e:
                print(f"ERROR: {image_path.name} → {e}")

        counter += 1

    # Résumé
    print("-" * 60)
    print(f"RÉSUMÉ:")
    print(f"   • Images renommées: {renamed_count}")
    print(f"   • Images ignorées: {skipped_count}")
    print(f"   • Total traité: {len(image_files)}")

    if dry_run:
        print(f"\nMODE DRY-RUN: Aucun fichier n'a été modifié")
        print(f"   Lance sans --dry-run pour appliquer les changements")


def main():
    """Point d'entrée principal du script."""
    parser = argparse.ArgumentParser(
        description="Renomme les images de test pour Niyya Face Detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples:
    python rename_test_images.py --folder tests/fixtures/with_faces --type faces
    python rename_test_images.py --folder tests/fixtures/without_faces --type nofaces
    python rename_test_images.py --folder ./images --type faces --start 10
    python rename_test_images.py --folder ./images --type faces --dry-run
        """,
    )

    parser.add_argument(
        "--folder",
        "-f",
        required=True,
        type=Path,
        help="Dossier contenant les images à renommer",
    )

    parser.add_argument(
        "--type",
        "-t",
        required=True,
        choices=["faces", "nofaces", "eyes"],
        help="Type d'images: 'faces' (avec visage) ou 'nofaces' (sans visage) ou 'eyes' (seulement les yeux)",
    )

    parser.add_argument(
        "--start",
        "-s",
        type=int,
        default=1,
        help="Index de départ pour la numérotation (défaut: 1)",
    )

    parser.add_argument(
        "--dry-run",
        "-d",
        action="store_true",
        help="Affiche les changements sans les appliquer (mode test)",
    )

    args = parser.parse_args()

    # Détermination du préfixe selon le type
    if args.type == "faces":
        prefix = "image_with_faces"
    elif args.type == "nofaces":
        prefix = "image_without_faces"
    elif args.type == "eyes":
        prefix = "image_only_eyes"
    else:
        prefix = "image"

    print("=" * 60)
    print("🎯 NIYYA FACE DETECTOR - Script de renommage d'images")
    print("=" * 60)

    try:
        rename_images(
            folder=args.folder,
            prefix=prefix,
            start_index=args.start,
            dry_run=args.dry_run,
        )
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"❌ Erreur: {e}")
        exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Opération annulée par l'utilisateur")
        exit(0)


if __name__ == "__main__":
    main()
