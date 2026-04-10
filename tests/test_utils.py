from pathlib import Path
from typing import List

import pytest


def get_all_images() -> List[Path]:
    test_file_dir = Path(__file__).parent
    images_folder = test_file_dir / "images"
    if not images_folder.exists():
        pytest.skip(f"{images_folder} n'existe pas")
    valid_extensions = ["jpg", "jpeg", "png", "webp"]
    image_file = [
        f
        for f in images_folder.iterdir()
        if f.is_file() and f.suffix.lower() in valid_extensions
    ]
    return image_file
