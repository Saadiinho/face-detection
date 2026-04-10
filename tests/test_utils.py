import os
from pathlib import Path
from typing import List

import pytest


def get_all_images() -> List[Path]:
    test_file_dir = Path(__file__).parent
    images_folder = os.path.join(test_file_dir, "images")
    if not os.path.exists(images_folder):
        pytest.skip(f"{images_folder} n'existe pas")
    valid_extensions = "jpg"
    image_files = []
    for image in os.listdir(images_folder):
        if image.endswith(valid_extensions):
            image_files.append(os.path.join(images_folder, image))

    return image_files
