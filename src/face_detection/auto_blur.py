from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

from PIL import Image, ImageFilter

from src.face_detection.types import FaceBox

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BlurConfig:
    blur_radius: int = 15  # Rayon du flou gaussien
    bbox_padding: int = 10  # Marge autour de la bbox (px)
    min_face_size: int = 20  # Ignorer les détections trop petites (bruit)
    preserve_original_if_empty: bool = True  # Retourner l'originale si aucun visage


@dataclass(frozen=True)
class BlurResult:
    image: Image.Image
    faces_detected: int
    was_blurred: bool
    bounding_boxes: List[Tuple[int, int, int, int]]


def auto_blur_faces(
    image: Union[str, Path, Image.Image],
    detections: List[FaceBox],
    config: BlurConfig | None = None,
) -> BlurResult:
    cfg = config or BlurConfig()

    # Charger l'image si chemin fourni
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    else:
        img = image.copy()

    # Filtrer les détections invalides ou trop petites
    valid_detections = []
    for det in detections:
        bbox = det.get("bbox")
        if not bbox or len(bbox) != 4:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        width = x2 - x1
        height = y2 - y1

        if width < cfg.min_face_size or height < cfg.min_face_size:
            continue

        valid_detections.append((x1, y1, x2, y2))

    # Aucun visage valide → retour direct
    if not valid_detections:
        return BlurResult(
            image=img, faces_detected=0, was_blurred=False, bounding_boxes=[]
        )

    # Appliquer le flou sur chaque bbox
    img_w, img_h = img.size
    for x1, y1, x2, y2 in valid_detections:
        # Appliquer padding et clamp aux bords de l'image
        x1 = max(0, x1 - cfg.bbox_padding)
        y1 = max(0, y1 - cfg.bbox_padding)
        x2 = min(img_w, x2 + cfg.bbox_padding)
        y2 = min(img_h, y2 + cfg.bbox_padding)

        # Ignorer si région invalide après padding
        if x2 <= x1 or y2 <= y1:
            continue

        # Crop → Blur → Paste (méthode PIL optimisée)
        region = img.crop((x1, y1, x2, y2))
        blurred_region = region.filter(ImageFilter.GaussianBlur(radius=cfg.blur_radius))
        img.paste(blurred_region, (x1, y1))

    logger.debug(
        "Auto-blur applied: %d faces processed, radius=%d, padding=%d",
        len(valid_detections),
        cfg.blur_radius,
        cfg.bbox_padding,
    )

    return BlurResult(
        image=img,
        faces_detected=len(valid_detections),
        was_blurred=True,
        bounding_boxes=valid_detections,
    )
