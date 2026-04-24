from dataclasses import dataclass, field, asdict
from typing import List, Literal, Optional
import json

# --- Types Intermédiaires ---


@dataclass(frozen=True)
class FaceBox:
    bbox: List[int]
    confidence: float
    source: str = "unknown"

    def __post_init__(self):
        """Validation basique des données à la création."""
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain exactly 4 integers [x1, y1, x2, y2]")
        if not all(isinstance(x, int) for x in self.bbox):
            # Conversion automatique si nécessaire (ex: numpy int64)
            object.__setattr__(self, "bbox", [int(x) for x in self.bbox])

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")

    def to_dict(self) -> dict:
        """Convertit l'objet en dictionnaire standard (pour JSON)."""
        return asdict(self)


# --- Résultat Principal ---


@dataclass(frozen=True)
class DetectionResult:
    image_path: str
    has_face: bool
    face_count: int
    confidence: float
    model_type: str
    faces: List[FaceBox] = field(default_factory=list)

    def to_dict(self) -> dict:
        data = asdict(self)
        # Nettoyage optionnel si besoin
        return data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass(frozen=True)
class BlurDataResult:
    real_image: str
    blurred_image: str
    done: bool
    faces_detected: int
    detection_method: str
