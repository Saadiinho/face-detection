"""Exceptions personnalisées pour le module de détection."""


class NiyyaDetectorError(Exception):
    """Exception de base pour le détecteur."""
    pass


class ModelLoadingError(NiyyaDetectorError):
    """Erreur lors du chargement du modèle."""
    pass


class ImageProcessingError(NiyyaDetectorError):
    """Erreur lors du traitement d'image."""
    pass