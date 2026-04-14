=====
Usage
=====

Usage Basique
=============

.. code-block:: python

   from face_detection import FaceDetector

   # Initialisation du détecteur
   detector = FaceDetector(model_type="haar")

   # Analyse depuis un chemin de fichier
   result = detector.analyze("photo.jpg")

   print(f"Visage détecté: {result['has_face']}")
   print(f"Nombre de visages: {result['face_count']}")

Modèles disponibles
===================

.. list-table:: Modèles de Détection
   :widths: 25 25 25 25
   :header-rows: 1

   * - Modèle
     - Précision
     - Vitesse
     - Usage Recommandé
   * - ``haar``
     - ⭐⭐
     - ⚡⚡
     - Tests rapides, prototype
   * - ``dnn`` (en cours d'implémentation)
     - ⭐⭐⭐⭐
     - ⚡⚡
     - Production légère
   * - ``retinaface``
     - ⭐⭐⭐⭐⭐
     - ⚡⚡
     - Partiellement caché, occlusions, production

Usage Avancé
============

Analyse depuis des bytes (pour API)::

   with open("photo.jpg", "rb") as f:
       image_bytes = f.read()

   result = detector.analyze_bytes(image_bytes)

Gestion des Erreurs
===================

.. code-block:: python

   from face_detection import FaceDetector
   from face_detection.exceptions import ImageProcessingError, ModelLoadingError

   detector = FaceDetector(model_type="haar")

   try:
       result = detector.analyze("photo.jpg")
   except FileNotFoundError:
       print("Fichier non trouvé")
   except ImageProcessingError as e:
       print(f"Erreur de traitement: {e}")
   except ModelLoadingError as e:
       print(f"Erreur de modèle: {e}")