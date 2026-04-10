==============
Face Detection
==============


| 

**Module de détection faciale pour la modération de contenu**

Face Detection est un package Python modulaire et extensible pour la
détection de visages dans des images. Il supporte plusieurs modèles de 
détection (Haar Cascades, DNN, RetinaFace) via une interface unifiée.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Installation
      :link: installation
      :link-type: doc

      Guide d'installation et configuration

   .. grid-item-card:: Usage
      :link: usage
      :link-type: doc

      Exemples et tutoriels d'utilisation

   .. grid-item-card:: Contributing
      :link: contributing
      :link-type: doc

      Comment contribuer au projet

Quick Start
===========

.. code-block:: python

   from face_detection import FaceDetector

   # Initialisation
   detector = FaceDetector(model_type="retinaface") # Dans la prochaine version

   # Analyse d'une image
   result = detector.analyze("photo.jpg")
   print(f"Visage détecté: {result['has_face']}")

.. hint::

   Pour une détection optimale sur des visages partiellement caché,
   utilisez le modèle ``retinaface``.

Table des matières
==================

.. toctree::
   :maxdepth: 2
   :caption: Sommaire:

   installation
   usage
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`