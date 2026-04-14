============
Installation
============

Prérequis
=========

- Python 3.9 ou supérieur
- pip >= 21.0

Installation basique
====================

Installation depuis PyPI::

   pip install face-detection


Installation de développement
=============================

Pour contribuer au projet::

   git clone https://github.com/Saadiinho/face-detection.git
   cd face-detection
   pip install -e ".[dev]"

Vérification
============

Vérifie que l'installation fonctionne::

   python -c "from face_detection import FaceDetector; print('✅ OK')"

Dependences du système
======================

Pour OpenCV, certaines librairies système peuvent être requises:

.. tab-set::

   .. tab-item:: Ubuntu/Debian

      .. code-block:: bash

         sudo apt-get install -y libgl1-mesa-glx libglib2.0-0

   .. tab-item:: macOS

      .. code-block:: bash

         brew install opencv

   .. tab-item:: Windows

      Aucune dépendance supplémentaire requise