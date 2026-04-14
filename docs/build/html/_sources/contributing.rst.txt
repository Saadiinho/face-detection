============
Contribution
============

Merci de vouloir contribuer à Face Detection !

Setup de Développement
======================

.. code-block:: bash

   git clone https://github.com/Saadiinho/face-detection.git
   cd face-detection
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"

Code Style
==========

Nous utilisons Black pour le formatage::

   make black

LAncement des Tests
===================

.. code-block:: bash

   make test           # Tests unitaires

Génération de la Documentation
==============================

.. code-block:: bash

   cd docs
   make html
   open _build/html/index.html  # macOS
   xdg-open _build/html/index.html  # Linux

Pull Request Process
====================

1. Fork le repository
2. Crée une branche feature (``git checkout -b feature/amazing-feature``)
3. Commit tes changements (``git commit -m 'Add amazing feature'``)
4. Push vers la branche (``git push origin feature/amazing-feature``)
5. Ouvre une Pull Request

Code of Conduct
===============

Soyez respectueux et inclusifs dans toutes vos interactions.