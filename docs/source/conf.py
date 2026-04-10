# docs/source/conf.py
# Configuration file for the Sphinx documentation builder.

import os
import sys
from datetime import datetime

# ─────────────────────────────────────────────────────────────
# Path Setup
# ─────────────────────────────────────────────────────────────
# Ajouter le dossier src au PATH pour que Sphinx trouve ton module
sys.path.insert(0, os.path.abspath("../../src"))

# ─────────────────────────────────────────────────────────────
# Project Information
# ─────────────────────────────────────────────────────────────
project = "Niyya Face Detector"
copyright = f"{datetime.now().year}, Niyya Team"
author = "Niyya Team"
release = "0.1.0"  # Sera mis à jour automatiquement via git tags

# ─────────────────────────────────────────────────────────────
# Extensions
# ─────────────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",  # Génération auto depuis docstrings
    "sphinx.ext.viewcode",  # Liens vers le code source
    "sphinx.ext.napoleon",  # Support Google/NumPy docstrings
    "sphinx.ext.intersphinx",  # Liens vers docs externes (Python, etc.)
    "sphinx.ext.coverage",  # Rapport de couverture de docs
    "sphinx_autodoc_typehints",  # Affichage des type hints
    "sphinx_copybutton",  # Bouton copy pour les blocs de code
    "sphinx_design",  # Composants UI (cards, tabs, etc.)
    "myst_parser",  # Support Markdown
]

# ─────────────────────────────────────────────────────────────
# Theme Configuration
# ─────────────────────────────────────────────────────────────
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
    "display_version": True,
    "logo_only": False,
    # 'logo': {
    #     'text': 'Niyya Face Detector',
    #     'image_light': '_static/logo-light.png',
    #     'image_dark': '_static/logo-dark.png',
    # }
}

# ─────────────────────────────────────────────────────────────
# HTML Options
# ─────────────────────────────────────────────────────────────
html_static_path = ["_static"]
html_css_files = ["custom.css"]  # CSS personnalisé (optionnel)

# ─────────────────────────────────────────────────────────────
# Autodoc Options
# ─────────────────────────────────────────────────────────────
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

autodoc_typehints = "description"
autodoc_typehints_description_target = "documented"

# ─────────────────────────────────────────────────────────────
# Intersphinx (Liens vers docs externes)
# ─────────────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "opencv": ("https://docs.opencv.org/4.x/", None),
}

# ─────────────────────────────────────────────────────────────
# Other Options
# ─────────────────────────────────────────────────────────────
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ─────────────────────────────────────────────────────────────
# Napoleon (Google/NumPy style docstrings)
# ─────────────────────────────────────────────────────────────
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
