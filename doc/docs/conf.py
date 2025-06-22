# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys
from pathlib import Path

sys.path.insert(0, str(Path("../../src").resolve()))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PyTorch-Project-Template"
copyright = "2025, Junya Morioka"
author = "Junya Morioka"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "autoapi.extension",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- Options for HTML output -----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

# -- Options for autodoc --------------------------------------------------
autodoc_typehints = "signature"

# -- Options for viewcode --------------------------------------------------
viewcode_follow_imported_members = True

# -- Options for autoapi ---------------------------------------------------
autoapi_type = "python"
autoapi_dirs = ["../../src"]
autoapi_root = "api"
autoapi_keep_files = True
autoapi_python_class_content = "init"
autoapi_options = [
    "members",
    "undoc-members",
    "private-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
