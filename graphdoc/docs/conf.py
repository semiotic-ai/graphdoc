# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "GraphDoc"
copyright = "2025, Semiotic Labs"
author = "Semiotic Labs"

version = "1.0"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",  # Add intersphinx for better cross-referencing
]

# Add Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# Configure autodoc settings to handle duplicate signatures
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# Configure intersphinx mapping for external projects
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# Ensure that objects are documented only once at their canonical location
canonical_module_mapping = {}


# Configure nitpicky mode to be less strict
nitpicky = False


def setup(app):
    # Create static directory if it doesn't exist to avoid the warning
    if not os.path.exists(os.path.join(os.path.dirname(__file__), "_static")):
        os.makedirs(os.path.join(os.path.dirname(__file__), "_static"))
