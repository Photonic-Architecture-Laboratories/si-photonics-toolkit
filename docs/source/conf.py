# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
import sipkit

sys.path.insert(0, os.path.abspath("../../sipkit"))

project = "sipkit"
copyright = "2023, Aycan Deniz Vit"
author = "Aycan Deniz Vit"
release = sipkit.__version__.version

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "myst_parser",
    "nbsphinx",
    "rst2pdf.pdfbuilder",
]


templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

pdf_documents = [
    ("index", "rst2pdf", "Sample rst2pdf doc", "Aycan Deniz Vit"),
]
pdf_stylesheets = ["sphinx", "kerning", "a4"]
pdf_break_level = 0
pdf_inline_footnotes = True
pdf_fit_mode = "shrink"
pdf_compressed = True
pdf_use_index = True
pdf_theme = "default"


source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

nbsphinx_allow_errors = True
