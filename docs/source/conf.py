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

project = "SiPhotonics Toolkit"
copyright = "2023, Photonic Architecture Laboratories"
author = "A. D. Vit, K. Gorgulu, A. N. Amiri, E. S. Magden"
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

# texescape.escape('asdasd\\and', 'lualatex')

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]

html_theme_options = {
    "repository_url": "https://https://github.com/Photonic-Architecture-Laboratories/si-photonics-toolkit",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
}

pdf_documents = [
    ("index", "rst2pdf", "Sample rst2pdf doc", "Aycan Deniz Vit, Kazim Gorgulu, Ali Najjar Amiri, Emir Salih Magden"),
]
pdf_stylesheets = ["sphinx", "kerning", "a4"]
pdf_break_level = 0
pdf_inline_footnotes = True
pdf_fit_mode = "shrink"
# wrap text doesnt fit in the page width and breaks the pdf build process
pdf_breakside = "any"
pdf_compressed = True
pdf_use_index = True
pdf_theme = "default"


source_suffix = {
    ".rst": "restructuredtext",
    ".txt": "markdown",
    ".md": "markdown",
}

nbsphinx_allow_errors = True
