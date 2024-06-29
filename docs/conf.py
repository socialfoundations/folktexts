# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'folktexts'
copyright = '2024, Social Foundations of Computation, at MPI-IS'
author = 'AndreFCruz'

# Import package version programmatically
import sys
from pathlib import Path

sys.path.insert(0, Path(__file__).parent.as_posix())
from folktexts._version import __version__

release = __version__
version = __version__


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx_copybutton',
    # 'sphinx_autopackagesummary',
    'myst_parser',
    'sphinx.ext.viewcode',
    'sphinxemoji.sphinxemoji',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
