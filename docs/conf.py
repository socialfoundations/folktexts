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
import os
import sys

sys.path.insert(0, os.path.abspath('../'))
from folktexts._version import __version__

release = __version__
version = __version__

# Copy notebooks folder to the documentation folder
import shutil
shutil.copytree(src="../notebooks", dst="notebooks", dirs_exist_ok=True)


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme',
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',  # needs to be AFTER napoleon
    'numpydoc',
    'sphinx_copybutton',
    # 'sphinx_autopackagesummary',
    'myst_parser',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'nbsphinx',             # for rendering jupyter notebooks
    'sphinx_gallery.load_style',
    'IPython.sphinxext.ipython_console_highlighting',   # work-around for syntax-highlighting on jupyter notebooks
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autoclass_content = 'both'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_js_files = [
    'custom.js',    # custom JS file
]

# nbsphinx configuration
nbsphinx_execute = 'never'  # Set to 'always' if you want to execute the notebooks during the build process

# numpydoc configuration
numpydoc_show_class_members = False
