# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'ZarrDataset'
copyright = '2023, The Jackson Laboratory'
author = 'Fernando Cervantes Sanchez (The Jackson Laboratory)'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "autoapi.extension",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode"
]

autoapi_dirs = ["../../zarrdataset"]
templates_path = ['_templates']
exclude_patterns = []

# -- Options for myst_nb -------------------------------------------------
nb_execution_mode = "cache"
nb_execution_cache_path = "docs/build/.jupyter_cache"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
