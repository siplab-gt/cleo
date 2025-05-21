# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------

project = "Cleo"
copyright = "2024"
author = "Kyle Johnsen, Nathan Cruzado"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.duration",
    "sphinx_copybutton",
    "sphinx-favicon",
    "myst_nb",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Extension settings -------------------------------------------------------

autoclass_content = "both"
intersphinx_mapping = {
    "brian2": ("https://brian2.readthedocs.io/en/stable", None),
    "matplotlib": ("https://matplotlib.org/stable", None),
    "neo": ("https://neo.readthedocs.io/en/latest", None),
}

autosectionlabel_prefix_document = True

nb_execution_mode = "cache"
nb_execution_excludepatterns = ["tutorials/*"]
nb_execution_timeout = 120
myst_enable_extensions = [
    "dollarmath",
]

# napoleon_custom_sections = "Visualization Keyword Arguments"
napoleon_custom_sections = [
    ("Visualization kwargs", "params_style"),
    ("Injection kwargs", "params_style"),
]

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]


# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = [
    "css/custom.css",
]

html_logo = "_static/logo.svg"
# favicon_url = "_static/favicon.png"
favicons = [
    {"static-file": "favicon.svg"},  # => use `_static/favicon.svg`
    {
        "sizes": "16x16",
        "static-file": "favicon.png",
    },
]

html_theme_options = {
    # "show_nav_level": 2,
    # for Furo:
    "light_css_variables": {
        "color-brand-primary": "#C500CC",
        "color-brand-content": "#C500CC",
        "color-api-name": "#36827F",
        "color-api-pre-name": "#36827F",
        # "color-brand-primary": "#8000b4",
        # "color-brand-content": "#8000b4",
    },
    "dark_css_variables": {
        # "color-brand-primary": "#C500CC",
        # "color-brand-content": "#C500CC",
        "color-brand-primary": "#df87e1",
        "color-brand-content": "#df87e1",
        "color-api-name": "#69fff8",
        "color-api-pre-name": "#69fff8",
    },
    "sidebar_hide_name": True,
}
