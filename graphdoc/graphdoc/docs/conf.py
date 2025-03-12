# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Extension configuration -------------------------------------------------

# Set the primary domain to Python to avoid duplicate object descriptions
primary_domain = "py"

# Add autodoc settings to prevent duplicate warnings
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# Properly handle duplicates
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}


# Skip dspy.Signature classes which are causing warnings
def skip_dspy_signatures(app, what, name, obj, skip, options):
    import inspect

    if inspect.isclass(obj):
        # Skip classes that inherit from dspy.Signature
        for base in obj.__mro__:
            if base.__name__ == "Signature" and base.__module__.startswith("dspy"):
                return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", skip_dspy_signatures)
