#!/usr/bin/env python
"""
Script to automatically generate Sphinx documentation RST files.
Run this script before building the documentation to ensure all RST files are up-to-date.
"""
import os
import shutil
import subprocess
import sys


def main():
    # Get the directory where this script is located
    docs_dir = os.path.dirname(os.path.abspath(__file__))

    # The path to the module we want to document
    module_dir = os.path.abspath(os.path.join(docs_dir, ".."))

    # Where to output the rst files
    output_dir = docs_dir

    # Clean up existing RST files except for special ones
    preserve_files = ["index.rst", "conf.py", "generate_docs.py"]
    for filename in os.listdir(output_dir):
        filepath = os.path.join(output_dir, filename)
        if (
            filename.endswith(".rst")
            and filename not in preserve_files
            and os.path.isfile(filepath)
        ):
            print(f"Removing {filepath}")
            os.unlink(filepath)

    # Run sphinx-apidoc
    subprocess.run(
        [
            "sphinx-apidoc",
            "-f",  # Force overwriting of existing files
            "-e",  # Put module documentation before submodule documentation
            "-M",  # Put module documentation before member documentation
            "-o",
            output_dir,  # Output directory
            module_dir,  # Module directory
            "setup.py",  # Exclude these files/patterns
            "*tests*",
            "*venv*",
            "*docs*",
        ]
    )

    # Add custom content to the module RST files
    customize_rst_files(output_dir)

    print("\nRST files have been generated successfully!")
    print("You can now build the documentation with: cd docs && make html")


def customize_rst_files(output_dir):
    """Add custom content to the RST files."""
    # Example: Add a note about auto-generation to each RST file
    for filename in os.listdir(output_dir):
        if filename.endswith(".rst") and filename != "index.rst":
            filepath = os.path.join(output_dir, filename)
            with open(filepath, "r") as f:
                content = f.read()

            # Add noindex to submodules to prevent duplicates
            content = content.replace(
                ":show-inheritance:", ":show-inheritance:\n   :noindex:"
            )

            with open(filepath, "w") as f:
                f.write(content)

    # Create or update index.rst if it doesn't exist
    index_path = os.path.join(output_dir, "index.rst")
    if not os.path.exists(index_path):
        with open(index_path, "w") as f:
            f.write(
                """.. GraphDoc documentation master file

Welcome to GraphDoc's documentation
==================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""
            )


if __name__ == "__main__":
    main()
