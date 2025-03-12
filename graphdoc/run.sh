#!/bin/bash

# development and installation commands
python_command() {
    poetry run python
}

shell_command() {
    poetry shell
}

install_command() {
    poetry install --without dev
}

dev_command() {
    poetry install --with dev
}

requirements_command() {
    poetry export -f requirements.txt --without-hashes --with dev,docs --output requirements.txt
}

format_command() {
    poetry run black .
}

docstring_format_command() {
    poetry run docformatter --black --style sphinx --in-place --exclude="prompts" --recursive graphdoc/  
    poetry run docformatter --black --style sphinx --in-place --recursive runners/ 
    poetry run docformatter --black --style sphinx --in-place --recursive tests/ 
}

pep8_check_command() {
    poetry run flake8 graphdoc/
    poetry run flake8 runners/
    poetry run flake8 tests/
}

sort_command() {
    poetry run isort .
}

lint_command() {
    poetry run pyright .
}

test_command() {
    poetry run pytest --testmon -p no:warnings
}

commit_command() {
    format_command
    docstring_format_command
    sort_command
    lint_command
    pep8_check_command
    test_command
    requirements_command
}

# Documentation commands
docs_generate() {
    echo "Generating RST files..."
    cd docs && python generate_docs.py
    echo "RST files generated successfully!"
}

docs() {
    echo "Building documentation..."
    cd docs && make clean html
    echo "Documentation built in docs/_build/html"
}

docs_init() {
    echo "Initializing Sphinx documentation..."
    # Remove existing docs directory if it exists
    rm -rf docs
    # Create fresh docs directory
    mkdir -p docs
    cd docs
    sphinx-quickstart -q \
        -p GraphDoc \
        -a "Semiotic Labs" \
        -v 1.0 \
        -r 1.0 \
        -l en \
        --ext-autodoc \
        --ext-viewcode \
        --makefile \
        --batchfile
    # Create necessary directories
    mkdir -p source/_static source/_templates
    echo "Sphinx documentation initialized"
}

# train commands
doc_quality_train_command() {
    poetry run python runners/train/single_prompt_trainer.py --config-path assets/configs/single_prompt_doc_quality_trainer.yaml
}

doc_generator_train_command() {
    poetry run python runners/train/single_prompt_trainer.py --config-path assets/configs/single_prompt_doc_generator_trainer.yaml
}

# eval commands
doc_generator_eval_command() {
    poetry run python runners/eval/eval_doc_generator_module.py --config-path assets/configs/single_prompt_doc_generator_module_eval.yaml
}

# help menu
show_help() {
    echo "Usage: ./nli [option]"
    echo "Options:"
    
    # development and installation commands
    echo "  python                 Run Python"
    echo "  shell                  Run shell"
    echo "  install                Install dependencies"
    echo "  dev                    Install dependencies with dev"
    echo "  requirements           Generate requirements.txt"
    echo "  format                 Format the code"
    echo "  docstring-format       Format the docstrings"
    echo "  pep-check              Check the PEP8 compliance"
    echo "  lint                   Lint the code"
    echo "  test                   Run the tests"
    echo "  commit                 Format, lint, and test the code"
    echo "  docs-generate          Generate documentation RST files"
    echo "  docs                   Build the documentation"

    # train commands
    echo "  doc-quality-train      Train a document quality model"
    echo "  doc-generator-train    Train a document generator model"

    # eval commands
    echo "  doc-generator-eval     Evaluate a document generator model"
}

# handle command line arguments
if [ -z "$1" ]; then
    show_help
else
    case "$1" in

        # development and installation commands
        "python") python_command ;;
        "shell") shell_command ;;
        "install") install_command ;;
        "dev") dev_command ;;
        "requirements") requirements_command ;;
        "format") format_command ;;
        "docstring-format") docstring_format_command ;;
        "pep-check") pep8_check_command ;;
        "lint") lint_command ;;
        "test") test_command ;;
        "commit") commit_command ;;
        "docs-generate") docs_generate ;;
        "docs") docs ;;
        "doc-quality-train") doc_quality_train_command ;;
        "doc-generator-train") doc_generator_train_command ;;
        "doc-generator-eval") doc_generator_eval_command ;;
        *)
            echo "Usage: $0 {test|lint|format|docs|docs-init|doc-quality-train|doc-generator-train|doc-generator-eval}"
            exit 1
            ;;
    esac
fi