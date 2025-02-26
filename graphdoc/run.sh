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

lint_command() {
    poetry run pyright .
}

test_command() {
    poetry run pytest --testmon -p no:warnings
}

commit_command() {
    format_command
    lint_command
    test_command
    requirements_command
}

# Documentation commands
docs() {
    echo "Building documentation..."
    cd docs && make clean html
    echo "Documentation built in docs/build/html"
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
    echo "  lint                   Lint the code"
    echo "  test                   Run the tests"
    echo "  commit                 Format, lint, and test the code"
    echo "  docs                   Build the documentation"
    echo "  docs-init              Initialize the Sphinx documentation"

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
        "lint") lint_command ;;
        "test") test_command ;;
        "commit") commit_command ;;
        "docs") docs ;;
        "docs-init") docs_init ;;
        "doc-quality-train") doc_quality_train_command ;;
        "doc-generator-train") doc_generator_train_command ;;
        "doc-generator-eval") doc_generator_eval_command ;;
        *)
            echo "Usage: $0 {test|lint|format|docs|docs-init|doc-quality-train|doc-generator-train|doc-generator-eval}"
            exit 1
            ;;
    esac
fi