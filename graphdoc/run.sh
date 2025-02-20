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

format_command() {
    poetry run black .
}

lint_command() {
    poetry run pyright .
}

test_command() {
    poetry run pytest -p no:warnings
}

commit_command() {
    format_command
    lint_command
    test_command
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
    echo "  format                 Format the code"
    echo "  lint                   Lint the code"
    echo "  test                   Run the tests"
    echo "  commit                 Format, lint, and test the code"
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
        "format") format_command ;;
        "lint") lint_command ;;
        "test") test_command ;;
        "commit") commit_command ;;
        *) show_help ;;
    esac
fi