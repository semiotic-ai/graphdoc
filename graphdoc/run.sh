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

# help menu
show_help() {
    echo "Usage: ./nli [option]"
    echo "Options:"
    
    # development and installation commands
    echo "  python                 Run Python"
    echo "  shell                  Run shell"
    echo "  install                Install dependencies"
    echo "  dev                    Install dependencies with dev"
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
        *) show_help ;;
    esac
fi