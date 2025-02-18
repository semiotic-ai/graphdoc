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

# make command
make_install_command() {
    make install
}

make_test_command() {
    make test
}

make_clean_command() {
    make clean
}

# runs install, test, and clean
make_all_command() {
    make all
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

    # make commands
    echo "  make-install           Install dependencies"
    echo "  make-test              Run tests"
    echo "  make-clean             Clean the environment"
    echo "  make-all               Install, test, and clean"
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

        # make commands
        "make-install") make_install_command ;;
        "make-test") make_test_command ;;
        "make-clean") make_clean_command ;;
        "make-all") make_all_command ;;
        *) show_help ;;
    esac
fi