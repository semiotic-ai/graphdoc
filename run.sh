#!/bin/bash

# ensure that the scripts are executable
chmod +x ./graphdoc/run.sh
chmod +x ./graphdoc-server/run.sh

# setup commands 
dev_install() {
    echo "Installing all package dependencies..."
    
    echo "Setting up graphdoc..."
    cd graphdoc && ./run.sh dev
    cd ..
    
    echo "Setting up mlflow-manager..."
    cd mlflow-manager && ./run.sh dev
    cd ..
}

# make command
mlflow_setup() {
    echo "Setting up mlflow-manager..."
    cd mlflow-manager && ./run.sh install && ./run.sh make-install
    cd ..
}

install() {
    echo "Installing all package dependencies..."
    
    echo "Setting up graphdoc..."
    cd graphdoc && ./run.sh install
    cd ..
    
    echo "Setting up mlflow-manager..."
    cd mlflow-manager && ./run.sh install
    cd ..
}

# help command
show_help() {
    echo "Usage: ./nli [option]"
    echo "Options:"
    echo "  dev                    Install all package dependencies in development mode"
    echo "  install                Install all package dependencies in production mode"

    # make commands
    echo "  mlflow-setup           Install mlflow-manager dependencies and run the services"
}

# run the script
if [ -z "$1" ]; then
    show_help
else
    case "$1" in
        "dev") dev_install ;;
        "install") install ;;

        # make commands
        "mlflow-setup") mlflow_setup ;;
        *) show_help ;;
    esac
fi