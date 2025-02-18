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
}

# run the script
if [ -z "$1" ]; then
    show_help
else
    case "$1" in
        "dev") dev_install ;;
        "install") install ;;
        *) show_help ;;
    esac
fi