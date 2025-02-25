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

mlflow_teardown() {
    echo "Teardown mlflow-manager..."
    cd mlflow-manager && ./run.sh make-clean
    cd ..
}

# train commands
doc_quality_train() {
    echo "Training a document quality model..."
    cd graphdoc && ./run.sh dev && ./run.sh doc-quality-train
    cd ..
}

build_and_run_doc_quality_trainer() {
    mlflow_setup
    doc_quality_train
}

# help command
show_help() {
    echo "Usage: ./nli [option]"
    echo "Options:"
    echo "  dev                             Install all package dependencies in development mode"
    echo "  install                         Install all package dependencies in production mode"

    # make commands
    echo "  mlflow-setup                    Install mlflow-manager dependencies and run the services"
    echo "  mlflow-teardown                 Teardown mlflow-manager services"

    # train commands
    echo "  doc-quality-train               Train a document quality model"

    # build and run commands
    echo "  build-and-run-doc-quality-train Build and run the document quality trainer"
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
        "mlflow-teardown") mlflow_teardown ;;

        # train commands
        "doc-quality-train") doc_quality_train ;;

        # build and run commands
        "build-and-run-doc-quality-train") build_and_run_doc_quality_trainer ;;
        *) show_help ;;
    esac
fi