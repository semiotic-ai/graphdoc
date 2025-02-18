# mlflow-manager

The `mlflow-manager` package is a Python package for managing MLflow experiments and models. It includes the necessary components for spinning up a local MLFlow server and tracking experiments. For our purposes, we will use a `postgres` and `minio` backend for storing the metadata and artifacts. Alternatively, one could opt to utilize a series of local folders and files to store the metadata and artifacts.

## Development

We utilize [poetry](https://python-poetry.org/) for dependency management. Please run `poetry install` to install the dependencies. You can also run `poetry shell` to activate the virtual environment. Please see the [poetry documentation](https://python-poetry.org/docs/) for more information.

### run.sh

The `run.sh` script is a convenience script for development. It provides a few shortcuts for running useful commands.

```bash 
# ensure that the script is executable
chmod +x run.sh

# install dependencies (including dev dependencies)
./run.sh dev # use `./run.sh install` to install dependencies excluding dev dependencies

# set up the environment variables for the docker compose file
cp docker/.env.example docker/.env

# build, start, test, and clean the services
./run.sh make-all

# build and run the image
./run.sh make-install
```

## Docker 

Thanks to `violincoding` for their useful template for setting up a mlflow server with a postgres and minio backend. Their repository can be found [here](https://github.com/violincoding/mlflow-setup.git). We make slight modifications to the original template to suit our needs, but largely things are unchanges. We include a Makefile in the root of this directory to simplify the process of building and starting the services.

### Prerequisites

Docker and docker-compose should be installed on your machine, either through [Docker Desktop](https://www.docker.com/products/docker-desktop/), or its alternatives such as [Orbstack](https://orbstack.dev/). If you are using docker desktop, make sure it is running.

### Configure environment variables

Make a copy of the `docker/.env.example` file and rename it to `docker/.env`. Then, update the environment variables in the `.env` file as per your requirements.

### Build and start the services

Note, if you are using `pip3`, you will need to alias `pip` to `pip3` in order to install the dependencies.

```bash
alias pip=pip3
```

Build and start the services.

```bash
docker compose up -d --build
```

If everything is setup properly, you should be able to access the services at the following URLs:

- MLFlow Tracking Server: [http://localhost:5001](http://localhost:5001)
- MinIO Console UI: [http://localhost:9001](http://localhost:9001)