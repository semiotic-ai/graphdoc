[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Conventional Commits](https://img.shields.io/badge/Conventional%20Commits-1.0.0-%23FE5196?logo=conventionalcommits&logoColor=white)](https://conventionalcommits.org)
[![Semantic Versioning](https://img.shields.io/badge/semver-2.0.0-green)](https://semver.org/spec/v2.0.0.html)

# graphdoc

Automating the generation of subgraph documentation through the utilization of Large Language Models.

## Development

We utilize [poetry](https://python-poetry.org/) for dependency management. Please run `poetry install` to install the dependencies from within the relevant directory (e.g. `graphdoc` or `mlflow-manager`). You can also run `poetry shell` to activate the virtual environment. Please see the [poetry documentation](https://python-poetry.org/docs/) for more information.

We utilize [commitizen](https://commitizen-tools.github.io/commitizen/) for commit messages and semantic versioning. Please run `cz commit` to commit your changes. Commitizen can be installed with `pip install commitizen` or `brew install commitizen`.

We utilize [docker](https://www.docker.com/) for managing the tracking of our service and associated expirements through [mlflow](https://mlflow.org/). In our docker image, we spin up a [mlflow](https://mlflow.org/), [postgres](https://www.postgresql.org/), and [minio](https://min.io/) instance. This is very similar to our production setup, and allows for a pretty smooth development flow between local and prod. Please ensure you have downloaded and are running docker in the background of your machine. 

Here are some quick commands for getting started:

```bash 
brew install poetry
brew install commitizen
```

```bash 
cd graphdoc
poetry install 

cd ../mlflow-manager
poetry install 
```

### .env

There are two `.env` files that we expect the user to set up. They are divided between `mlflow-manager` and `graphdoc`. First, let's setup the `mlflow-manager` `.env` file. You can leave these values as they are, or modify them as you see fit:

```bash
# navigate to the docker root
cd mlflow-manager
cd docker

# copy the .env.example for setup
cp .env.example .env # set values directly in your newly created .env file 
```

Next, let's set up the `.env` file to be used by our `graphdoc` program. 

```bash
# navigate to the graphdoc root 
cd ../..

# copy the .env.example for setup
cp .env.example .env # set values directly in your newly created .env file 
```

### run.sh

The `run.sh` script is a convenience script for development. It provides a few shortcuts for running useful commands.

```bash 
# make sure you are in the root of the repository 
# ensure that the script is executable
chmod +x run.sh

# install both packages in development mode
./run.sh dev
```

To setup the mlflow-manager services, run the following command:

```bash 
# default username: admin
# default password: password
./run.sh mlflow-setup
```

Below, we provide an overview of the commands available in the `run.sh` script. 

| Command | Description |
|---------|-------------|
| `./run.sh dev` | Install both packages (graphdoc and mlflow-manager) in development mode |
| `./run.sh install` | Install both packages (graphdoc and mlflow-manager) in production mode |
| `./run.sh mlflow-setup` | Setup the mlflow-manager services |
| `./run.sh mlflow-teardown` | Teardown the mlflow-manager services |
| `./run.sh doc-quality-train` | Train a document quality model (using the default config in `graphdoc/assets/configs/single_prompt_doc_quality_trainer.yaml`) |
| `./run.sh build-and-run-doc-quality-trainer` | Build and run the document quality trainer (./run.sh mlflow-setup && ./run.sh doc-quality-train) |

### python 

We utilize `python==3.13.0` for this project. Please see the [python documentation](https://www.python.org/downloads/) for more information. We recommend using [pyenv](https://github.com/pyenv/pyenv) to install python.

Below, we include some useful commands for installing python using pyenv.

```bash 
# install pyenv
brew install pyenv

# install python 3.13.0
pyenv install 3.13.0

# set the python version
pyenv local 3.13.0

# check the python version
python --version
```

There is a chance that your system may have python allocated under the name `python3`. If this is the case, the following commands may help resolve the `python` namespace to your `python3` installation.

```bash 
# export the pyenv path
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"

# reload the shell
source ~/.zshrc

# check the versions
python --version  # Should show 3.13.0
which python  # Should show ~/.pyenv/shims/python
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE).
