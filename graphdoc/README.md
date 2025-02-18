# graphdoc

The `graphdoc` package is a Python package for assisting in the generation of subgraph documentation through the utilization of Large Language Models.

## Development

We utilize [poetry](https://python-poetry.org/) for dependency management. Please run `poetry install` to install the dependencies. You can also run `poetry shell` to activate the virtual environment. Please see the [poetry documentation](https://python-poetry.org/docs/) for more information.

### run.sh

The `run.sh` script is a convenience script for development. It provides a few shortcuts for running useful commands.

```bash 
# ensure that the script is executable
chmod +x run.sh

# install dependencies (including dev dependencies)
./run.sh dev # use `./run.sh install` to install dependencies excluding dev dependencies
```