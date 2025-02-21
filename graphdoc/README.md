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

### Future Work

- [ ] Add [pytest-watch](https://github.com/joeyespo/pytest-watch) to dev dependencies

While there could be some benefit of using something such as [vcrpy]([text](https://github.com/kevin1024/vcrpy?tab=readme-ov-file)), we are going to get a majority of our benefit from using [pytest-testmon]([text](https://github.com/joeyespo/pytest-watch)). Once we start adding more compute intensive tests and functionality (like having a test for optimization runs), we will want to consider using `vcrpy`, or better caching for reproducibility.