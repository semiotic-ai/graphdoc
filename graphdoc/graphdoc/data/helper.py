# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import os
import yaml
import logging
from pathlib import Path
from yaml import SafeLoader
from typing import Literal, Union

# internal packages

# external packages

# logging
log = logging.getLogger(__name__)


def check_directory_path(directory_path: Union[str, Path]) -> None:
    """
    Check if the provided path resolves to a valid directory.

    :param directory_path: The path to check.
    :type directory_path: Union[str, Path]
    :raises ValueError: If the path does not resolve to a valid directory.
    :return: None
    :rtype: None
    """
    _directory_path = Path(directory_path).resolve()
    if not _directory_path.is_dir():
        raise ValueError(
            f"The provided path does not resolve to a valid directory: {directory_path}"
        )


def check_file_path(file_path: Union[str, Path]) -> None:
    """
    Check if the provided path resolves to a valid file.

    :param file_path: The path to check.
    :type file_path: Union[str, Path]
    :raises ValueError: If the path does not resolve to a valid file.
    :return: None
    :rtype: None
    """
    _file_path = Path(file_path).resolve()
    if not _file_path.is_file():
        raise ValueError(
            f"The provided path does not resolve to a valid file: {file_path}"
        )


def _env_constructor(loader: SafeLoader, node: yaml.nodes.ScalarNode) -> str:
    """
    Custom constructor for environment variables.

    :param loader: The YAML loader.
    :type loader: yaml.SafeLoader
    :param node: The node to construct.
    :type node: yaml.nodes.ScalarNode
    :return: The environment variable value.
    :rtype: str
    :raises ValueError: If the environment variable is not set.
    """
    value = loader.construct_scalar(node)
    env_value = os.getenv(value)
    if env_value is None:
        raise ValueError(f"Environment variable '{value}' is not set.")
    return env_value


def load_yaml_config(file_path: Union[str, Path], use_env: bool = True) -> dict:
    """
    Load a YAML configuration file.

    :param file_path: The path to the YAML file.
    :type file_path: Union[str, Path]
    :param use_env: Whether to use environment variables.
    :type use_env: bool
    :return: The YAML configuration.
    :rtype: dict
    :raises ValueError: If the path does not resolve to a valid file or the environment variable is not set.
    """
    if use_env:
        SafeLoader.add_constructor("!env", _env_constructor)

    _file_path = Path(file_path).resolve()
    if not _file_path.is_file():
        raise ValueError(
            f"The provided path does not resolve to a valid file: {file_path}"
        )
    with open(_file_path, "r") as file:
        return yaml.load(file, Loader=SafeLoader)


def setup_logging(
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
):
    """
    Setup logging for the application.

    :param log_level: The log level.
    :type log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    """
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(getattr(logging, log_level))

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )

    root_logger.addHandler(handler)
