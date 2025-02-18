# system packages
import logging
from pathlib import Path
from typing import Union

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
