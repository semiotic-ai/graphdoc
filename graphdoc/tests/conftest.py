# system packages
import os
import logging
from pathlib import Path

# internal packages
from graphdoc import Parser

# external packages
from pytest import fixture
from dotenv import load_dotenv

# logging
log = logging.getLogger(__name__)


@fixture
def par() -> Parser:
    return Parser()
