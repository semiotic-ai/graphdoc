# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import Parser

# external packages
from graphql import DocumentNode
import pytest

# logging
log = logging.getLogger(__name__)

# global variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class TestParser:
    def test_parse_schema_from_file(self, par: Parser):
        schema_file = (
            Path(BASE_DIR)
            / "tests"
            / "assets"
            / "schemas"
            / "opensea_original_schema.graphql"
        )
        schema = par.parse_schema_from_file(schema_file)
        assert isinstance(schema, DocumentNode)
