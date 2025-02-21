# system packages
import os
import logging
from enum import Enum
from pathlib import Path

# internal packages
from graphdoc import Parser
from graphdoc import LocalDataHelper
from .conftest import (
    OverwriteSchemaCategory,
    OverwriteSchemaRating,
    OverwriteSchemaCategoryRatingMapping,
)

# external packages
from pytest import fixture
from dotenv import load_dotenv

# logging
log = logging.getLogger(__name__)


class TestFixtures:

    def test_parser(self, par: Parser):
        assert par is not None
        assert isinstance(par, Parser)

    def test_default_ldh(self, default_ldh: LocalDataHelper):
        assert default_ldh is not None
        assert isinstance(default_ldh, LocalDataHelper)

    def test_overwrite_ldh(self, overwrite_ldh: LocalDataHelper):
        assert overwrite_ldh is not None
        assert isinstance(overwrite_ldh, LocalDataHelper)
        assert overwrite_ldh.categories == OverwriteSchemaCategory
        assert overwrite_ldh.ratings == OverwriteSchemaRating
        assert (
            overwrite_ldh.categories_ratings
            == OverwriteSchemaCategoryRatingMapping.get_rating
        )
