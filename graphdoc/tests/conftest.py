# system packages
import os
import logging
from enum import Enum
from pathlib import Path

# internal packages
from graphdoc import Parser
from graphdoc import LocalDataHelper

# external packages
from pytest import fixture
from dotenv import load_dotenv

# logging
log = logging.getLogger(__name__)


# global variables
class OverwriteSchemaCategory(Enum):
    PERFECT = "perfect (TEST)"
    ALMOST_PERFECT = "almost perfect (TEST)"
    POOR_BUT_CORRECT = "poor but correct (TEST)"
    INCORRECT = "incorrect (TEST)"
    BLANK = "blank (TEST)"


class OverwriteSchemaRating(Enum):
    FOUR = "8"
    THREE = "6"
    TWO = "4"
    ONE = "2"
    ZERO = "0"


class OverwriteSchemaCategoryRatingMapping:
    def get_rating(self, category: OverwriteSchemaCategory) -> OverwriteSchemaRating:
        mapping = {
            OverwriteSchemaCategory.PERFECT: OverwriteSchemaRating.FOUR,
            OverwriteSchemaCategory.ALMOST_PERFECT: OverwriteSchemaRating.THREE,
            OverwriteSchemaCategory.POOR_BUT_CORRECT: OverwriteSchemaRating.TWO,
            OverwriteSchemaCategory.INCORRECT: OverwriteSchemaRating.ONE,
            OverwriteSchemaCategory.BLANK: OverwriteSchemaRating.ZERO,
        }
        return mapping.get(category, OverwriteSchemaRating.ZERO)


@fixture
def par() -> Parser:
    return Parser()


@fixture
def default_ldh() -> LocalDataHelper:
    return LocalDataHelper()


@fixture
def overwrite_ldh() -> LocalDataHelper:
    return LocalDataHelper(
        categories=OverwriteSchemaCategory,
        ratings=OverwriteSchemaRating,
        categories_ratings=OverwriteSchemaCategoryRatingMapping.get_rating,
    )
