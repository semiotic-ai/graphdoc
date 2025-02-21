# system packages
import os
import logging
from enum import Enum
from pathlib import Path

# internal packages
from graphdoc import Parser
from graphdoc import GraphDoc
from graphdoc import LocalDataHelper
from graphdoc import DocGeneratorPrompt, DocQualityPrompt

# external packages
from pytest import fixture
from dotenv import load_dotenv

# logging
log = logging.getLogger(__name__)

# load environment variables
load_dotenv("../../.env")


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


@fixture
def gd() -> GraphDoc:
    return GraphDoc(
        model_args={
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "cache": True,
        }
    )


@fixture
def dgp():
    return DocGeneratorPrompt(
        prompt="base_doc_gen",
        prompt_type="chain_of_thought",
        prompt_metric=DocQualityPrompt(
            prompt="doc_quality",
            prompt_type="predict",
            prompt_metric="rating",
        ),
    )
