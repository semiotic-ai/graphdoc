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
import mlflow

# logging
log = logging.getLogger(__name__)

# define test asset paths
TEST_DIR = Path(__file__).resolve().parent
ASSETS_DIR = TEST_DIR / "assets"
MLRUNS_DIR = ASSETS_DIR / "mlruns"
ENV_PATH = TEST_DIR / ".env"

# load environment variables
# load_dotenv(ENV_FILE)
load_dotenv(ENV_PATH)


# Set default environment variables if not present
def ensure_env_vars():
    """Ensure all required environment variables are set with defaults if needed."""
    env_defaults = {
        "OPENAI_API_KEY": None,  # No default, must be provided
        "HF_DATASET_KEY": None,  # No default, must be provided
        "MLFLOW_TRACKING_URI": str(MLRUNS_DIR),
    }
    log.info(f"Environment variable path: {ENV_PATH}")

    for key in env_defaults:
        log.info(
            f"Environment variable {key}: {'SET' if key in os.environ else 'NOT SET'}"
        )

    for key, default in env_defaults.items():
        if key not in os.environ and default is not None:
            os.environ[key] = default
            log.info(f"Setting default for {key}: {default}")
        elif key not in os.environ and default is None:
            log.warning(f"Required environment variable {key} not set")


@fixture(autouse=True)
def setup_env():
    """Fixture to ensure environment is properly set up before each test."""
    ensure_env_vars()


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
    """Fixture for GraphDoc with proper environment setup."""
    ensure_env_vars()
    return GraphDoc(
        model_args={
            "model": "gpt-4o-mini",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "cache": True,
        }
    )


@fixture
def dqp():
    return DocQualityPrompt(
        prompt="doc_quality",
        prompt_type="predict",
        prompt_metric="rating",
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
