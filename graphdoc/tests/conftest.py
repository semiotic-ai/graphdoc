# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

import logging

# system packages
import os
from enum import Enum
from pathlib import Path

from dotenv import load_dotenv

# external packages
from pytest import fixture

# internal packages
from graphdoc import (
    DocGeneratorPrompt,
    DocQualityPrompt,
    GraphDoc,
    LocalDataHelper,
    Parser,
)

# logging
log = logging.getLogger(__name__)

# define test asset paths
TEST_DIR = Path(__file__).resolve().parent
ASSETS_DIR = TEST_DIR / "assets"
MLRUNS_DIR = ASSETS_DIR / "mlruns"
ENV_PATH = TEST_DIR / ".env"

# Check if .env file exists
if not ENV_PATH.exists():
    log.error(f".env file not found at {ENV_PATH}")
else:
    log.info(f".env file found at {ENV_PATH}")
    load_dotenv(dotenv_path=ENV_PATH, override=True)


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
        value = os.environ.get(key, "NOT SET")
        if value != "NOT SET":
            if "API_KEY" in key or "DATASET_KEY" in key:
                log.info(f"Environment variable {key}: SET (value masked)")
            else:
                log.info(f"Environment variable {key}: SET to {value}")
        else:
            log.info(f"Environment variable {key}: NOT SET")

    for key, default in env_defaults.items():
        if key not in os.environ and default is not None:
            os.environ[key] = default
            log.info(f"Setting default for {key}: {default}")
        elif key not in os.environ and default is None:
            log.warning(f"Required environment variable {key} not set")


@fixture(autouse=True, scope="session")
def setup_env():
    """Fixture to ensure environment is properly set up before each test."""
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)
    ensure_env_vars()


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
    # Ensure environment is set up correctly
    if ENV_PATH.exists():
        load_dotenv(dotenv_path=ENV_PATH, override=True)
    ensure_env_vars()

    api_key = os.environ.get("OPENAI_API_KEY")
    mlflow_tracking_username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    mlflow_tracking_password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    if not api_key:
        log.error("OPENAI_API_KEY still not available after loading .env file")

    return GraphDoc(
        model_args={
            "model": "gpt-4o-mini",
            "api_key": api_key,
            "cache": True,
        },
        mlflow_tracking_uri=MLRUNS_DIR,
        mlflow_tracking_username=mlflow_tracking_username,
        mlflow_tracking_password=mlflow_tracking_password,
        log_level="INFO",
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
