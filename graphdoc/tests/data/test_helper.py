# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
import os
from pathlib import Path

# external packages
import pytest

# internal packages
from graphdoc import (
    check_directory_path,
    check_file_path,
    load_yaml_config,
    setup_logging,
)

# logging
log = logging.getLogger(__name__)

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCHEMA_DIR = BASE_DIR / "tests" / "assets" / "schemas"
CONFIG_DIR = BASE_DIR / "tests" / "assets" / "configs"


class TestHelper:
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Reset logging configuration before and after each test."""
        self.original_level = logging.getLogger().getEffectiveLevel()
        yield
        logging.getLogger().setLevel(self.original_level)

    def test_check_directory_path(self):
        with pytest.raises(ValueError):
            check_directory_path("invalid_path")
        assert check_directory_path(str(SCHEMA_DIR)) is None

    def test_check_file_path(self):
        with pytest.raises(ValueError):
            check_file_path("invalid_path")
        assert (
            check_file_path(str(SCHEMA_DIR / "opensea_original_schema.graphql")) is None
        )

    def test_load_yaml_config(self):
        OPENAI_API_KEY = "test"
        HF_DATASET_KEY = "test"
        MLFLOW_TRACKING_URI = "test"
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        os.environ["HF_DATASET_KEY"] = HF_DATASET_KEY
        os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI
        config_path = CONFIG_DIR / "single_prompt_trainer.yaml"
        config = load_yaml_config(str(config_path))
        assert config is not None
        assert config["language_model"]["lm_api_key"] is not None
        assert config["language_model"]["lm_api_key"] == OPENAI_API_KEY
        assert config["data"]["hf_api_key"] is not None
        assert config["data"]["hf_api_key"] == HF_DATASET_KEY
        assert config["trainer"]["mlflow_tracking_uri"] is not None
        assert config["trainer"]["mlflow_tracking_uri"] == MLFLOW_TRACKING_URI
        del os.environ["OPENAI_API_KEY"]
        del os.environ["HF_DATASET_KEY"]
        del os.environ["MLFLOW_TRACKING_URI"]

    def test_setup_logging(self):
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.WARNING)
        assert root_logger.getEffectiveLevel() == logging.WARNING

        setup_logging("DEBUG")
        assert root_logger.getEffectiveLevel() == logging.DEBUG

        setup_logging("INFO")
        assert root_logger.getEffectiveLevel() == logging.INFO

        setup_logging("WARNING")
        assert root_logger.getEffectiveLevel() == logging.WARNING

        setup_logging("ERROR")
        assert root_logger.getEffectiveLevel() == logging.ERROR
