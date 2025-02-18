# system packages
import logging
import os
from pathlib import Path

# internal packages
from graphdoc import check_directory_path, check_file_path, load_yaml_config

# external packages
import pytest

# logging
log = logging.getLogger(__name__)

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCHEMA_DIR = BASE_DIR / "tests" / "assets" / "schemas"
CONFIG_DIR = BASE_DIR / "tests" / "assets" / "configs"


class TestHelper:

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
