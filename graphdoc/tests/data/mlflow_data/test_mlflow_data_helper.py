# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import MlflowDataHelper

# external packages
import dspy
import mlflow

# logging
log = logging.getLogger(__name__)

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
MLFLOW_DIR = Path(BASE_DIR) / "tests" / "assets" / "mlruns"


class TestMlflowDataHelper:
    def test_init_mlflow_data_helper(self):
        mdh = MlflowDataHelper(mlflow_tracking_uri=MLFLOW_DIR)
        assert mdh is not None
        assert isinstance(mdh, MlflowDataHelper)
        assert isinstance(mdh.mlflow_client, mlflow.MlflowClient)

    def test_latest_model_version(self):
        mdh = MlflowDataHelper(mlflow_tracking_uri=MLFLOW_DIR)
        log.info(f"mlflow_tracking_uri: {mdh.mlflow_tracking_uri}")
        model = mdh.latest_model_version(model_name="doc_generator_model")
        assert model is not None
        assert isinstance(model, dspy.ChainOfThought)

    def test_model_by_name_and_version(self):
        mdh = MlflowDataHelper(mlflow_tracking_uri=MLFLOW_DIR)
        tracking_uri = mlflow.get_tracking_uri()
        model = mdh.model_by_name_and_version(
            model_name="doc_generator_model", model_version="1"
        )
        assert model is not None
        assert isinstance(model, dspy.ChainOfThought)

    def test_model_by_uri(self):
        model_uri = (
            Path(MLFLOW_DIR)
            / "513408250948216117"
            / "976d330558344c41b30bd1531571de18"
            / "artifacts"
            / "model"
        )
        mdh = MlflowDataHelper(mlflow_tracking_uri=MLFLOW_DIR)
        model = mdh.model_by_uri(model_uri=str(model_uri))
        assert model is not None
        assert isinstance(model, dspy.ChainOfThought)
