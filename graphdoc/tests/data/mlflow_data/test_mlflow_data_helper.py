# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import MlflowDataHelper

# external packages
import mlflow

# logging
log = logging.getLogger(__name__)

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MLFLOW_DIR = Path(BASE_DIR) / "assets" / "mlruns"


class TestMlflowDataHelper:
    def test_init_mlflow_data_helper(self):
        mlflow_data_helper = MlflowDataHelper(mlflow_tracking_uri=MLFLOW_DIR)
        assert mlflow_data_helper is not None
        assert isinstance(mlflow_data_helper, MlflowDataHelper)
        assert isinstance(mlflow_data_helper.mlflow_client, mlflow.MlflowClient)

    def test_model_by_name_and_version(self):
        pass
