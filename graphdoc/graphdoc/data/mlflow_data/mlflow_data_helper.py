# system packages
import ast
import logging
from pathlib import Path
from typing import Any, Union

# internal packages

# external packages
import dspy
import mlflow
from mlflow.models import ModelSignature

# logging
log = logging.getLogger(__name__)


class MlflowDataHelper:
    def __init__(self, mlflow_tracking_uri: Union[str, Path]):
        """
        A helper class for loading and saving models and metadata from mlflow.

        :param mlflow_tracking_uri: The uri of the mlflow tracking server.
        :type mlflow_tracking_uri: Union[str, Path]
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(str(mlflow_tracking_uri))
        self.mlflow_client = mlflow.MlflowClient(
            tracking_uri=str(self.mlflow_tracking_uri)
        )

    #########################
    # Model Loading Methods #
    #########################
    # TODO: add return type and error handling
    def latest_model_version(self, model_name: str):
        """
        Load the latest version of a model from mlflow.

        :param model_name: The name of the model to load.
        :type model_name: str
        :return: The loaded model.
        """
        model_latest_version = self.mlflow_client.get_latest_versions(model_name)
        return mlflow.dspy.load_model(model_latest_version[0].source)

    # TODO: add return type and error handling
    def model_by_name_and_version(self, model_name: str, model_version: str):
        """
        Load a model from mlflow by name and version.

        :param model_name: The name of the model to load.
        :type model_name: str
        :param model_version: The version of the model to load.
        :type model_version: str
        :return: The loaded model.
        """
        model = self.mlflow_client.get_model_version(model_name, model_version)
        return mlflow.dspy.load_model(model.source)

    # TODO: add return type and error handling
    def model_by_uri(self, model_uri: str):
        """
        Load a model from mlflow by uri.

        :param model_uri: The uri of the model to load.
        :type model_uri: str
        :return: The loaded model.
        """
        return mlflow.dspy.load_model(model_uri)

    #########################
    # Model Saving Methods  #
    #########################

    def save_model(
        self, model: dspy.Signature, model_signature: ModelSignature, model_name: str
    ):
        """
        Save a model to mlflow.

        :param model: The model to save.
        :type model: dspy.Signature
        :param model_signature: The signature of the model.
        :type model_signature: ModelSignature
        :param model_name: The name of the model to save.
        :type model_name: str
        """
        mlflow.dspy.log_model(
            dspy_model=model,
            artifact_path="model",
            signature=model_signature,
            task=None,
            registered_model_name=model_name,
        )  # TODO: add metadata related to trainset and evalset

    ############################
    # Metadata Loading Methods #
    ############################

    def run_parameters(self, run_id: str) -> dict[str, Any]:
        """
        Load the parameters of a run from mlflow.

        :param run_id: The id of the run to load the parameters from.
        :type run_id: str
        :return: The parameters of the run.
        """
        run = self.mlflow_client.get_run(run_id)

        # go through and convert the nested dictionaries to actual dictionaries (as they are currently strings)
        for key, value in run.data.params.items():
            run.data.params[key] = ast.literal_eval(value)
        return run.data.params
