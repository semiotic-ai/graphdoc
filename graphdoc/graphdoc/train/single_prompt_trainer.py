# system packages
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple

# internal packages
from ..prompts import SinglePrompt
from ..data import MlflowDataHelper

# external packages
import dspy
import mlflow

# logging
log = logging.getLogger(__name__)


class SinglePromptTrainer(ABC):
    def __init__(
        self,
        prompt: SinglePrompt,
        optimizer_type: str,
        optimizer_kwargs: Dict[str, Any],
        mlflow_model_name: str,
        mlflow_experiment_name: str,
        mlflow_tracking_uri: str,
        trainset: List[dspy.Example],
        evalset: List[dspy.Example],
    ):
        """
        Initialize the SinglePromptTrainer. This is the base class for implementing a trainer for a single prompt.

        :param prompt: The prompt to train.
        :type prompt: SinglePrompt
        :param optimizer_type: The type of optimizer to use.
        :type optimizer_type: str
        :param optimizer_kwargs: The keyword arguments for the optimizer.
        :type optimizer_kwargs: Dict[str, Any]
        :param mlflow_model_name: The name of the model in mlflow.
        :type mlflow_model_name: str
        :param mlflow_experiment_name: The name of the experiment in mlflow.
        :type mlflow_experiment_name: str
        :param mlflow_tracking_uri: The uri of the mlflow tracking server.
        :type mlflow_tracking_uri: str
        :param trainset: The training set.
        :type trainset: List[dspy.Example]
        """
        self.prompt = prompt
        self.optimizer_type = optimizer_type
        self.optimizer_kwargs = optimizer_kwargs
        self.mlflow_model_name = mlflow_model_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_experiment_name = mlflow_experiment_name
        self.trainset = trainset
        self.evalset = evalset

        # setup mlflow
        log.info(f"---------------------------------------------------------")
        log.info(f"Setting MLFlow tracking URI to {self.mlflow_tracking_uri}")
        log.info(f"---------------------------------------------------------")

        mlflow.dspy.autolog()
        self.mlflow_data_helper = MlflowDataHelper(self.mlflow_tracking_uri)
        experiment = mlflow.set_experiment(self.mlflow_experiment_name)

        log.info(f"Setting MLFlow experiment to {self.mlflow_experiment_name}")
        log.info(f"Experiment_id: {experiment.experiment_id}")
        log.info(f"Artifact Location: {experiment.artifact_location}")
        log.info(f"Tags: {experiment.tags}")
        log.info(f"Lifecycle_stage: {experiment.lifecycle_stage}")
        log.info(f"---------------------------------------------------------")

    ####################
    # Abstract Methods #
    ####################

    # TODO: decide on a return type and implement better type checking for parameters
    @abstractmethod
    def evaluation_metrics(self, base_evaluation, optimized_evaluation):
        """
        Log evaluation metrics to mlflow.

        :param base_evaluation: The evaluation metrics of the base model.
        :type base_evaluation: Any
        :param optimized_evaluation: The evaluation metrics of the optimized model.
        :type optimized_evaluation: Any
        """
        pass

    @abstractmethod
    def evaluate_training(
        self, base_model, optimized_model
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Evaluate the training of the model. Comparing the base and optimized models.

        :param base_model: The base model.
        :type base_model: Any
        :param optimized_model: The optimized model.
        :type optimized_model: Any
        """
        pass

    @abstractmethod
    def train(
        self, load_model_args: Optional[Dict[str, Any]] = None, save_model: bool = True
    ):
        """
        Train the model.

        :param load_model_args: The arguments to load the model.
        :type load_model_args: Dict[str, Any]
        :param save_model: Whether to save the model.
        :type save_model: bool
        """
        pass
