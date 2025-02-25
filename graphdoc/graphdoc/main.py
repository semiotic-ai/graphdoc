# system packages
import logging
from pathlib import Path
from typing import List, Literal, Optional, Union

# internal packages
from .prompts import SinglePrompt, PromptFactory
from .train import TrainerFactory, SinglePromptTrainer
from .data import (
    setup_logging,
    load_yaml_config,
    LocalDataHelper,
    QualityDataHelper,
    GenerationDataHelper,
)

# external packages
import dspy

# logging
log = logging.getLogger(__name__)


class GraphDoc:
    def __init__(
        self,
        model_args: dict,
        mlflow_tracking_uri: Optional[str] = None,
        log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    ):
        """
        Main entry point for the GraphDoc class. Refer to DSPy for a complete list of model arguments.

        :param model_args: Dictionary containing model arguments.
        :type model_args: dict
        :param mlflow_tracking_uri: MLflow tracking URI.
        :type mlflow_tracking_uri: Optional[str]
        :param log_level: Logging level.
        """
        setup_logging(log_level)
        log.info(f"GraphDoc initialized with model_args: {model_args}")

        try:
            self.lm = dspy.LM(**model_args)
            dspy.configure(lm=self.lm)
        except Exception as e:
            log.error(f"Error initializing LM: {e}")
            raise e

    #######################
    # Data Methods        #
    #######################
    def trainset_from_dict(self, trainset_dict: dict) -> List[dspy.Example]:
        """
        Load a trainset from a dictionary of parameters.

        {
            "hf_api_key": !env HF_DATASET_KEY,                    # Must be a valid Hugging Face API key (with permission to access graphdoc) # TODO: we may make this public in the future
            "load_from_hf": false,                                # Whether to load the dataset from Hugging Face
            "load_from_local": true,                              # Whether to load the dataset from a local directory
            "load_local_specific_category": false,                # Whether to load all categories or a specific category (if load_from_local is true)
            "local_specific_category": perfect,                   # The specific category to load from the dataset (if load_from_local is true)
            "local_parse_objects": true,                          # Whether to parse the objects in the dataset (if load_from_local is true)
            "split_for_eval": true,                               # Whether to split the dataset into trainset and evalset
            "trainset_size": 1000,                                # The size of the trainset
            "evalset_ratio": 0.1,                                 # The proportionate size of the evalset
            "data_helper_type": "quality"                         # Type of data helper to use (quality, generation)
        }

        :param trainset_dict: Dictionary containing trainset parameters.
        :type trainset_dict: dict
        :return: A trainset.
        :rtype: List[dspy.Example]
        """
        # TODO: refactor to enable the passing of alternative schema_directory_path, and the related enums that must be passed in turn
        ldh = LocalDataHelper()

        if trainset_dict["data_helper_type"] == "quality":
            dh = QualityDataHelper()
        elif trainset_dict["data_helper_type"] == "generation":
            dh = GenerationDataHelper()
        else:
            raise ValueError(
                f"Invalid data helper type: {trainset_dict['data_helper_type']}"
            )

        # TODO: refactor to be more ergonomic once we have more data sources implemented
        if trainset_dict["load_from_hf"]:
            raise NotImplementedError("loading from Hugging Face is not implemented")
        if trainset_dict["load_from_local"]:
            if trainset_dict["load_local_specific_category"]:
                raise NotImplementedError(
                    "loading a specific category is not implemented"
                )
            dataset = ldh.folder_of_folders_to_dataset(
                parse_objects=trainset_dict["local_parse_objects"]
            )
            trainset = dh.trainset(dataset)
            if trainset_dict["trainset_size"] and isinstance(
                trainset_dict["trainset_size"], int
            ):
                trainset = trainset[: trainset_dict["trainset_size"]]
            return trainset
        else:
            raise ValueError(
                f"Current implementation only supports loading from local directory"
            )

    def trainset_from_yaml(self, yaml_path: Union[str, Path]) -> List[dspy.Example]:
        """
        Load a trainset from a YAML file.

        data:
            hf_api_key: !env HF_DATASET_KEY                       # Must be a valid Hugging Face API key (with permission to access graphdoc) # TODO: we may make this public in the future
            load_from_hf: false                                   # Whether to load the dataset from Hugging Face
            load_from_local: true                                 # Whether to load the dataset from a local directory
            load_local_specific_category: false                   # Whether to load all categories or a specific category (if load_from_local is true)
            local_specific_category: perfect                      # The specific category to load from the dataset (if load_from_local is true)
            local_parse_objects: True                             # Whether to parse the objects in the dataset (if load_from_local is true)
            split_for_eval: True                                  # Whether to split the dataset into trainset and evalset
            trainset_size: 1000                                   # The size of the trainset
            evalset_ratio: 0.1                                    # The proportionate size of the evalset
            data_helper_type: quality                             # Type of data helper to use (quality, generation)

        :param yaml_path: Path to the YAML file.
        :type yaml_path: Union[str, Path]
        :return: A trainset.
        :rtype: List[dspy.Example]
        """
        pass

    def split_trainset(
        self, trainset: List[dspy.Example], evalset_ratio: float
    ) -> tuple[List[dspy.Example], List[dspy.Example]]:
        """
        Split a trainset into a trainset and evalset.
        """
        pass

    def trainset_and_evalset_from_yaml(
        self, yaml_path: Union[str, Path]
    ) -> tuple[List[dspy.Example], List[dspy.Example]]:
        """
        Load a trainset and evalset from a YAML file.
        """
        pass

    #######################
    # Prompt Methods      #
    #######################
    def single_prompt_from_dict(
        self, prompt_dict: dict, prompt_metric: Union[str, SinglePrompt]
    ) -> SinglePrompt:
        """
        Load a single prompt from a dictionary of parameters.

        {
            "prompt": "doc_quality",             # Which prompt signature to use
            "class": "SchemaDocQualityPrompt",   # Must be a child of SinglePrompt (we will use an enum to map this)
            "type": "predict",                   # The type of prompt to use (predict, chain_of_thought)
            "metric": "rating",                  # The type of metric to use (rating, category)
            "load_from_mlflow": false,           # Whether to load the prompt from an MLFlow URI
            "model_uri": null,                   # The tracking URI for MLflow
            "model_name": null,                  # The name of the model in MLflow
            "model_version": null                # The version of the model in MLflow
            "prompt_metric": False               # Whether another prompt is used to calculate the metric (in which case we must also load that prompt)
        }

        :param prompt_dict: Dictionary containing prompt information.
        :type prompt_dict: dict
        :param prompt_metric: The metric to use to calculate the metric. Can be another prompt signature or a string.
        :type prompt_metric: Union[str, SinglePrompt]
        :return: A SinglePrompt object.
        :rtype: SinglePrompt
        """
        try:
            return PromptFactory.single_prompt(
                prompt=prompt_dict["prompt"],
                prompt_class=prompt_dict["class"],
                prompt_type=prompt_dict["type"],
                prompt_metric=prompt_metric,
            )
        except Exception as e:
            log.error(f"Error creating single prompt: {e}")
            raise e

    def single_prompt_from_yaml(self, yaml_path: Union[str, Path]) -> SinglePrompt:
        """
        Load a single prompt from a YAML file.

        prompt:
            prompt: doc_quality                                   # Which prompt signature to use
            class: SchemaDocQualityPrompt                         # Must be a child of SinglePrompt (we will use an enum to map this)
            type: predict                                         # The type of prompt to use (predict, chain_of_thought)
            metric: rating                                        # The type of metric to use (rating, category)
            load_from_mlflow: false                               # Whether to load the prompt from an MLFlow URI
            model_uri: null                                       # The tracking URI for MLflow
            model_name: null                                      # The name of the model in MLflow
            model_version: null                                   # The version of the model in MLflow
            prompt_metric: False                                  # Whether another prompt is used to calculate the metric (in which case we must also load that prompt)

        prompt_metric:                                            # Follows the same format as the prompt section
            prompt: null                                          # The prompt to use to calculate the metric
            class: null                                           # The class of the prompt to use to calculate the metric
            type: null                                            # The type of prompt to use to calculate the metric
            metric: null                                          # The metric to use to calculate the metric
            load_from_mlflow: false                               # Whether to load the prompt from an MLFlow URI
            model_uri: null                                       # The tracking URI for MLflow
            model_name: null                                      # The name of the model in MLflow
            model_version: null                                   # The version of the model in MLflow

        :param yaml_path: Path to the YAML file.
        :type yaml_path: str
        :return: A SinglePrompt object.
        :rtype: SinglePrompt
        """
        config = load_yaml_config(yaml_path)
        if config["prompt"]["prompt_metric"]:
            prompt_metric_config = config["prompt_metric"]
            prompt_metric_metric = prompt_metric_config["metric"]
            prompt_metric = self.single_prompt_from_dict(
                prompt_metric_config, prompt_metric_metric
            )
        else:
            prompt_metric = config["prompt"]["metric"]
        prompt = self.single_prompt_from_dict(config["prompt"], prompt_metric)
        return prompt

    #######################
    # Trainer Methods     #
    #######################
    def single_trainer_from_dict(
        self, trainer_dict: dict, prompt: SinglePrompt
    ) -> SinglePromptTrainer:
        """
        Load a single trainer from a dictionary of parameters.

        :param trainer_dict: Dictionary containing trainer parameters.
        :type trainer_dict: dict
        :param prompt: The prompt to use for this trainer.
        :type prompt: SinglePrompt
        :return: A SinglePromptTrainer object.
        :rtype: SinglePromptTrainer
        """
        try:
            return TrainerFactory.single_trainer(
                trainer_class=trainer_dict["trainer"]["class"],
                prompt=prompt,
                optimizer_type=trainer_dict["optimizer"]["optimizer_type"],
                optimizer_kwargs=trainer_dict["optimizer"],
                mlflow_model_name=trainer_dict["trainer"]["mlflow_model_name"],
                mlflow_experiment_name=trainer_dict["trainer"][
                    "mlflow_experiment_name"
                ],
                mlflow_tracking_uri=trainer_dict["trainer"]["mlflow_tracking_uri"],
                trainset=[
                    dspy.Example()
                ],  # TODO: we will need to add a method to load the trainset and evalset based on the config
                evalset=[
                    dspy.Example()
                ],  # TODO: we will need to add a method to load the trainset and evalset based on the config
            )
        except Exception as e:
            log.error(f"Error creating single trainer: {e}")
            raise e

    def single_trainer_from_yaml(
        self, yaml_path: Union[str, Path]
    ) -> SinglePromptTrainer:
        """
        Load a single trainer from a YAML file.

        :param yaml_path: Path to the YAML file.
        :type yaml_path: Union[str, Path]
        :return: A SinglePromptTrainer object.
        :rtype: SinglePromptTrainer
        """
        try:
            config = load_yaml_config(yaml_path)
            prompt = self.single_prompt_from_yaml(yaml_path)
            return self.single_trainer_from_dict(config, prompt)
        except Exception as e:
            log.error(f"Error creating trainer from YAML: {e}")
            raise e
