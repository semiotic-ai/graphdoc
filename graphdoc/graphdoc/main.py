# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
import random
from pathlib import Path
from typing import List, Literal, Optional, Union

# internal packages
from .train import TrainerFactory, SinglePromptTrainer
from .data import (
    setup_logging,
    load_yaml_config,
    LocalDataHelper,
    QualityDataHelper,
    GenerationDataHelper,
    MlflowDataHelper,
    DspyDataHelper,
)
from .prompts import SinglePrompt, PromptFactory, DocGeneratorPrompt
from .modules import DocGeneratorModule
from .eval import DocGeneratorEvaluator

# external packages
import dspy

# logging
log = logging.getLogger(__name__)

# global variables
random.seed(42)


class GraphDoc:
    def __init__(
        self,
        model_args: dict,
        mlflow_tracking_uri: Optional[Union[str, Path]] = None,
        mlflow_tracking_username: Optional[str] = None,
        mlflow_tracking_password: Optional[str] = None,
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

        if mlflow_tracking_uri:
            self.mdh = MlflowDataHelper(
                mlflow_tracking_uri, mlflow_tracking_username, mlflow_tracking_password
            )
        else:
            self.mdh = None
        self.mlflow_tracking_uri = mlflow_tracking_uri

    #######################
    # Class Methods       #
    #######################
    @classmethod
    def from_dict(cls, config_dict: dict) -> "GraphDoc":
        """
        Create a GraphDoc object from a dictionary of parameters.

        {
            "graphdoc": {
                "log_level": "INFO",
                "mlflow_tracking_uri": "http://localhost:5001",
                "mlflow_tracking_username": "admin",
                "mlflow_tracking_password": "password"
            },
            "language_model": {
                "model": "openai/gpt-4o",
                "api_key": "!env OPENAI_API_KEY",
            }
        }
        """
        return GraphDoc(
            model_args=config_dict["language_model"],
            mlflow_tracking_uri=config_dict["graphdoc"].get(
                "mlflow_tracking_uri", None
            ),
            mlflow_tracking_username=config_dict["graphdoc"].get(
                "mlflow_tracking_username", None
            ),
            mlflow_tracking_password=config_dict["graphdoc"].get(
                "mlflow_tracking_password", None
            ),
            log_level=config_dict["graphdoc"].get("log_level", "INFO"),
        )

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "GraphDoc":
        """
        Create a GraphDoc object from a YAML file.

        graphdoc:
            log_level: INFO                                       # The log level to use

        language_model:
            model: openai/gpt-4o                                  # Must be a valid dspy language model
            api_key: !env OPENAI_API_KEY                          # Must be a valid dspy language model API key
            cache: true                                           # Whether to cache the calls to the language model
        """
        config = load_yaml_config(yaml_path)
        return GraphDoc.from_dict(config)

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
        config = load_yaml_config(yaml_path)
        trainset = self.trainset_from_dict(config["data"])
        return trainset

    def split_trainset(
        self, trainset: List[dspy.Example], evalset_ratio: float
    ) -> tuple[List[dspy.Example], List[dspy.Example]]:
        """
        Split a trainset into a trainset and evalset.

        :param trainset: The trainset to split.
        :type trainset: List[dspy.Example]
        :param evalset_ratio: The proportionate size of the evalset.
        :type evalset_ratio: float
        :return: A tuple of trainset and evalset.
        :rtype: tuple[List[dspy.Example], List[dspy.Example]]
        """
        split_idx = int(len(trainset) * (1 - evalset_ratio))
        random.shuffle(trainset)
        evalset = trainset[split_idx:]
        trainset = trainset[:split_idx]
        return trainset, evalset

    def trainset_and_evalset_from_yaml(
        self, yaml_path: Union[str, Path]
    ) -> tuple[List[dspy.Example], List[dspy.Example]]:
        """
        Load a trainset and evalset from a YAML file.

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
        :return: A tuple of trainset and evalset.
        :rtype: tuple[List[dspy.Example], List[dspy.Example]]
        """
        config = load_yaml_config(yaml_path)
        trainset = self.trainset_from_dict(config["data"])
        return self.split_trainset(trainset, config["data"]["evalset_ratio"])

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
            # if we are loading from mlflow, modify the prompt_dict with the loaded model
            if prompt_dict["load_from_mlflow"]:
                if self.mdh:
                    log.info(f"Loading prompt from MLflow: {prompt_dict}")
                    prompt = self.mdh.model_by_args(prompt_dict)
                    log.info(f"Prompt loaded from MLflow: {prompt}")
                    prompt_signature = DspyDataHelper.prompt_signature(prompt)
                    prompt_dict["prompt"] = prompt_signature
                else:
                    raise ValueError("MLflow tracking URI not provided")

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
        self,
        trainer_dict: dict,
        prompt: SinglePrompt,
        trainset: List[dspy.Example] = [dspy.Example()],
        evalset: List[dspy.Example] = [dspy.Example()],
    ) -> SinglePromptTrainer:
        """
        Load a single trainer from a dictionary of parameters.

        {
            "trainer": {
                "class": "DocQualityTrainer",
                "mlflow_model_name": "doc_quality_model",
                "mlflow_experiment_name": "doc_quality_experiment",
                "mlflow_tracking_uri": "http://localhost:5000"
            },
            "optimizer": {
                "optimizer_type": "miprov2",
                "auto": "light",
                "max_labeled_demos": 2,
                "max_bootstrapped_demos": 4,
                "num_trials": 2,
                "minibatch": true
            },
        }

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
                trainset=trainset,
                evalset=evalset,
            )
        except Exception as e:
            log.error(f"Error creating single trainer: {e}")
            raise e

    def single_trainer_from_yaml(
        self, yaml_path: Union[str, Path]
    ) -> SinglePromptTrainer:
        """
        Load a single trainer from a YAML file.

        data:
            hf_api_key: !env HF_DATASET_KEY                       # Must be a valid Hugging Face API key (with permission to access graphdoc) # TODO: we may make this public in the future
            load_from_hf: false                                   # Whether to load the dataset from Hugging Face
            load_from_local: true                                 # Whether to load the dataset from a local directory
            load_local_specific_category: false                   # Whether to load all categories or a specific category (if load_from_local is true)
            local_specific_category: perfect                      # The specific category to load from the dataset (if load_from_local is true)
            local_parse_objects: True                             # Whether to parse the objects in the dataset (if load_from_local is true)
            split_for_eval: True                                  # Whether to split the dataset into trainset and evalset
            trainset_size: 10                                     # The size of the trainset
            evalset_ratio: 0.1                                    # The proportionate size of the evalset
            data_helper_type: generation                          # Type of data helper to use (quality, generation)

        prompt:
            prompt: base_doc_gen                                  # Which prompt signature to use
            class: DocGeneratorPrompt                             # Must be a child of SinglePrompt (we will use an enum to map this)
            type: chain_of_thought                                # The type of prompt to use (predict, chain_of_thought)
            metric: rating                                        # The type of metric to use (rating, category)
            load_from_mlflow: false                               # Whether to load the prompt from an MLFlow URI
            model_uri: null                                       # The tracking URI for MLflow
            model_name: null                                      # The name of the model in MLflow
            model_version: null                                   # The version of the model in MLflow
            prompt_metric: true                                   # Whether another prompt is used to calculate the metric (in which case we must also load that prompt)

        prompt_metric:
            prompt: doc_quality                                   # The prompt to use to calculate the metric
            class: DocQualityPrompt                               # The class of the prompt to use to calculate the metric
            type: predict                                         # The type of prompt to use to calculate the metric
            metric: rating                                        # The metric to use to calculate the metric
            load_from_mlflow: false                               # Whether to load the prompt from an MLFlow URI
            model_uri: null                                       # The tracking URI for MLflow
            model_name: null                                      # The name of the model in MLflow
            model_version: null                                   # The version of the model in MLflow

        :param yaml_path: Path to the YAML file.
        :type yaml_path: Union[str, Path]
        :return: A SinglePromptTrainer object.
        :rtype: SinglePromptTrainer
        """
        try:
            config = load_yaml_config(yaml_path)
            prompt = self.single_prompt_from_yaml(yaml_path)
            trainset, evalset = self.trainset_and_evalset_from_yaml(yaml_path)
            return self.single_trainer_from_dict(config, prompt, trainset, evalset)
        except Exception as e:
            log.error(f"Error creating trainer from YAML: {e}")
            raise e

    #######################
    # Module Methods      #
    #######################
    def doc_generator_module_from_dict(
        self, module_dict: dict, prompt: Union[DocGeneratorPrompt, SinglePrompt]
    ) -> DocGeneratorModule:
        """
        Load a doc generator module from a dictionary of parameters.

        {
            "retry": true,
            "retry_limit": 1,
            "rating_threshold": 3,
            "fill_empty_descriptions": true
        }

        :param module_dict: Dictionary containing module parameters.
        :type module_dict: dict
        :param prompt: The prompt to use for this module.
        :type prompt: DocGeneratorPrompt
        :return: A DocGeneratorModule object.
        :rtype: DocGeneratorModule
        """
        return DocGeneratorModule(
            prompt=prompt,
            retry=module_dict["retry"],
            retry_limit=module_dict["retry_limit"],
            rating_threshold=module_dict["rating_threshold"],
            fill_empty_descriptions=module_dict["fill_empty_descriptions"],
        )

    def doc_generator_module_from_yaml(
        self, yaml_path: Union[str, Path]
    ) -> DocGeneratorModule:
        """
        Load a doc generator module from a YAML file.

        prompt:
            prompt: base_doc_gen                                  # Which prompt signature to use
            class: DocGeneratorPrompt                             # Must be a child of SinglePrompt (we will use an enum to map this)
            type: chain_of_thought                                # The type of prompt to use (predict, chain_of_thought)
            metric: rating                                        # The type of metric to use (rating, category)
            load_from_mlflow: false                               # Whether to load the prompt from an MLFlow URI
            model_uri: null                                       # The tracking URI for MLflow
            model_name: null                                      # The name of the model in MLflow
            model_version: null                                   # The version of the model in MLflow
            prompt_metric: true                                   # Whether another prompt is used to calculate the metric (in which case we must also load that prompt)

        prompt_metric:
            prompt: doc_quality                                   # The prompt to use to calculate the metric
            class: DocQualityPrompt                               # The class of the prompt to use to calculate the metric
            type: predict                                         # The type of prompt to use to calculate the metric
            metric: rating                                        # The metric to use to calculate the metric
            load_from_mlflow: false                               # Whether to load the prompt from an MLFlow URI
            model_uri: null                                       # The tracking URI for MLflow
            model_name: null                                      # The name of the model in MLflow
            model_version: null                                   # The version of the model in MLflow

        module:
            retry: true                                           # Whether to retry the generation if the quality check fails
            retry_limit: 1                                        # The maximum number of retries
            rating_threshold: 3                                   # The rating threshold for the quality check
            fill_empty_descriptions: true                         # Whether to fill the empty descriptions in the schema
        """
        config = load_yaml_config(yaml_path)["module"]
        prompt = self.single_prompt_from_yaml(yaml_path)
        return self.doc_generator_module_from_dict(config, prompt)

    #######################
    # Eval Methods        #
    #######################
    def doc_generator_eval_from_yaml(
        self, yaml_path: Union[str, Path]
    ) -> DocGeneratorEvaluator:
        """
        Load a doc generator evaluator from a YAML file.
        """
        # load the generator
        generator = self.doc_generator_module_from_yaml(yaml_path)
        config = load_yaml_config(yaml_path)

        # load the evaluator
        metric_config = config["prompt_metric"]
        evaluator = self.single_prompt_from_dict(metric_config, metric_config["metric"])

        # load the eval config
        if self.mdh is not None:
            mlflow_tracking_uri = self.mdh.mlflow_tracking_uri
        else:
            mlflow_tracking_uri = config["eval"]["mlflow_tracking_uri"]
        mlflow_experiment_name = config["eval"]["mlflow_experiment_name"]
        generator_prediction_field = config["eval"]["generator_prediction_field"]
        evaluator_prediction_field = config["eval"]["evaluator_prediction_field"]
        readable_value = config["eval"]["readable_value"]

        # load the evalset
        evalset = self.trainset_from_yaml(yaml_path)

        # return the evaluator
        return DocGeneratorEvaluator(
            generator=generator,
            evaluator=evaluator,
            evalset=evalset,
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_experiment_name=mlflow_experiment_name,
            generator_prediction_field=generator_prediction_field,
            evaluator_prediction_field=evaluator_prediction_field,
            readable_value=readable_value,
        )
