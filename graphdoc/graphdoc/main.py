# system packages
import logging
from pathlib import Path
from typing import Literal, Optional, Union

# internal packages
from .prompts import SinglePrompt, PromptFactory
from .data import setup_logging, load_yaml_config

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
