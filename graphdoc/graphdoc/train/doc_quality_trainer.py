# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import io
import logging
from typing import Any, Dict, List, Optional, Tuple

# external packages
import dspy
import mlflow
import pandas as pd

from ..data import DspyDataHelper, QualityDataHelper

# internal packages
from ..prompts import DocQualityPrompt
from .optimizers import optimizer_compile
from .single_prompt_trainer import SinglePromptTrainer

# logging
log = logging.getLogger(__name__)


class DocQualityTrainer(SinglePromptTrainer):
    def __init__(
        self,
        prompt: DocQualityPrompt,
        optimizer_type: str,
        optimizer_kwargs: Dict[str, Any],
        mlflow_model_name: str,
        mlflow_experiment_name: str,
        mlflow_tracking_uri: str,
        trainset: List[dspy.Example],
        evalset: List[dspy.Example],
    ):
        """
        Initialize the DocQualityTrainer. This is the base class for implementing a trainer for a DocQualityPrompt.

        :param prompt: The prompt to train.
        :type prompt: DocQualityPrompt
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
        """
        super().__init__(
            prompt=prompt,
            optimizer_type=optimizer_type,
            optimizer_kwargs=optimizer_kwargs,
            mlflow_model_name=mlflow_model_name,
            mlflow_experiment_name=mlflow_experiment_name,
            mlflow_tracking_uri=mlflow_tracking_uri,
            trainset=trainset,
            evalset=evalset,
        )

    ####################
    # Abstract Methods #
    ####################
    def evaluation_metrics(self, base_evaluation, optimized_evaluation):
        """
        Log evaluation metrics to mlflow. We will log the overall scores and the per category scores. Per category scores will be logged as a csv file.

        :param base_evaluation: The evaluation metrics of the base model.
        :type base_evaluation: Any
        :param optimized_evaluation: The evaluation metrics of the optimized model.
        :type optimized_evaluation: Any
        """
        base_evaluation_overall_score = base_evaluation["overall_score"]
        optimized_evaluation_overall_score = optimized_evaluation["overall_score"]

        mlflow.log_metrics(
            {
                "base_evaluation_overall_score": base_evaluation_overall_score,
                "optimized_evaluation_overall_score": optimized_evaluation_overall_score,
            }
        )

        metrics_data = {
            "Evaluation Type": ["Base Evaluation", "Optimized Evaluation"],
            "Overall Score": [
                base_evaluation_overall_score,
                optimized_evaluation_overall_score,
            ],
        }

        for key, value in base_evaluation["per_category_scores"].items():
            metrics_data[f"{key} Percent Correct"] = [
                value["percent_correct"],
                optimized_evaluation["per_category_scores"][key]["percent_correct"],
            ]
            metrics_data[f"{key} Total"] = [
                value["total"],
                optimized_evaluation["per_category_scores"][key]["total"],
            ]
            metrics_data[f"{key} Correct"] = [
                value["correct"],
                optimized_evaluation["per_category_scores"][key]["correct"],
            ]

        df = pd.DataFrame(metrics_data)
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        mlflow.log_text(csv_buffer.getvalue(), "evaluation_comparison.csv")

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
        base_prompt = DocQualityPrompt(
            prompt=DspyDataHelper.prompt_signature(base_model),
            prompt_type=self.prompt.prompt_type,  # type: ignore # TODO: we should have better type handling, but we know this works
            prompt_metric=self.prompt.prompt_metric,  # type: ignore # TODO: we should have better type handling, but we know this works
        )

        optimized_prompt = DocQualityPrompt(
            prompt=DspyDataHelper.prompt_signature(optimized_model),
            prompt_type=self.prompt.prompt_type,  # type: ignore # TODO: we should have better type handling, but we know this works
            prompt_metric=self.prompt.prompt_metric,  # type: ignore # TODO: we should have better type handling, but we know this works
        )

        base_evaluation = base_prompt.evaluate_evalset(self.evalset)
        optimized_evaluation = optimized_prompt.evaluate_evalset(self.evalset)

        self.evaluation_metrics(base_evaluation, optimized_evaluation)
        return base_evaluation, optimized_evaluation

    def train(
        self, load_model_args: Optional[Dict[str, Any]] = None, save_model: bool = True
    ):
        """
        Train the model. If provided, we will load the model from mlflow. Otherwise, we will use the provided DocQualityPrompt as the base model.

        :param load_model_args: The arguments to load the model.
        :type load_model_args: Dict[str, Any]
        :param save_model: Whether to save the model.
        :type save_model: bool
        """
        # if model args are provided, load the model from mlflow.
        if load_model_args:
            # we assume the user wants to load the model as was stored, without modifying the module type (e.g. dspy.Predict, dspy.ChainOfThought)
            base_model = self.mlflow_data_helper.model_by_args(load_model_args)
        else:
            base_model = self.prompt.infer

        # make sure the optimizer_kwargs include the student, overwriting whatever was provided if necessary
        self.optimizer_kwargs["student"] = base_model

        # run the trainer
        optimized_model = optimizer_compile(self.optimizer_type, self.optimizer_kwargs)

        # evaluate the training
        base_evaluation, optimized_evaluation = self.evaluate_training(
            base_model, optimized_model
        )

        # log the prompts
        base_signature = DspyDataHelper.prompt_signature(base_model)
        optimized_signature = DspyDataHelper.prompt_signature(optimized_model)

        base_prompt = DspyDataHelper.formatted_signature(
            base_signature, QualityDataHelper.example_example()
        )
        optimized_prompt = DspyDataHelper.formatted_signature(
            optimized_signature, QualityDataHelper.example_example()
        )

        mlflow.log_text(base_prompt, "base_prompt.txt")
        mlflow.log_text(optimized_prompt, "optimized_prompt.txt")

        # save the model
        if save_model:
            model_signature = QualityDataHelper.model_signature()
            self.mlflow_data_helper.save_model(
                optimized_model, model_signature, self.mlflow_model_name
            )

        # compare the models
        if self.prompt.compare_metrics(base_evaluation, optimized_evaluation):
            log.info(
                "Model training successful, optimized model performs better than base model"
            )
        else:
            log.info("Trained model did not improve on base model")
        return optimized_model
