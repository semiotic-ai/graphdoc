# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

# external packages
import dspy
import mlflow

from graphdoc.data import DspyDataHelper, GenerationDataHelper

# internal packages
from graphdoc.prompts import DocGeneratorPrompt
from graphdoc.train.optimizers import optimizer_compile
from graphdoc.train.single_prompt_trainer import SinglePromptTrainer

# logging
log = logging.getLogger(__name__)


class DocGeneratorTrainer(SinglePromptTrainer):
    def __init__(
        self,
        prompt: DocGeneratorPrompt,
        optimizer_type: str,
        optimizer_kwargs: Dict[str, Any],
        mlflow_model_name: str,
        mlflow_experiment_name: str,
        mlflow_tracking_uri: str,
        trainset: List[dspy.Example],
        evalset: List[dspy.Example],
    ):
        """Initialize the DocGeneratorTrainer.

        :param prompt: The prompt to train.
        :type prompt: DocGeneratorPrompt
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
        :param evalset: The evaluation set.
        :type evalset: List[dspy.Example]

        """
        super().__init__(
            prompt,
            optimizer_type,
            optimizer_kwargs,
            mlflow_model_name,
            mlflow_experiment_name,
            mlflow_tracking_uri,
            trainset,
            evalset,
        )
        # Cast to DocGeneratorPrompt for type checking
        if not isinstance(prompt, DocGeneratorPrompt):
            raise TypeError(f"Expected DocGeneratorPrompt, got {type(prompt)}")
        self.doc_generator_prompt = prompt

    def _calculate_average_score(self, evaluation: dict) -> float:
        """Given a dictionary of evaluation results, calculate the average score.

        :param evaluation: The evaluation results.
        :type evaluation: Dict[str, Any]
        :return: The average score.
        :rtype: float

        """
        examples = evaluation["results"]
        total = 0
        for ex in examples:
            rating = math.sqrt(ex[2]) * 25
            total += rating
        return round(total / len(examples), 6)

    def evaluation_metrics(
        self, base_evaluation: Dict[str, Any], optimized_evaluation: Dict[str, Any]
    ) -> None:
        """Log evaluation metrics to mlflow.

        :param base_evaluation: The evaluation metrics of the base model.
        :type base_evaluation: Dict[str, Any]
        :param optimized_evaluation: The evaluation metrics of the optimized model.
        :type optimized_evaluation: Dict[str, Any]

        """
        base_evaluation_overall_score = self._calculate_average_score(base_evaluation)
        optimized_evaluation_overall_score = self._calculate_average_score(
            optimized_evaluation
        )

        mlflow.log_metrics(
            {
                "base_evaluation_overall_score": base_evaluation_overall_score,
                "optimized_evaluation_overall_score": optimized_evaluation_overall_score,
            }
        )
        log.info(f"Base Evaluation: {base_evaluation}")
        log.info(f"Optimized Evaluation: {optimized_evaluation}")
        mlflow.log_dict(base_evaluation, "base_evaluation.json")
        mlflow.log_dict(optimized_evaluation, "optimized_evaluation.json")

    def evaluate_training(
        self, base_model, optimized_model
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Evaluate the training of the model. Comparing the base and optimized models.

        :param base_model: The base model.
        :type base_model: Any
        :param optimized_model: The optimized model.
        :type optimized_model: Any

        """
        base_prompt = DocGeneratorPrompt(
            prompt=DspyDataHelper.prompt_signature(base_model),
            prompt_type=self.doc_generator_prompt.prompt_type,  # type: ignore
            # TODO: we should have better type checking here
            prompt_metric=self.doc_generator_prompt.prompt_metric,  # type: ignore
            # TODO: we should have better type checking here
        )

        optimized_prompt = DocGeneratorPrompt(
            prompt=DspyDataHelper.prompt_signature(optimized_model),
            prompt_type=self.doc_generator_prompt.prompt_type,  # type: ignore
            # TODO: we should have better type checking here
            prompt_metric=self.doc_generator_prompt.prompt_metric,  # type: ignore
            # TODO: we should have better type checking here
        )

        base_evaluation = base_prompt.evaluate_evalset(self.evalset)
        optimized_evaluation = optimized_prompt.evaluate_evalset(self.evalset)

        self.evaluation_metrics(base_evaluation, optimized_evaluation)
        return base_evaluation, optimized_evaluation

    def train(
        self, load_model_args: Optional[Dict[str, Any]] = None, save_model: bool = True
    ):
        """Train the model. If load_model_args is provided, load the model from MLFlow.

        :param load_model_args: The arguments to load the model from mlflow.
        :type load_model_args: Optional[Dict[str, Any]]
        :param save_model: Whether to save the model to mlflow.
        :type save_model: bool

        """
        # if model args are provided, load the model from mlflow
        if load_model_args:
            base_model = self.mlflow_data_helper.model_by_args(load_model_args)
        else:
            base_model = self.doc_generator_prompt.infer

        # make sure the optimizer_kwargs include the student,
        # overwriting whatever was provided if necessary
        self.optimizer_kwargs["student"] = base_model

        # run the optimizer
        log.info(f"Running {self.optimizer_type} optimizer...")
        optimized_model = optimizer_compile(self.optimizer_type, self.optimizer_kwargs)

        # evaluate the training
        base_evaluation, optimized_evaluation = self.evaluate_training(
            base_model, optimized_model
        )

        # log the prompts
        base_signature = DspyDataHelper.prompt_signature(base_model)
        optimized_signature = DspyDataHelper.prompt_signature(optimized_model)

        base_prompt = DspyDataHelper.formatted_signature(
            base_signature, GenerationDataHelper.example_example()
        )
        optimized_prompt = DspyDataHelper.formatted_signature(
            optimized_signature, GenerationDataHelper.example_example()
        )

        mlflow.log_text(base_prompt, "base_prompt.txt")
        mlflow.log_text(optimized_prompt, "optimized_prompt.txt")

        # save the model
        if save_model:
            model_signature = GenerationDataHelper.model_signature()
            self.mlflow_data_helper.save_model(
                optimized_model, model_signature, self.mlflow_model_name
            )

        # compare the models
        if self.doc_generator_prompt.compare_metrics(
            base_evaluation, optimized_evaluation
        ):  # TODO: we should enable the passing of different comparison metrics
            log.info(
                "Model training successful, optimized model performs better than base model"
            )
        else:
            log.info("Trained model did not improve on base model")

        return optimized_model
