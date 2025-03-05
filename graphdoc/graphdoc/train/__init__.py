# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
from typing import Any, Dict, List

# external packages
import dspy

from .doc_generator_trainer import DocGeneratorTrainer
from .doc_quality_trainer import DocQualityTrainer

# internal packages
from .optimizers import _optimizer_kwargs_filter, optimizer_class, optimizer_compile
from .single_prompt_trainer import SinglePrompt, SinglePromptTrainer

# logging
log = logging.getLogger(__name__)


class TrainerFactory:
    @staticmethod
    def single_trainer(
        trainer_class: str,
        prompt: SinglePrompt,
        optimizer_type: str,
        optimizer_kwargs: Dict[str, Any],
        mlflow_tracking_uri: str,
        mlflow_model_name: str,
        mlflow_experiment_name: str,
        trainset: List[dspy.Example],
        evalset: List[dspy.Example],
    ):
        """Returns an instance of the specified trainer class."""
        # update any potentially missing or conflicting values
        optimizer_kwargs["metric"] = prompt.evaluate_metric
        optimizer_kwargs["student"] = prompt.infer
        optimizer_kwargs["trainset"] = trainset

        trainer_classes = {
            "DocQualityTrainer": DocQualityTrainer,
            "DocGeneratorTrainer": DocGeneratorTrainer,
        }
        if trainer_class not in trainer_classes:
            raise ValueError(f"Unknown trainer class: {trainer_class}")

        try:
            return trainer_classes[trainer_class](
                prompt=prompt,  # type: ignore # TODO: we should have better type checking here
                optimizer_type=optimizer_type,
                optimizer_kwargs=optimizer_kwargs,
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_model_name=mlflow_model_name,
                mlflow_experiment_name=mlflow_experiment_name,
                trainset=trainset,
                evalset=evalset,
            )
        except Exception as e:
            raise ValueError(
                f"Failed to initialize trainer class ({trainer_class}): {e}"
            )
