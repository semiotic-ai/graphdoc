# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
from pathlib import Path
from typing import Any, List, Union

# external packages
import dspy
import mlflow
from graphql import parse, print_ast

# internal packages
from ..data import MlflowDataHelper
from ..modules import DocGeneratorModule
from ..prompts import DocQualityPrompt, SinglePrompt

# logging
log = logging.getLogger(__name__)


class DocGeneratorEvaluator(dspy.Module):
    def __init__(
        self,
        generator: Union[
            DocGeneratorModule, dspy.Module, Any
        ],  # we have type hints, but accept any type for flexibility
        evaluator: Union[
            DocQualityPrompt, SinglePrompt, Any
        ],  # we have type hints, but accept any type for flexibility
        evalset: Union[List[dspy.Example], Any],
        mlflow_tracking_uri: Union[str, Path],
        mlflow_experiment_name: str = "doc_generator_eval",
        generator_prediction_field: str = "documented_schema",
        evaluator_prediction_field: str = "rating",
        readable_value: int = 25,
    ):
        """
        A simple module for evaluating the quality of generated documentation. We will make this extensible to include more complex evaluation metrics in the future.

        Important: we assume that the rating values returned by the evaluator are [1, 2, 3, 4]. We will make this more flexible in the future.
        """
        self.generator = generator
        self.evaluator = evaluator
        self.evalset = evalset
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.generator_prediction_field = generator_prediction_field
        self.evaluator_prediction_field = evaluator_prediction_field
        self.mlflow_helper = MlflowDataHelper(mlflow_tracking_uri)
        self.mlflow_experiment_name = mlflow_experiment_name
        self.readable_value = readable_value

    def forward(self, database_schema: str) -> dict[str, Any]:
        """
        Takes a database schema, documents it, and then evaluates each component and the aggregate.
        """
        generator_result = self.generator.forward(database_schema)  # type: ignore (we assume we are using DocGeneratorModule)
        # TODO: let's decide if this is how we want to handle this in the future. Alternatively, we could return the documented schema from forward, not as a prediction object.
        documented_schema = getattr(generator_result, self.generator_prediction_field)

        try:
            documented_ast = parse(documented_schema)
            component_ratings = []
            for node in documented_ast.definitions:
                p = self.evaluator.infer(database_schema=print_ast(node))
                # TODO: let's decide if this is how we want to handle this, or if we should standardize the return type of the evaluator.
                rating = getattr(p, self.evaluator_prediction_field)
                rating = rating * self.readable_value if rating != 1 else 0
                component_ratings.append(rating)

            overall_p = self.evaluator.infer(database_schema=documented_schema)
            overall_rating = getattr(overall_p, self.evaluator_prediction_field)
            overall_rating = (
                overall_rating * self.readable_value if overall_rating != 1 else 0
            )

            return {
                "overall_rating": overall_rating,
                "average_component_rating": sum(component_ratings)
                / len(component_ratings),
                "component_ratings": component_ratings,
            }

        except Exception as e:
            log.warning(f"Generated schema was not valid GraphQL: {e}")
            document_ast = parse(database_schema)
            # TODO: we should have a dynamic way of knowing the rating values that are being used by the evaluator.
            return {
                "overall_rating": 0,
                "average_component_rating": 0,
                "component_ratings": [0] * len(document_ast.definitions),
            }

    def evaluate(self):
        """
        Batches the evaluation set and logs the results to mlflow.
        """
        mlflow.set_experiment(self.mlflow_experiment_name)
        with mlflow.start_run():
            # evalset = [x.database_schema for x in self.evalset]
            evaluation_results = self.batch(self.evalset, num_threads=32)
            avg_overall_rating = sum(
                [x["overall_rating"] for x in evaluation_results]
            ) / len(evaluation_results)
            avg_component_rating = sum(
                [x["average_component_rating"] for x in evaluation_results]
            ) / len(evaluation_results)

            mlflow.log_metric("average_overall_rating", avg_overall_rating)
            mlflow.log_metric("average_component_rating", avg_component_rating)
            mlflow.log_dict(
                {
                    "component_ratings": [
                        x["component_ratings"] for x in evaluation_results
                    ]
                },
                "component_ratings.json",
            )

            # TODO: log the parameters so we can reproduce the results.
            # TODO: utilize huggingface datasets and commit hashes to enable reproducibility.
