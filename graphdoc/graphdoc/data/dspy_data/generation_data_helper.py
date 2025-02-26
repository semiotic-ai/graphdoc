# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging

# internal packages
from typing import Any, Optional, Union
from .dspy_data_helper import DspyDataHelper

# external packages
import dspy
from datasets import Dataset
from mlflow.models import ModelSignature, infer_signature

# logging
log = logging.getLogger(__name__)


class GenerationDataHelper(DspyDataHelper):
    """
    A helper class for creating data objects related to our Documentation Generation dspy.Signature.

    The example signature is defined as:
    ```
    database_schema: str = dspy.InputField()
    documented_schema: str = dspy.OutputField()
    ```
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def example(inputs: dict[str, Any]) -> dspy.Example:
        return dspy.Example(
            database_schema=inputs.get("database_schema", ""),
            documented_schema=inputs.get("documented_schema", ""),
        ).with_inputs("database_schema")

    @staticmethod
    def example_example() -> dspy.Example:
        return GenerationDataHelper.example(
            {
                "database_schema": "test database schema",
                "documented_schema": "test documented schema",
            }
        )

    @staticmethod
    def model_signature() -> ModelSignature:
        # TODO: decide if this should be here or in the mlflow_data_helper
        example = GenerationDataHelper.example_example().toDict()
        example.pop("documented_schema")
        return infer_signature(example)

    @staticmethod
    def prediction(inputs: dict[str, Any]) -> dspy.Prediction:
        return dspy.Prediction(
            database_schema=inputs.get("database_schema", ""),
            documented_schema=inputs.get("documented_schema", ""),
        )

    @staticmethod
    def prediction_example() -> dspy.Prediction:
        return GenerationDataHelper.prediction(
            {
                "database_schema": "test database schema",
                "documented_schema": "test documented schema",
            }
        )

    @staticmethod
    def trainset(
        inputs: Union[dict[str, Any], Dataset],
        filter_args: Optional[dict[str, Any]] = None,
    ) -> list[dspy.Example]:
        if isinstance(inputs, dict):
            # TODO: implement this
            raise NotImplementedError("from dictionary is not implemented")
        if isinstance(inputs, Dataset):
            # TODO: here is where we will want to enable post-processing of the inputs
            examples = []
            for i in range(len(inputs)):
                item = inputs[i]
                database_schema = item.get("schema_str", None)
                documented_schema = item.get("schema_str", None)
                if database_schema is None or documented_schema is None:
                    raise ValueError(
                        f"dataset item {i} is missing one or more required fields"
                    )
                example_dict = {
                    "database_schema": database_schema,
                    "documented_schema": documented_schema,
                }
                examples.append(GenerationDataHelper.example(example_dict))
            return examples
        raise ValueError(
            f"inputs must be a dictionary or a datasets, not: {type(inputs)}"
        )
