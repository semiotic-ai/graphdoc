# system packages
import logging

# internal packages
from typing import Any, Optional, Union
from .dspy_data_helper import DspyDataHelper

# external packages
import dspy
from datasets import Dataset

# logging
log = logging.getLogger(__name__)


class QualityDataHelper(DspyDataHelper):
    """
    A helper class for creating data objects related to our Documentation Quality dspy.Signature.

    The example signature is defined as:
    ```
    database_schema: str = dspy.InputField()
    category: Literal["perfect", "almost perfect", "poor but correct", "incorrect"] = (
        dspy.OutputField()
    )
    rating: Literal[4, 3, 2, 1] = dspy.OutputField()
    ```
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def example(inputs: dict[str, Any]) -> dspy.Example:
        return dspy.Example(
            database_schema=inputs.get("database_schema", ""),
            category=inputs.get("category", ""),
            rating=inputs.get("rating", 0),
        ).with_inputs("database_schema")

    @staticmethod
    def example_example(inputs: dict[str, Any] = {}) -> dspy.Example:
        return dspy.Example(
            database_schema=inputs.get("database_schema", "test database schema"),
            category=inputs.get("category", "perfect"),
            rating=inputs.get("rating", 4),
        ).with_inputs("database_schema")

    @staticmethod
    def prediction(inputs: dict[str, Any]) -> dspy.Prediction:
        return dspy.Prediction(
            database_schema=inputs.get("database_schema", ""),
            category=inputs.get("category", ""),
            rating=inputs.get("rating", 0),
        )

    @staticmethod
    def prediction_example(inputs: dict[str, Any] = {}) -> dspy.Prediction:
        return dspy.Prediction(
            database_schema=inputs.get("database_schema", "test database schema"),
            category=inputs.get("category", "perfect"),
            rating=inputs.get("rating", 4),
        )

    @staticmethod
    def trainset(inputs: Union[dict[str, Any], Dataset]) -> list[dspy.Example]:
        # TODO: implement this
        raise NotImplementedError("trainset is not implemented")
