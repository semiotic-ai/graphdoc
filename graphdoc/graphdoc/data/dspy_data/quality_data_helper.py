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
    def example_example() -> dspy.Example:
        return QualityDataHelper.example(
            {
                "database_schema": "test database schema",
                "category": "perfect",
                "rating": 4,
            }
        )

    @staticmethod
    def model_signature() -> ModelSignature:
        # TODO: decide if this should be here or in the mlflow_data_helper
        example = QualityDataHelper.example_example().toDict()
        example.pop("category")
        example.pop("rating")
        return infer_signature(example)

    @staticmethod
    def prediction(inputs: dict[str, Any]) -> dspy.Prediction:
        return dspy.Prediction(
            database_schema=inputs.get("database_schema", ""),
            category=inputs.get("category", ""),
            rating=inputs.get("rating", 0),
        )

    @staticmethod
    def prediction_example() -> dspy.Prediction:
        return QualityDataHelper.prediction(
            {
                "database_schema": "test database schema",
                "category": "perfect",
                "rating": 4,
            }
        )

    @staticmethod
    def trainset(inputs: Union[dict[str, Any], Dataset], filter_args: Optional[dict[str, Any]] = None) -> list[dspy.Example]:
        # TODO: implement this
        raise NotImplementedError("trainset is not implemented")
