# system packages
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional, Union

# internal packages

# external packages
import dspy
from datasets import Dataset

# logging
log = logging.getLogger(__name__)


class DspyDataHelper(ABC):
    """
    Abstract class for creating data objects related to a given dspy.Signature.
    """

    def __init__(self):
        # TODO: we should consider adding a signature object here
        pass

    #######################
    # Abstract Methods    #
    #######################
    @staticmethod
    @abstractmethod
    def example(inputs: dict[str, Any]) -> dspy.Example:
        pass

    @staticmethod
    @abstractmethod
    def example_example(inputs: dict[str, Any] = {}) -> dspy.Example:
        pass

    @staticmethod
    @abstractmethod
    def prediction(inputs: dict[str, Any]) -> dspy.Prediction:
        pass

    @staticmethod
    @abstractmethod
    def prediction_example(inputs: dict[str, Any] = {}) -> dspy.Prediction:
        pass

    @staticmethod
    @abstractmethod
    def trainset(inputs: Union[dict[str, Any], Dataset]) -> list[dspy.Example]:
        pass
