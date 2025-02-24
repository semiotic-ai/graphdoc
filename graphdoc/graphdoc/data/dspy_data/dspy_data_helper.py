# system packages
import logging
from abc import ABC, abstractmethod
from functools import singledispatch
from typing import Any, Optional, Union

# internal packages

# external packages
import dspy
from datasets import Dataset
from mlflow.models import ModelSignature

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
    # Class Methods       #
    #######################
    @singledispatch
    @staticmethod
    def prompt_signature(
        prompt: Any,
    ) -> Union[dspy.Signature, dspy.SignatureMeta, None]:
        """
        Given a prompt, return a dspy.Signature object.

        :param prompt: A prompt.
        :type prompt: Any
        """
        log.warning("No prompt signature found for the given prompt.")
        return None

    @prompt_signature.register(dspy.ChainOfThought)
    @staticmethod
    def _(prompt: dspy.ChainOfThought) -> Union[dspy.Signature, dspy.SignatureMeta]:
        """
        Given a dspy.ChainOfThought object, return a dspy.Signature object.
        """
        return prompt.predict.signature

    @prompt_signature.register(dspy.Predict)
    @staticmethod
    def _(prompt: dspy.Predict) -> Union[dspy.Signature, dspy.SignatureMeta]:
        """
        Given a dspy.Predict object, return a dspy.Signature object.
        """
        return prompt.signature

    #######################
    # Abstract Methods    #
    #######################
    @staticmethod
    @abstractmethod
    def example(inputs: dict[str, Any]) -> dspy.Example:
        """
        Given a dictionary of inputs, return a dspy.Example object.

        :param inputs: A dictionary of inputs.
        :type inputs: dict[str, Any]
        :return: A dspy.Example object.
        :rtype: dspy.Example
        """
        pass

    @staticmethod
    @abstractmethod
    def example_example() -> dspy.Example:
        """
        Return an example dspy.Example object with the inputs set to the example values.

        :return: A dspy.Example object.
        :rtype: dspy.Example
        """
        pass

    @staticmethod
    @abstractmethod
    def model_signature() -> ModelSignature:
        """
        Return a mlflow.models.ModelSignature object. Based on the example object, removes the output fields and utilizes the remaining fields to infer the model signature.

        :return: A mlflow.models.ModelSignature object.
        :rtype: mlflow.models.ModelSignature
        """
        # TODO: decide if this should be here or in the mlflow_data_helper
        pass

    @staticmethod
    @abstractmethod
    def prediction(inputs: dict[str, Any]) -> dspy.Prediction:
        """
        Given a dictionary of inputs, return a dspy.Prediction object.

        :param inputs: A dictionary of inputs.
        :type inputs: dict[str, Any]
        :return: A dspy.Prediction object.
        :rtype: dspy.Prediction
        """
        pass

    @staticmethod
    @abstractmethod
    def prediction_example() -> dspy.Prediction:
        """
        Return an example dspy.Prediction object with the inputs set to the example values.

        :return: A dspy.Prediction object.
        :rtype: dspy.Prediction
        """
        pass

    @staticmethod
    @abstractmethod
    def trainset(inputs: Union[dict[str, Any], Dataset]) -> list[dspy.Example]:
        """
        Given a dictionary of inputs or a datasets.Dataset object, return a list of dspy.Example objects.

        :param inputs: A dictionary of inputs or a datasets.Dataset object.
        :type inputs: Union[dict[str, Any], datasets.Dataset]
        :return: A list of dspy.Example objects.
        :rtype: list[dspy.Example]
        """
        pass
