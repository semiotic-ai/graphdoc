# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# useful tutorial on dspy signatures: https://dspy.ai/tutorials/multihop_search/

# system packages
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Union

# external packages
import dspy

# internal packages


# logging
log = logging.getLogger(__name__)


class SinglePrompt(ABC):
    def __init__(
        self,
        prompt: Union[dspy.Signature, dspy.SignatureMeta],
        prompt_type: Union[Literal["predict", "chain_of_thought"], Callable],
        prompt_metric: Any,
        # TODO: we should consider adding a DspyDataHelper object here for convenience
        # and tighter coupling
    ) -> None:
        """Initialize a single prompt.

        :no-index:

        :param prompt: The prompt to use.
        :type prompt: dspy.Signature
        :param prompt_type: The type of prompt to use. Can be "predict" or
            "chain_of_thought". Optionally, pass another dspy.Module.
        :type prompt_type: Union[Literal["predict", "chain_of_thought"], Callable]
        :param prompt_metric: The metric to use. Marked as Any for flexibility (as
            metrics can be other prompts).
        :type prompt_metric: Any

        """
        self.prompt = prompt
        self.prompt_type = prompt_type
        self.prompt_metric = prompt_metric

        module_mapping = {
            "predict": dspy.Predict,
            "chain_of_thought": dspy.ChainOfThought,
        }
        # functools.singledispatch - less oop approach
        # oop: make two classes for passing the callable
        if self.prompt_type in module_mapping:
            self.infer = module_mapping[self.prompt_type](
                self.prompt
            )  # .get and then we can remove the error
        elif isinstance(self.prompt_type, Callable):
            log.warning(
                "Using alternative dspy.Module for inference, please know what you are doing"
            )
            self.infer = self.prompt_type(self.prompt)
        else:
            raise ValueError(f"Invalid prompt type: {self.prompt_type}")

    #######################################
    # methods for evaluating the prompt   #
    #######################################
    @abstractmethod
    def evaluate_metric(
        self, example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> Any:
        """This is the metric used to evalaute the prompt.

        :param example: The example to evaluate the metric on.
        :type example: dspy.Example
        :param prediction: The prediction to evaluate the metric on.
        :type prediction: dspy.Prediction
        :param trace: The trace to evaluate the metric on. This is for DSPy.
        :type trace: Any

        """
        pass

    @abstractmethod
    def format_metric(
        self,
        examples: List[dspy.Example],
        overall_score: float,
        results: List,
        scores: List,
    ) -> Dict[str, Any]:
        """This takes the results from the evaluate_evalset and does any necessary
        formatting, taking into account the metric type.

        :param examples: The examples to evaluate the metric on.
        :type examples: List[dspy.Example]
        :param overall_score: The overall score of the metric.
        :type overall_score: float
        :param results: The results from the evaluate_evalset.
        :type results: List
        :param scores: The scores from the evaluate_evalset.
        :type scores: List

        """
        pass

    @abstractmethod
    def compare_metrics(
        self,
        base_metrics: Any,
        optimized_metrics: Any,
        comparison_value: str = "overall_score",
    ) -> bool:
        """Compare the metrics of the base and optimized models. Return true if the
        optimized model is better than the base model.

        :param base_metrics: The metrics of the base model.
        :type base_metrics: Any
        :param optimized_metrics: The metrics of the optimized model.
        :type optimized_metrics: Any
        :param comparison_value: The value to compare the metrics on.
                                 Determines which metric is used to compare the models.
        :type comparison_value: str
        :return: True if the optimized model is better than the base model, False otherwise.
        :rtype: bool

        """
        pass

    def evaluate_evalset(
        self,
        examples: List[dspy.Example],
        num_threads: int = 1,
        display_progress: bool = True,
        display_table: bool = True,
    ) -> Dict[str, Any]:
        """Take in a list of examples and evaluate the results.

        :param examples: The examples to evaluate the results on.
        :type examples: List[dspy.Example]
        :param num_threads: The number of threads to use for evaluation.
        :type num_threads: int
        :param display_progress: Whether to display the progress of the evaluation.
        :type display_progress: bool
        :param display_table: Whether to display the table of the evaluation.
        :type display_table: bool
        :return: A dictionary containing the overall score, results, and scores.
        :rtype: Dict[str, Any]

        """
        evaluator = dspy.Evaluate(
            devset=examples,
            num_threads=num_threads,
            display_progress=display_progress,
            display_table=display_table,
            return_all_scores=True,
            return_outputs=True,
        )
        try:
            overall_score, results, scores = evaluator(self.infer, self.evaluate_metric)  # type: ignore
            return self.format_metric(examples, overall_score, results, scores)  # type: ignore
        except Exception as e:
            log.error("Error evaluating evalset: " + str(e))
            raise e
