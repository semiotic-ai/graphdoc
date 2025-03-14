# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List, Literal, Union

# external packages
import dspy

# internal packages
from graphdoc.prompts.single_prompt import SinglePrompt

# logging
log = logging.getLogger(__name__)


###################
# DSPy Signatures #
###################
class DocQualitySignature(dspy.Signature):
    """
    You are a documentation quality evaluator specializing in GraphQL schemas. Your task is to assess the quality of documentation provided for a given database schema. Carefully analyze the schema's descriptions for clarity, accuracy, and completeness. Categorize the documentation into one of the following ratings based on your evaluation:
    - perfect (4): The documentation is comprehensive and leaves no room for ambiguity in understanding the schema and its database content.
    - almost perfect (3): The documentation is clear and mostly free of ambiguity, but there is potential for further improvement.
    - poor but correct (2): The documentation is correct but lacks detail, resulting in some ambiguity. It requires enhancement to be more informative.
    - incorrect (1): The documentation contains errors or misleading information, regardless of any correct segments present. Such inaccuracies necessitate an incorrect rating.
    Provide a step-by-step reasoning to support your evaluation, along with the appropriate category label and numerical rating.
    """  # noqa: B950

    database_schema: str = dspy.InputField()
    category: Literal[
        "perfect", "almost perfect", "poor but correct", "incorrect"
    ] = dspy.OutputField()
    rating: Literal[4, 3, 2, 1] = dspy.OutputField()


class DocQualityDemonstrationSignature(dspy.Signature):
    """You are evaluating the output of an LLM program, expect hallucinations. Given a GraphQL Schema, evaluate the quality of documentation for that schema and provide a category rating.

    The categories are described as:
    - perfect (4): The documentation contains enough information so that the interpretation of the schema and its database content is completely free of ambiguity.
        perfect (4) example:
        type Domain @entity {
            " The namehash (id) of the parent name. References the Domain entity that is the parent of the current domain. Type: Domain "
            parent: Domain
        }
    - almost perfect (3): The documentation is almost perfect and free from ambiguity, but there is room for improvement.
        almost perfect (3) example:
        type Token @entity {
            " Name of the token, mirrored from the smart contract "
            name: String!
        }
    - poor but correct (2): The documentation is poor but correct and has room for improvement due to missing information. The documentation is not incorrect.
        poor but correct (2) example:
        type InterestRate @entity {
            "Description for column: id"
            id: ID!
        }
    - incorrect (1): The documentation is incorrect and contains inaccurate or misleading information. Any incorrect information automatically leads to an incorrect rating, even if some correct information is present.
        incorrect (1) example:
        type BridgeProtocol implements Protocol @entity {
            " Social Security Number of the protocol's main developer "
            id: Bytes!
        }
    Output a number rating that corresponds to the categories described above.

    """  # noqa: B950

    database_schema: str = dspy.InputField()
    category: Literal[
        "perfect", "almost perfect", "poor but correct", "incorrect"
    ] = dspy.OutputField()
    rating: Literal[4, 3, 2, 1] = dspy.OutputField()


def doc_quality_factory(
    key: Union[str, dspy.Signature, dspy.SignatureMeta]
) -> Union[dspy.Signature, dspy.SignatureMeta]:
    """Factory function to return the correct signature based on the key. Currently only
    supports two signatures (doc_quality and doc_quality_demo).

    :param key: The key to return the signature for.
    :type key: Union[str, dspy.Signature]
    :return: The signature for the given key.

    """
    # allow the user to pass in their own dspy signature
    if isinstance(key, dspy.Signature) or isinstance(key, dspy.SignatureMeta):
        return key
    factory = {
        "doc_quality": DocQualitySignature,
        "doc_quality_demo": DocQualityDemonstrationSignature,
    }
    signature = factory.get(key, None)
    if signature is None:
        raise ValueError(f"Invalid signature (type: {type(key)}): {key}")
    return signature


#######################
# Single Prompt Class #
#######################
class DocQualityPrompt(SinglePrompt):
    """DocQualityPrompt class for evaluating documentation quality.

    This is a single prompt that can be used to evaluate the quality of the documentation
    for a given schema. This is a wrapper around the SinglePrompt class that implements
    the abstract methods.
    """

    def __init__(
        self,
        prompt: Union[
            Literal["doc_quality", "doc_quality_demo"],
            dspy.Signature,
            dspy.SignatureMeta,
        ] = "doc_quality",
        prompt_type: Union[
            Literal["predict", "chain_of_thought"], Callable
        ] = "predict",
        prompt_metric: Union[Literal["rating", "category"], Callable] = "rating",
    ) -> None:
        # TODO: we should think about if we want to add checks on any provided dspy.Signature
        """Initialize the DocQualityPrompt.

        :param prompt: The prompt to use. Can either be a string that maps to a defined
          signature, as set in the doc_quality_factory, or a dspy.Signature.
        :type prompt: Union[str, dspy.Signature]
        :param prompt_type: The type of prompt to use.
        :type prompt_type: Union[Literal["predict", "chain_of_thought"], Callable]
        :param prompt_metric: The metric to use. Can either be a string that maps to a defined
          metric, as set in the doc_quality_factory, or a custom callable function.
          Function must have the signature (example: dspy.Example, prediction:
          dspy.Prediction) -> bool.
        :type prompt_metric: Union[Literal["rating", "category"], Callable]
        """
        prompt_signature = doc_quality_factory(prompt)
        super().__init__(
            prompt=prompt_signature,
            prompt_type=prompt_type,
            prompt_metric=prompt_metric,
        )

    #######################
    # Internal Methods    #
    #######################
    def _evaluate_rating_metric(
        self, example: dspy.Example, prediction: dspy.Prediction
    ) -> bool:
        return example.rating == prediction.rating

    def _evaluate_category_metric(
        self, example: dspy.Example, prediction: dspy.Prediction
    ) -> bool:
        return example.category == prediction.category

    #######################
    # Abstract Methods    #
    #######################
    def evaluate_metric(
        self, example: dspy.Example, prediction: dspy.Prediction, trace=None
    ) -> bool:
        """Evaluate the metric for the given example and prediction.

        :param example: The example to evaluate the metric on.
        :type example: dspy.Example
        :param prediction: The prediction to evaluate the metric on.
        :type prediction: dspy.Prediction
        :param trace: Used for DSPy.
        :type trace: Any
        :return: The result of the evaluation. A boolean for if the metric is correct.
        :rtype: bool
        """
        evaluation_mapping = {
            "rating": self._evaluate_rating_metric,
            "category": self._evaluate_category_metric,
        }
        if isinstance(self.prompt_metric, str):
            evaluation_function = evaluation_mapping.get(self.prompt_metric)
            if evaluation_function is None:
                raise ValueError(f"Invalid metric type: {self.prompt_metric}")
        else:
            evaluation_function = self.prompt_metric
        return evaluation_function(example, prediction)

    def format_metric(
        self,
        examples: List[dspy.Example],
        overall_score: float,
        results: List,
        scores: List,
    ) -> Dict[str, Any]:
        """
        Formats evaluation metrics into a structured report containing:
        - Overall score across all categories
        - Percentage correct per category
        - Detailed results for each evaluation

        :param examples: The examples to evaluate the metric on.
        :type examples: List[dspy.Example]
        :param overall_score: The overall score across all categories.
        :type overall_score: float
        :param results: The results of the evaluation.
        :type results: List
        :param scores: The scores of the evaluation.
        :type scores: List
        :return: A dictionary containing the overall score, per category scores, and details.
                { "overall_score": 0, "per_category_scores": {}, "details": [], "results": [] }
        :rtype: Dict[str, Any]
        """

        def _initialize_formatted_results() -> Dict[str, Any]:
            """Initialize the results structure with empty containers."""
            return {
                "overall_score": overall_score,
                "per_category_scores": {},
                "details": [],
                "results": results,
            }

        def _process_single_result(result: tuple, score: Any) -> Dict[str, Any]:
            """Process individual result to extract metadata and update statistics."""
            example, prediction, is_correct = result
            example_data = dict(example.items())

            expected_category = example_data.get("category", "unknown")
            expected_rating = example_data.get("rating", None)
            predicted_category = getattr(prediction, "category", "unknown")
            predicted_rating = getattr(prediction, "rating", None)

            category_stats[expected_category]["total"] += 1
            if is_correct:
                category_stats[expected_category]["correct"] += 1

            return {
                "expected_category": expected_category,
                "expected_rating": expected_rating,
                "predicted_category": predicted_category,
                "predicted_rating": predicted_rating,
                "is_correct": is_correct,
            }

        def _calculate_percent_correct(correct: int, total: int) -> float:
            """Calculate percentage correct with safe division."""
            return (correct / total) * 100 if total > 0 else 0.0

        def _calculate_per_category_scores() -> Dict[str, Dict]:
            """Convert category statistics to percentage scores."""
            return {
                category: {
                    "percent_correct": _calculate_percent_correct(
                        stats["correct"], stats["total"]
                    ),
                    "total": stats["total"],
                    "correct": stats["correct"],
                }
                for category, stats in category_stats.items()
            }

        # processing flow
        formatted_results = _initialize_formatted_results()
        category_stats = defaultdict(lambda: {"correct": 0, "total": 0})

        # process all results and collect details
        formatted_results["details"] = [
            _process_single_result(result, score)
            for result, score in zip(results, scores)
        ]

        # calculate final scores per category
        formatted_results["per_category_scores"] = _calculate_per_category_scores()

        return formatted_results

    def compare_metrics(
        self,
        base_metrics: Any,
        optimized_metrics: Any,
        comparison_value: str = "overall_score",
    ) -> bool:
        """Compare the metrics of the base and optimized models. Returns true if the
        optimized model is better than the base model.

        :param base_metrics: The metrics of the base model.
        :type base_metrics: Any
        :param optimized_metrics: The metrics of the optimized model.
        :type optimized_metrics: Any
        :param comparison_value: The value to compare.
        :type comparison_value: str
        :return: True if the optimized model is better than the base model.
        :rtype: bool
        """
        if comparison_value == "overall_score":
            return optimized_metrics["overall_score"] > base_metrics["overall_score"]
        else:
            raise ValueError(f"Invalid comparison value: {comparison_value}")
