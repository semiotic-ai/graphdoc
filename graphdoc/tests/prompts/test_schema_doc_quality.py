# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging

# external packages
import dspy

# internal packages
from graphdoc.prompts import DocQualityPrompt

# Configure logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class CustomPrompt(dspy.Signature):
    """This is a custom prompt."""

    input = dspy.InputField()
    output = dspy.OutputField()


def custom_metric(example: dspy.Example, prediction: dspy.Prediction) -> bool:
    return example.input == prediction.output


class TestDocQualityPrompt:
    def test_doc_quality_prompt(self):
        dqp = DocQualityPrompt(
            prompt="doc_quality", prompt_type="predict", prompt_metric="rating"
        )
        assert isinstance(dqp, DocQualityPrompt)
        assert isinstance(dqp.infer, dspy.Predict)
        assert dqp.prompt_metric == "rating"

        dqp = DocQualityPrompt(
            prompt="doc_quality_demo",
            prompt_type="chain_of_thought",
            prompt_metric="category",
        )
        assert isinstance(dqp, DocQualityPrompt)
        assert isinstance(dqp.infer, dspy.ChainOfThought)
        assert dqp.prompt_metric == "category"

        dqp = DocQualityPrompt(
            prompt=CustomPrompt,
            prompt_type="predict",
            prompt_metric=custom_metric,
        )
        assert isinstance(dqp, DocQualityPrompt)
        assert isinstance(dqp.infer, dspy.Predict)
        assert dqp.prompt_metric == custom_metric

    def test_evaluate_metric(self):
        dqp = DocQualityPrompt(
            prompt="doc_quality",
            prompt_type="predict",
            prompt_metric="rating",
        )
        example = dspy.Example(
            database_schema="this is a test",
            category="perfect",
            rating=4,
        )
        pass_prediction = dspy.Prediction(
            category="perfect",
            rating=4,
        )
        fail_prediction = dspy.Prediction(
            category="incorrect",
            rating=1,
        )
        assert dqp.evaluate_metric(example, pass_prediction) is True
        assert dqp.evaluate_metric(example, fail_prediction) is False

    def test_format_metric(self):
        dqp = DocQualityPrompt(
            prompt="doc_quality",
            prompt_type="predict",
            prompt_metric="rating",
        )
        examples = []
        overall_score = 0
        results = []
        scores = []
        formatted_results = dqp.format_metric(examples, overall_score, results, scores)
        assert formatted_results == {
            "overall_score": 0,
            "per_category_scores": {},
            "details": [],
            "results": [],
        }

    def test_compare_metrics(self):
        dqp = DocQualityPrompt(
            prompt="doc_quality",
            prompt_type="predict",
            prompt_metric="rating",
        )
        base_metrics = {
            "overall_score": 0,
            "per_category_scores": {},
            "details": [],
            "results": [],
        }
        optimized_metrics = {
            "overall_score": 1,
            "per_category_scores": {},
            "details": [],
            "results": [],
        }
        assert dqp.compare_metrics(base_metrics, optimized_metrics) is True
        assert dqp.compare_metrics(optimized_metrics, base_metrics) is False
