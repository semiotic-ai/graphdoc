# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
from typing import Callable

# external packages
import dspy
import pytest

# internal packages
from graphdoc import (
    DocGeneratorPrompt,
    DocGeneratorSignature,
    DocQualityPrompt,
    GenerationDataHelper,
    GraphDoc,
)

# logging
log = logging.getLogger(__name__)


class TestDocGeneratorPrompt:
    def test_doc_generator_prompt(self, dgp):
        assert isinstance(dgp, DocGeneratorPrompt)
        assert dgp.prompt_type == "chain_of_thought"
        assert isinstance(dgp.prompt_metric, DocQualityPrompt)

    def test_evaluate_documentation_quality(self, dgp):
        example = GenerationDataHelper.example_example()
        prediction = GenerationDataHelper.prediction_example()
        score = dgp.evaluate_documentation_quality(example, prediction)
        assert isinstance(score, int)
        assert score >= 0
        assert score <= 4

    def test_format_metric(self, dgp):
        examples = GenerationDataHelper.example_example()
        overall_score = 0.5
        results = ["result1", "result2", "result3"]
        scores = [0.5, 0.5, 0.5]
        formatted_metric = dgp.format_metric(examples, overall_score, results, scores)
        assert isinstance(formatted_metric, dict)
        assert formatted_metric["overall_score"] == overall_score
        assert len(formatted_metric["scores"]) == len(scores)
        assert len(formatted_metric["results"]) == len(results)

    def test_compare_metrics(self, dgp):
        base_metrics = {"overall_score": 0.5}
        optimized_metrics = {"overall_score": 0.7}
        assert dgp.compare_metrics(base_metrics, optimized_metrics) is True
        assert (
            dgp.compare_metrics(base_metrics, optimized_metrics, "overall_score")
            is True
        )
