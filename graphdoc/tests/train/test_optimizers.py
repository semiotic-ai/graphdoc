# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging

# external packages
import dspy
import pytest

# internal packages
from graphdoc import (
    DocQualityPrompt,
    optimizer_class,
)

# logging
log = logging.getLogger(__name__)


class TestOptimizers:
    def test_optimizer_class_miprov2(self, dqp: DocQualityPrompt):
        optimizer_kwargs = {
            "metric": dqp.evaluate_metric,
            "auto": "light",
        }
        optimizer = optimizer_class("miprov2", optimizer_kwargs)
        assert isinstance(optimizer, dspy.MIPROv2)

    def test_optimizer_class_bootstrap_few_shot(self, dqp: DocQualityPrompt):
        optimizer_kwargs = {
            "metric": dqp.evaluate_metric,
            "teacher_settings": {},
            "max_bootstrapped_demos": 4,
            "max_labeled_demos": 16,
            "max_rounds": 1,
            "num_candidate_programs": 16,
            "num_threads": 6,
            "max_errors": 10,
            "stop_at_score": None,
            "metric_threshold": None,
        }
        optimizer = optimizer_class(
            "BootstrapFewShotWithRandomSearch", optimizer_kwargs
        )
        assert isinstance(optimizer, dspy.BootstrapFewShotWithRandomSearch)

    def test_optimizer_class_invalid_type(self):
        with pytest.raises(ValueError, match="Invalid optimizer type: invalid_type"):
            optimizer_class("invalid_type", {})

    # TODO: it would be nice to make this optional based on a flag
    # def test_optimizer_compile_miprov2(self, gd: GraphDoc, dqp: DocQualityPrompt):
    #     trainset = [QualityDataHelper.example_example() for _ in range(10)]
    #     optimizer_kwargs = {
    #         "metric": dqp.evaluate_metric,
    #         "auto": "light",
    #         "student": dspy.ChainOfThought(DocQualitySignature),
    #         "trainset": trainset,
    #         "max_labeled_demos": 10,
    #         "max_bootstrapped_demos": 5,
    #     }
    #     optimizer = optimizer_compile("miprov2", optimizer_kwargs)
    #     assert optimizer is not None

    # def test_optimizer_compile_bootstrap_few_shot(
    #     self, gd: GraphDoc, dqp: DocQualityPrompt
    # ):
    #     trainset = [QualityDataHelper.example_example() for _ in range(10)]
    #     optimizer_kwargs = {
    #         "metric": dqp.evaluate_metric,
    #         "teacher_settings": {},
    #         "max_bootstrapped_demos": 4,
    #         "max_labeled_demos": 16,
    #         "max_rounds": 1,
    #         "num_candidate_programs": 16,
    #         "num_threads": 6,
    #         "max_errors": 10,
    #         "stop_at_score": None,
    #         "metric_threshold": None,
    #         "student": dspy.ChainOfThought(DocQualitySignature),
    #         "trainset": trainset,
    #     }
    #     optimizer = optimizer_compile(
    #         "BootstrapFewShotWithRandomSearch", optimizer_kwargs
    #     )
    #     assert optimizer is not None
