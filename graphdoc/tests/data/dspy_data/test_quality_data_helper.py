# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging

# internal packages
from graphdoc import QualityDataHelper

# external packages
import dspy
import pytest
from datasets import Dataset
from mlflow.models import ModelSignature

# logging
log = logging.getLogger(__name__)


class TestQualityDataHelper:
    def test_example(self):
        inputs = {
            "database_schema": "example database schema",
            "category": "perfect",
            "rating": 4,
        }
        example = QualityDataHelper.example(inputs)
        assert isinstance(example, dspy.Example)

    def test_example_example(self):
        example = QualityDataHelper.example_example()
        assert isinstance(example, dspy.Example)
        assert example.database_schema == "test database schema"
        assert example.category == "perfect"
        assert example.rating == 4

    def test_model_signature(self):
        signature = QualityDataHelper.model_signature()
        assert isinstance(signature, ModelSignature)

    def test_prediction(self):
        inputs = {
            "database_schema": "example database schema",
            "category": "perfect",
            "rating": 4,
        }
        prediction = QualityDataHelper.prediction(inputs)
        assert isinstance(prediction, dspy.Prediction)

    def test_prediction_example(self):
        prediction = QualityDataHelper.prediction_example()
        assert isinstance(prediction, dspy.Prediction)
        assert prediction.database_schema == "test database schema"
        assert prediction.category == "perfect"
        assert prediction.rating == 4

    def test_trainset(self):
        with pytest.raises(NotImplementedError):
            QualityDataHelper.trainset({})

    def test_trainset_from_dataset(self):
        dataset = Dataset.from_dict(
            {
                "schema_str": ["example database schema"],
                "category": ["perfect"],
                "rating": [4],
            }
        )
        trainset = QualityDataHelper.trainset(dataset)
        assert isinstance(trainset, list)
        assert len(trainset) == 1
        assert isinstance(trainset[0], dspy.Example)
        assert trainset[0].database_schema == "example database schema"
        assert trainset[0].category == "perfect"
        assert trainset[0].rating == 4
