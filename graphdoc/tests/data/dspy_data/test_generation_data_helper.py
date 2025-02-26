# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging

# internal packages
from graphdoc import GenerationDataHelper

# external packages
import dspy
import pytest
from datasets import Dataset
from mlflow.models import ModelSignature

# logging
log = logging.getLogger(__name__)


class TestGenerationDataHelper:
    def test_example(self):
        inputs = {
            "database_schema": "example database schema",
            "documented_schema": "example documented schema",
        }
        example = GenerationDataHelper.example(inputs)
        assert isinstance(example, dspy.Example)
        assert example.database_schema == "example database schema"
        assert example.documented_schema == "example documented schema"

    def test_example_example(self):
        example = GenerationDataHelper.example_example()
        assert isinstance(example, dspy.Example)
        assert example.database_schema == "test database schema"
        assert example.documented_schema == "test documented schema"

    def test_model_signature(self):
        signature = GenerationDataHelper.model_signature()
        assert isinstance(signature, ModelSignature)

    def test_prediction(self):
        inputs = {
            "database_schema": "example database schema",
            "documented_schema": "example documented schema",
        }
        prediction = GenerationDataHelper.prediction(inputs)
        assert isinstance(prediction, dspy.Prediction)
        assert prediction.database_schema == "example database schema"
        assert prediction.documented_schema == "example documented schema"

    def test_prediction_example(self):
        prediction = GenerationDataHelper.prediction_example()
        assert isinstance(prediction, dspy.Prediction)
        assert prediction.database_schema == "test database schema"
        assert prediction.documented_schema == "test documented schema"

    def test_trainset(self):
        with pytest.raises(NotImplementedError):
            GenerationDataHelper.trainset({})

    def test_trainset_from_dataset(self):
        dataset = Dataset.from_dict(
            {
                "schema_str": ["example database schema"],
            }
        )
        trainset = GenerationDataHelper.trainset(dataset)
        assert isinstance(trainset, list)
        assert len(trainset) == 1
        assert isinstance(trainset[0], dspy.Example)
        assert trainset[0].database_schema == "example database schema"
        assert trainset[0].documented_schema == "example database schema"
