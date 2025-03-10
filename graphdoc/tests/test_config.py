# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
from pathlib import Path

# external packages
import dspy

# internal packages
from graphdoc import (
    DocGeneratorEvaluator,
    DocGeneratorModule,
    DocGeneratorPrompt,
    DocGeneratorTrainer,
    DocQualityPrompt,
    DocQualityTrainer,
    SinglePromptTrainer,
    load_yaml_config,
)
from graphdoc.config import (
    doc_generator_eval_from_yaml,
    doc_generator_module_from_dict,
    doc_generator_module_from_yaml,
    single_prompt_from_dict,
    single_prompt_from_yaml,
    single_trainer_from_yaml,
    split_trainset,
    trainset_and_evalset_from_yaml,
    trainset_from_dict,
    trainset_from_yaml,
)

# logging
log = logging.getLogger(__name__)

# Define the base directory (project root)
BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_DIR = BASE_DIR / "tests" / "assets" / "schemas"
CONFIG_DIR = BASE_DIR / "tests" / "assets" / "configs"


class TestConfig:

    ############################################################
    # data tests                                               #
    ############################################################

    def test_trainset_from_dict(self):
        config_path = CONFIG_DIR / "single_prompt_doc_quality_trainer.yaml"
        config_dict = load_yaml_config(config_path)
        data_dict = config_dict["data"]

        trainset = trainset_from_dict(data_dict)
        assert isinstance(trainset, list)
        assert len(trainset) > 0
        assert isinstance(trainset[0], dspy.Example)

    def test_trainset_from_yaml(self):
        config_path = CONFIG_DIR / "single_prompt_doc_quality_trainer.yaml"
        trainset = trainset_from_yaml(config_path)
        assert isinstance(trainset, list)
        assert len(trainset) > 0
        assert isinstance(trainset[0], dspy.Example)

    def test_split_trainset(self):
        config_path = CONFIG_DIR / "single_prompt_doc_quality_trainer.yaml"
        config_dict = load_yaml_config(config_path)
        config_dict["data"]["trainset_size"] = 10
        config_dict["data"]["evalset_ratio"] = 0.2
        trainset = trainset_from_dict(config_dict["data"])
        trainset, evalset = split_trainset(
            trainset, config_dict["data"]["evalset_ratio"]
        )
        assert isinstance(trainset, list)
        assert len(trainset) == 8
        assert isinstance(trainset[0], dspy.Example)
        assert isinstance(evalset, list)
        assert len(evalset) == 2

    def test_trainset_and_evalset_from_yaml(self):
        config_path = CONFIG_DIR / "single_prompt_doc_quality_trainer.yaml"
        trainset, evalset = trainset_and_evalset_from_yaml(config_path)
        assert isinstance(trainset, list)
        assert len(trainset) == 900
        assert isinstance(trainset[0], dspy.Example)
        assert isinstance(evalset, list)
        assert len(evalset) == 100

    ############################################################
    # prompt tests                                             #
    ############################################################

    def test_single_prompt_from_dict(self):
        config_path = CONFIG_DIR / "single_prompt_doc_quality_trainer.yaml"
        prompt_dict = load_yaml_config(config_path)["prompt"]
        prompt_metric = prompt_dict["metric"]
        prompt = single_prompt_from_dict(prompt_dict, prompt_metric)
        assert isinstance(prompt, DocQualityPrompt)

        config_path = CONFIG_DIR / "single_prompt_doc_generator_trainer.yaml"
        prompt_dict = load_yaml_config(config_path)["prompt"]
        prompt_metric = prompt
        generator_prompt = single_prompt_from_dict(prompt_dict, prompt_metric)
        assert isinstance(generator_prompt, DocGeneratorPrompt)
        assert isinstance(generator_prompt.prompt_metric, DocQualityPrompt)

    def test_single_prompt_by_version_from_dict(self):
        config_path = CONFIG_DIR / "single_prompt_doc_quality_trainer.yaml"
        config_dict = load_yaml_config(config_path)
        prompt_dict = config_dict["prompt"]
        prompt_dict["load_from_mlflow"] = True
        prompt_dict["model_name"] = "doc_quality_model"
        prompt_dict["model_version"] = "1"
        prompt_dict["type"] = "predict"
        prompt_metric = prompt_dict["metric"]
        mlfow_dict = config_dict["mlflow"]
        prompt = single_prompt_from_dict(prompt_dict, prompt_metric, mlfow_dict)
        assert isinstance(prompt, DocQualityPrompt)

    def test_single_prompt_from_yaml(self):
        config_path = CONFIG_DIR / "single_prompt_doc_quality_trainer.yaml"
        prompt = single_prompt_from_yaml(config_path)
        assert isinstance(prompt, DocQualityPrompt)

        config_path = CONFIG_DIR / "single_prompt_doc_generator_trainer.yaml"
        prompt = single_prompt_from_yaml(config_path)
        assert isinstance(prompt, DocGeneratorPrompt)

    ############################################################
    # trainer tests                                            #
    ############################################################

    def test_single_trainer_from_yaml(self):
        config_path = CONFIG_DIR / "single_prompt_doc_quality_trainer.yaml"
        trainer = single_trainer_from_yaml(config_path)
        assert isinstance(trainer, SinglePromptTrainer)
        assert isinstance(trainer, DocQualityTrainer)
        assert isinstance(trainer.prompt, DocQualityPrompt)

        config_path = CONFIG_DIR / "single_prompt_doc_generator_trainer.yaml"
        trainer = single_trainer_from_yaml(config_path)
        assert isinstance(trainer, SinglePromptTrainer)
        assert isinstance(trainer, DocGeneratorTrainer)
        assert isinstance(trainer.prompt, DocGeneratorPrompt)

    ############################################################
    # module tests                                             #
    ############################################################

    def test_doc_generator_module_from_dict(self):
        config_path = CONFIG_DIR / "single_prompt_doc_generator_module.yaml"
        prompt = single_prompt_from_yaml(config_path)
        config_dict = load_yaml_config(config_path)
        module_dict = config_dict["module"]
        module = doc_generator_module_from_dict(module_dict, prompt)
        assert isinstance(module, DocGeneratorModule)

    def test_doc_generator_module_from_yaml(self):
        config_path = CONFIG_DIR / "single_prompt_doc_generator_module.yaml"
        module = doc_generator_module_from_yaml(config_path)
        assert isinstance(module, DocGeneratorModule)

    ############################################################
    # eval tests                                               #
    ############################################################

    def test_doc_generator_eval_from_yaml(self):
        config_path = CONFIG_DIR / "single_prompt_doc_generator_module_eval.yaml"
        evaluator = doc_generator_eval_from_yaml(config_path)
        assert isinstance(evaluator, DocGeneratorEvaluator)
