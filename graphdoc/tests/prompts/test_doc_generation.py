# system packages
import logging

# internal packages
from graphdoc import GraphDoc
from graphdoc import DocGeneratorPrompt, DocQualityPrompt
from graphdoc import DocGeneratorSignature
from graphdoc import GenerationDataHelper

# external packages
import dspy
from typing import Callable

# logging
log = logging.getLogger(__name__)


class TestDocGeneratorPrompt:
    def test_doc_generator_prompt(self, gd: GraphDoc):
        dgp = DocGeneratorPrompt(
            prompt="base_doc_gen",
            prompt_type="chain_of_thought",
            prompt_metric=DocQualityPrompt(
                prompt="doc_quality",
                prompt_type="predict",
                prompt_metric="rating",
            ),
        )
        assert isinstance(dgp, DocGeneratorPrompt)
        # assert isinstance(dgp.prompt, DocGeneratorSignature)
        assert dgp.prompt_type == "chain_of_thought"
        assert isinstance(dgp.prompt_metric, DocQualityPrompt)

    def test_evaluate_documentation_quality(self, gd: GraphDoc):
        example = GenerationDataHelper.example_example()
        prediction = GenerationDataHelper.prediction_example()
        dgp = DocGeneratorPrompt(
            prompt="base_doc_gen",
            prompt_type="chain_of_thought",
            prompt_metric=DocQualityPrompt(
                prompt="doc_quality", prompt_type="predict", prompt_metric="rating"
            ),
        )
        score = dgp.evaluate_documentation_quality(example, prediction)
        assert isinstance(score, int)
        assert score >= 0
        assert score <= 4
