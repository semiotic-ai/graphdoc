# system packages
import logging

# internal packages
from graphdoc.prompts import DocQualityPrompt

# external packages
import dspy

# logging
log = logging.getLogger(__name__)


class CustomPrompt(dspy.Signature):
    """
    This is a custom prompt.
    """

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
        assert dqp.evaluate_metric(example, pass_prediction) == True
        assert dqp.evaluate_metric(example, fail_prediction) == False
