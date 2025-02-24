# system packages
import logging

# internal packages
from graphdoc import DspyDataHelper

# external packages
import dspy

# logging
log = logging.getLogger(__name__)


class TestDspyDataHelper:
    def test_prompt_signature(self):
        class TestSignature(dspy.Signature):
            """Test signature."""

            test_input: str = dspy.InputField()
            test_output: str = dspy.InputField()

        cot = dspy.ChainOfThought(TestSignature)
        pred = dspy.Predict(TestSignature)

        cot_signature = DspyDataHelper.prompt_signature(cot)
        pred_signature = DspyDataHelper.prompt_signature(pred)

        log.info(f"cot_signature (type: {type(cot_signature)}): {cot_signature}")
        log.info(f"pred_signature (type: {type(pred_signature)}): {pred_signature}")

        assert isinstance(cot_signature, (dspy.Signature, dspy.SignatureMeta))
        assert isinstance(pred_signature, (dspy.Signature, dspy.SignatureMeta))

    def test_formatted_signature(self):
        class TestSignature(dspy.Signature):
            """Test signature."""

            test_input: str = dspy.InputField()
            test_output: str = dspy.InputField()

        example = dspy.Example(
            test_input="test input",
            test_output="test output",
        )

        formatted_signature = DspyDataHelper.formatted_signature(TestSignature, example)

        assert formatted_signature is not None
        assert isinstance(formatted_signature, str)
        assert "------\nSystem\n------" in formatted_signature
        assert "------\nUser\n------" in formatted_signature
        assert (
            formatted_signature
            == "------\nSystem\n------\n Your input fields are:\n1. `test_input` (str)\n2. `test_output` (str)\n\nYour output fields are:\n\n\nAll interactions will be structured in the following way, with the appropriate values filled in.\n\n[[ ## test_input ## ]]\n{test_input}\n\n[[ ## test_output ## ]]\n{test_output}\n\n\n\n[[ ## completed ## ]]\n\nIn adhering to this structure, your objective is: \n        Test signature. \n------\nUser\n------\n [[ ## test_input ## ]]\ntest input\n\n[[ ## test_output ## ]]\ntest output"
        )
