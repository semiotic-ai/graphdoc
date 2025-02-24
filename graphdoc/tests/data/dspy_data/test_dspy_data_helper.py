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

            test_inpu: str = dspy.InputField()
            test_output: str = dspy.InputField()

        cot = dspy.ChainOfThought(TestSignature)
        pred = dspy.Predict(TestSignature)

        cot_signature = DspyDataHelper.prompt_signature(cot)
        pred_signature = DspyDataHelper.prompt_signature(pred)

        log.info(f"cot_signature (type: {type(cot_signature)}): {cot_signature}")
        log.info(f"pred_signature (type: {type(pred_signature)}): {pred_signature}")

        assert isinstance(cot_signature, (dspy.Signature, dspy.SignatureMeta))
        assert isinstance(pred_signature, (dspy.Signature, dspy.SignatureMeta))
