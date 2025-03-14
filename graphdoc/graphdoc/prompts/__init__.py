# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
from typing import Union

# external packages
import dspy

from graphdoc.prompts.schema_doc_generation import (
    BadDocGeneratorSignature,
    DocGeneratorHelperSignature,
    DocGeneratorPrompt,
    DocGeneratorSignature,
)
from graphdoc.prompts.schema_doc_quality import (
    DocQualityDemonstrationSignature,
    DocQualityPrompt,
    DocQualitySignature,
)

# internal packages
from graphdoc.prompts.single_prompt import SinglePrompt

# logging
log = logging.getLogger(__name__)


class PromptFactory:
    @staticmethod
    def single_prompt(
        prompt: Union[str, dspy.Signature, dspy.SignatureMeta],
        prompt_class: str,
        prompt_type: str,
        prompt_metric: Union[str, DocQualityPrompt, SinglePrompt],
    ) -> SinglePrompt:
        """Returns an instance of the specified prompt class. Allows for the user to
        pass in their own dspy signature.

        :param prompt: The prompt to use.
        :type prompt: Union[str, dspy.Signature]
        :param prompt_class: The class of the prompt to use.
        :type prompt_class: str
        :param prompt_type: The type of the prompt to use.
        :type prompt_type: str
        :param prompt_metric: The metric to use for the prompt.
        :type prompt_metric: Union[str, DocQualityPrompt, SinglePrompt]
        :return: An instance of the specified prompt class.
        :rtype: SinglePrompt
        """
        prompt_classes = {
            "DocQualityPrompt": DocQualityPrompt,
            "DocGeneratorPrompt": DocGeneratorPrompt,
        }
        if prompt_class not in prompt_classes:
            raise ValueError(f"Unknown prompt class: {prompt_class}")
        try:
            # TODO: we should be able to have better type checking here
            return prompt_classes[prompt_class](
                prompt=prompt, prompt_type=prompt_type, prompt_metric=prompt_metric
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize prompt class ({prompt_class}): {e}")


__all__ = [
    "BadDocGeneratorSignature",
    "DocGeneratorHelperSignature",
    "DocGeneratorPrompt",
    "DocGeneratorSignature",
    "DocQualityDemonstrationSignature",
    "DocQualityPrompt",
    "DocQualitySignature",
    "PromptFactory",
    "SinglePrompt",
]
