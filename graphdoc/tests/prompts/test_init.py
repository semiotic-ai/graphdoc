# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging

# internal packages
from graphdoc import DocGeneratorPrompt, DocQualityPrompt, PromptFactory

# external packages

# logging
log = logging.getLogger(__name__)


class TestPromptInit:
    def test_single_prompt(self):
        """
        Test the single_prompt function.
        """
        dqp = PromptFactory.single_prompt(
            prompt="doc_quality",
            prompt_class="DocQualityPrompt",
            prompt_type="predict",
            prompt_metric="rating",
        )
        dgp = PromptFactory.single_prompt(
            prompt="base_doc_gen",
            prompt_class="DocGeneratorPrompt",
            prompt_type="chain_of_thought",
            prompt_metric=dqp,
        )
        assert isinstance(dqp, DocQualityPrompt)
        assert isinstance(dgp, DocGeneratorPrompt)
