# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging

# external packages
from dotenv import load_dotenv

# internal packages
from graphdoc import (
    DocGeneratorPrompt,
    DocQualityPrompt,
    LocalDataHelper,
    Parser,
)

from .conftest import (
    OverwriteSchemaCategory,
    OverwriteSchemaCategoryRatingMapping,
    OverwriteSchemaRating,
)

# logging
log = logging.getLogger(__name__)

# load the environment variables
load_dotenv("../.env")


class TestFixtures:
    def test_parser(self, par: Parser):
        assert par is not None
        assert isinstance(par, Parser)

    def test_default_ldh(self, default_ldh: LocalDataHelper):
        assert default_ldh is not None
        assert isinstance(default_ldh, LocalDataHelper)

    def test_overwrite_ldh(self, overwrite_ldh: LocalDataHelper):
        assert overwrite_ldh is not None
        assert isinstance(overwrite_ldh, LocalDataHelper)
        assert overwrite_ldh.categories == OverwriteSchemaCategory
        assert overwrite_ldh.ratings == OverwriteSchemaRating
        assert (
            overwrite_ldh.categories_ratings
            == OverwriteSchemaCategoryRatingMapping.get_rating
        )

    def test_dqp(self, dqp):
        assert isinstance(dqp, DocQualityPrompt)
        assert dqp.prompt_type == "predict"
        assert dqp.prompt_metric == "rating"

    def test_dgp(self, dgp):
        assert isinstance(dgp, DocGeneratorPrompt)
        assert dgp.prompt_type == "chain_of_thought"
        assert isinstance(dgp.prompt_metric, DocQualityPrompt)
