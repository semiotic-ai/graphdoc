# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import Parser

# external packages
from graphql import (
    DocumentNode,
    ObjectTypeDefinitionNode,
    EnumTypeDefinitionNode,
    EnumValueDefinitionNode,
)
import pytest

# logging
log = logging.getLogger(__name__)

# global variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class TestParser:

    def test__check_node_type(self, par: Parser):

        # DEFAULT_NODE_TYPES = {
        #     DocumentNode: "full schema",
        #     ObjectTypeDefinitionNode: "table schema",
        #     EnumTypeDefinitionNode: "enum schema",
        #     EnumValueDefinitionNode: "enum value",
        # }

        ALTERNATIVE_NODE_TYPES = {
            DocumentNode: "test full schema",
            ObjectTypeDefinitionNode: "test table schema",
            EnumTypeDefinitionNode: "test enum schema",
            EnumValueDefinitionNode: "test enum value",
        }
        document_node = DocumentNode()
        object_type_definition_node = ObjectTypeDefinitionNode()
        enum_type_definition_node = EnumTypeDefinitionNode()
        enum_value_definition_node = EnumValueDefinitionNode()

        # test the default mapping
        assert par._check_node_type(document_node) == "full schema"
        assert par._check_node_type(object_type_definition_node) == "table schema"
        assert par._check_node_type(enum_type_definition_node) == "enum schema"
        assert par._check_node_type(enum_value_definition_node) == "enum value"

        # check that we can pass in a custom mapping
        assert (
            par._check_node_type(document_node, ALTERNATIVE_NODE_TYPES)
            == "test full schema"
        )
        assert (
            par._check_node_type(object_type_definition_node, ALTERNATIVE_NODE_TYPES)
            == "test table schema"
        )
        assert (
            par._check_node_type(enum_type_definition_node, ALTERNATIVE_NODE_TYPES)
            == "test enum schema"
        )
        assert (
            par._check_node_type(enum_value_definition_node, ALTERNATIVE_NODE_TYPES)
            == "test enum value"
        )

    def test_parse_schema_from_file(self, par: Parser):
        schema_file = (
            Path(BASE_DIR)
            / "tests"
            / "assets"
            / "schemas"
            / "opensea_original_schema.graphql"
        )
        schema = par.parse_schema_from_file(schema_file)
        assert isinstance(schema, DocumentNode)

    # def test_update_node_descriptions(self, par: Parser):
    #     pass

    # def test_count_description_pattern_matching(self, par: Parser):
    #     pass

    # def test_fill_empty_descriptions(self, par: Parser):
    #     pass

    # def test_schema_equality_check(self, par: Parser):
    #     pass

    # def test_parse_objects_from_full_schema_object(self, par: Parser):
    #     pass
