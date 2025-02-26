# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import Parser
from graphdoc import SchemaObject

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
SCHEMA_DIR = BASE_DIR / "tests" / "assets" / "schemas"


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
        schema_file = SCHEMA_DIR / "opensea_original_schema.graphql"
        schema = par.parse_schema_from_file(
            schema_file, schema_directory_path=SCHEMA_DIR
        )
        assert isinstance(schema, DocumentNode)

    def test_update_node_descriptions(self, par: Parser):
        schema_file = "opensea_original_schema.graphql"
        schema = par.parse_schema_from_file(
            schema_file, schema_directory_path=SCHEMA_DIR
        )
        updated_schema = par.update_node_descriptions(
            node=schema, new_value="This is a test description"
        )
        for i in range(3, 6):
            for x in range(3):
                definitions = getattr(updated_schema, "definitions", None)
                if definitions:
                    test_node_definition = definitions[i].fields[x].description.value
                    assert test_node_definition == "This is a test description"

    def test_count_description_pattern_matching(self, par: Parser):
        gold_schema_file = SCHEMA_DIR / "opensea_original_schema_pattern.graphql"
        gold_schema = par.parse_schema_from_file(gold_schema_file)
        counts = par.count_description_pattern_matching(gold_schema, "test")
        assert counts["total"] == 12
        assert counts["pattern"] == 3
        assert counts["empty"] == 4

    def test_fill_empty_descriptions(self, par: Parser):
        schema_file = SCHEMA_DIR / "opensea_original_schema_sparse.graphql"
        schema = par.parse_schema_from_file(schema_file)
        updated_schema = par.fill_empty_descriptions(schema)
        definitions = getattr(updated_schema, "definitions", None)

        if definitions:
            test_entity_definition_updated = definitions[3].fields[0].description.value
            assert test_entity_definition_updated == "Description for column: id"

            test_enum_definition_content = definitions[2].values[0].description.value
            assert (
                test_enum_definition_content
                == " Strategy that executes an order at a fixed price that can be taken either by a bid or an ask. "
            )

            test_entity_description = definitions[5].description.value
            assert (
                test_entity_description
                == " Trades exist such as a combination of taker/order and bid/ask. "
            )

            test_entity_description_updated = definitions[3].description.value
            assert (
                test_entity_description_updated == "Description for table: Marketplace"
            )

    def test_schema_equality_check(self, par: Parser):
        gold_schema_file = SCHEMA_DIR / "opensea_original_schema.graphql"
        silver_schema_file = (
            SCHEMA_DIR
            / "opensea_original_schema_sparse.graphql"  # only the comments are different
        )
        check_schema_file = SCHEMA_DIR / "opensea_original_schema_modified.graphql"
        gold_schema = par.parse_schema_from_file(gold_schema_file)
        silver_schema = par.parse_schema_from_file(silver_schema_file)
        check_schema = par.parse_schema_from_file(check_schema_file)

        assert par.schema_equality_check(gold_schema, gold_schema)
        assert par.schema_equality_check(gold_schema, silver_schema)
        assert par.schema_equality_check(gold_schema, check_schema) is False

    def test_schema_object_from_file(self, par: Parser):
        schema_file = SCHEMA_DIR / "opensea_original_schema_sparse.graphql"
        schema = par.schema_object_from_file(schema_file, rating=3)
        assert isinstance(schema, SchemaObject)
        assert schema.schema_type == "full schema"
        assert schema.schema_name == "opensea_original_schema_sparse"
        assert schema.rating == "3"

    def test_parse_objects_from_full_schema_object(self, par: Parser):
        schema_file = SCHEMA_DIR / "opensea_original_schema_sparse.graphql"
        schema_object = par.schema_object_from_file(schema_file, rating=3)
        objects = par.parse_objects_from_full_schema_object(schema_object)
        assert isinstance(objects, dict)
        counter = 0
        for obj in objects.values():
            log.debug(f"obj type ({type(obj)}): {obj.schema_type}")
            assert isinstance(obj, SchemaObject)
            counter += 1
        assert counter == 9
