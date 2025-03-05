# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import copy
import logging
from pathlib import Path
from typing import Optional, Union

# external packages
from graphql import (
    DocumentNode,
    EnumTypeDefinitionNode,
    EnumValueDefinitionNode,
    FieldDefinitionNode,
    Node,
    ObjectTypeDefinitionNode,
    StringValueNode,
    parse,
    print_ast,
)

from .helper import check_directory_path, check_file_path

# internal packages
from .schema import SchemaObject

# logging
log = logging.getLogger(__name__)


class Parser:
    """A class for parsing and handling of GraphQL objects."""

    DEFAULT_NODE_TYPES = {
        DocumentNode: "full schema",
        ObjectTypeDefinitionNode: "table schema",
        EnumTypeDefinitionNode: "enum schema",
        EnumValueDefinitionNode: "enum value",
    }

    def __init__(self, type_mapping: Optional[dict[type, str]] = None) -> None:
        self.type_mapping = type_mapping or Parser.DEFAULT_NODE_TYPES

    @staticmethod
    def _check_node_type(
        node: Node, type_mapping: Optional[dict[type, str]] = None
    ) -> str:
        """Check the type of a schema node.

        :param node: The schema node to check :type node: Node :param type_mapping:
        Custom mapping of node types to strings. Defaults to DEFAULT_NODE_TYPES :type
        type_mapping: Optional[dict[type, str]] :return: The type of the schema node
        :rtype: str

        """
        # use provided mapping or fall back to defaults
        mapping = type_mapping or Parser.DEFAULT_NODE_TYPES
        return mapping.get(type(node), "unknown schema")

    @staticmethod
    def parse_schema_from_file(
        schema_file: Union[str, Path],
        schema_directory_path: Optional[Union[str, Path]] = None,
    ) -> DocumentNode:
        """Parse a schema from a file.

        :param schema_file: The name of the schema file :type schema_
        file:
        str
        :param schema_directory_path: A path to a directory containing schemas
        :type schema_directory_path: str
        :return: The parsed schema
        :rtype: DocumentNode
        :raises Exception: If the schema cannot be parsed

        """
        if schema_directory_path:
            check_directory_path(schema_directory_path)
            schema_path = Path(schema_directory_path) / schema_file
        else:
            check_file_path(schema_file)
            schema_path = Path(schema_file)

        try:
            schema = schema_path.read_text()
            return parse(schema)
        except Exception as e:
            log.error(f"Error parsing schema from file: {e}")
            raise e

    @staticmethod
    def update_node_descriptions(node: Node, new_value: Optional[str] = None) -> Node:
        """Given a GraphQL node, recursively traverse the node and its children,
        updating all descriptions with the new value. Can also be used to remove
        descriptions by passing None as the new value.

        :param node: The GraphQL node to update :type node: Node :param new_value: The
        new description value. If None, the description will be removed. :type
        new_value: Optional[str] :return: The updated node :rtype: Node

        """
        if hasattr(node, "description"):
            description = getattr(node, "description", None)
            if isinstance(description, StringValueNode):
                if new_value:
                    description.value = new_value
                else:
                    setattr(node, "description", None)

        for attr in dir(node):
            if attr.startswith("__") or attr == "description":
                continue
            child = getattr(node, attr, None)
            if isinstance(child, (list, tuple)):
                for item in child:
                    if isinstance(item, Node):
                        Parser.update_node_descriptions(item, new_value)
            elif isinstance(child, Node):
                Parser.update_node_descriptions(child, new_value)
        return node

    @staticmethod
    def count_description_pattern_matching(node: Node, pattern: str) -> dict[str, int]:
        """Counts the number of times a pattern matches a description in a node and its
        children.

        :param node: The GraphQL node to count the pattern matches in :type node: Node
        :param pattern: The pattern to count the matches of :type pattern: str :return:
        A dictionary containing the total number of descriptions, the number of
        descriptions that match the pattern, and the number of descriptions that are
        empty. { "total": int, "pattern": int, "empty": int } :rtype: dict[str, int]

        """
        counts = {
            "total": 0,
            "pattern": 0,
            "empty": 0,
        }

        def update_counts(node: Node, counts: dict):
            if hasattr(node, "description"):
                description = getattr(node, "description", None)
                counts["total"] += 1
                if description is None:
                    counts["empty"] += 1
                elif pattern in description.value:
                    counts["pattern"] += 1
            return counts

        def traverse(node: Node, counts: dict):
            counts = update_counts(node, counts)

            for attr in dir(node):
                if attr.startswith("__") or attr == "description":
                    continue
                child = getattr(node, attr, None)
                if isinstance(child, (list, tuple)):
                    for item in child:
                        if isinstance(item, Node):
                            traverse(item, counts)
                elif isinstance(child, Node):
                    traverse(child, counts)
            return counts

        counts = traverse(node, counts)
        return counts

    @staticmethod
    def fill_empty_descriptions(
        node: Node,
        new_column_value: str = "Description for column: {}",
        new_table_value: str = "Description for table: {}",
        use_value_name: bool = True,
        value_name: Optional[str] = None,
    ):
        """Recursively traverse the node and its children, filling in empty descriptions
        with the new column or table value. Do not update descriptions that already have
        a value. Default values are provided for the new column and table descriptions.

        :param node: The GraphQL node to update :type node: Node :param
        new_column_value: The new column description value :type new_column_value: str
        :param new_table_value: The new table description value :type new_table_value:
        str :param use_value_name: Whether to use the value name in the description
        :type use_value_name: bool :param value_name: The name of the value :type
        value_name: Optional[str] :return: The updated node :rtype: Node

        """
        if hasattr(node, "description"):  # and node.description == None:
            description = getattr(node, "description", None)
            if description == None:
                # if the node is a table, use the table value
                if isinstance(node, ObjectTypeDefinitionNode):
                    new_value = new_table_value
                elif isinstance(node, EnumTypeDefinitionNode):  # this is an enum type
                    new_value = f"Description for enum type: {value_name}"  # TODO: we should add this back to the fill_empty_descriptions parameter list
                # else the node is a column, use the column value
                else:
                    new_value = new_column_value
                # format with the value name if needed (table/column name)
                if use_value_name:
                    update_value = new_value.format(value_name)
                else:
                    update_value = new_value

                node.description = StringValueNode(value=update_value)

        for attr in dir(node):
            if attr.startswith("__") or attr == "description":
                continue
            child = getattr(node, attr, None)
            if isinstance(child, (list, tuple)):
                for item in child:
                    if isinstance(item, Node):
                        if (
                            isinstance(item, FieldDefinitionNode)
                            or isinstance(item, EnumValueDefinitionNode)
                            or isinstance(item, ObjectTypeDefinitionNode)
                            or isinstance(
                                item, EnumTypeDefinitionNode
                            )  # EnumTypeDefinitionNode: check
                        ):
                            if isinstance(child, ObjectTypeDefinitionNode):
                                log.debug(
                                    f"found an instance of a ObjectTypeDefinitionNode: {item.name.value}"
                                )
                            value_name = item.name.value
                        Parser.fill_empty_descriptions(
                            item,
                            new_column_value,
                            new_table_value,
                            use_value_name,
                            value_name,
                        )
            elif isinstance(child, Node):
                if (
                    isinstance(child, FieldDefinitionNode)
                    or isinstance(child, EnumValueDefinitionNode)
                    or isinstance(child, ObjectTypeDefinitionNode)
                    or isinstance(child, EnumTypeDefinitionNode)
                ):
                    if isinstance(child, ObjectTypeDefinitionNode):
                        log.debug(
                            f"found an instance of a ObjectTypeDefinitionNode: {child.name.value}"
                        )
                    value_name = child.name.value
                Parser.fill_empty_descriptions(
                    child,
                    new_column_value,
                    new_table_value,
                    use_value_name,
                    value_name,
                )
        return node

    @staticmethod
    def schema_equality_check(gold_node: Node, check_node: Node) -> bool:
        """A method to check if two schema nodes are equal. Only checks that the schemas
        structures are equal, not the descriptions.

        :param gold_node: The gold standard schema node :type gold_node: Node :param
        check_node: The schema node to check :type check_node: Node :return: Whether the
        schemas are equal :rtype: bool

        """
        gold_node_copy = copy.deepcopy(gold_node)
        check_node_copy = copy.deepcopy(check_node)
        gold_node = Parser.update_node_descriptions(gold_node_copy)
        check_node = Parser.update_node_descriptions(check_node_copy)

        if print_ast(gold_node) != print_ast(check_node):
            return False
        else:
            return True

    @staticmethod
    def schema_object_from_file(
        schema_file: Union[str, Path],
        category: Optional[str] = None,
        rating: Optional[int] = None,
    ) -> SchemaObject:
        """Parse a schema object from a file."""
        try:
            schema_ast = Parser.parse_schema_from_file(schema_file)
            schema_str = print_ast(schema_ast)
            schema_type = Parser._check_node_type(schema_ast)
            return SchemaObject.from_dict(
                {
                    "key": str(schema_file),
                    "category": category,
                    "rating": rating,
                    "schema_name": str(Path(schema_file).stem),
                    "schema_type": schema_type,
                    "schema_str": schema_str,
                    "schema_ast": schema_ast,
                }
            )
        except Exception as e:
            log.error(f"Error parsing schema file {schema_file}: {e}")
            raise ValueError(f"Failed to parse schema from file {schema_file}: {e}")

    @staticmethod
    def parse_objects_from_full_schema_object(
        schema: SchemaObject, type_mapping: Optional[dict[type, str]] = None
    ) -> Union[dict[str, SchemaObject], None]:
        """Parse out all available tables from a full schema object.

        :param schema: The full schema object to parse :type schema: SchemaObject :param
        type_mapping: Custom mapping of node types to strings. Defaults to
        DEFAULT_NODE_TYPES :type type_mapping: Optional[dict[type, str]] :return: The
        parsed objects (tables and enums) :rtype: Union[dict, None]

        """
        if schema.schema_ast is None:
            log.info(f"Schema object has no schema_ast: {schema.schema_name}")
            return None
        elif not isinstance(schema.schema_ast, DocumentNode):
            log.info(
                f"Schema object cannot be further decomposed: {schema.schema_name}"
            )
            return None

        tables = {}
        for definition in schema.schema_ast.definitions:
            if isinstance(definition, ObjectTypeDefinitionNode):
                log.debug(f"found table schema")
                key = f"{schema.key}_{definition.name.value}"
                schema_type = Parser._check_node_type(definition, type_mapping)
            elif isinstance(definition, EnumTypeDefinitionNode):
                log.debug(f"found enum schema")
                key = f"{schema.key}_{definition.name.value}"
                schema_type = Parser._check_node_type(definition, type_mapping)
            else:
                log.debug(f"skipping schema of type: {type(definition)}")
                continue
            object_schema = SchemaObject.from_dict(
                {
                    "key": key,
                    "category": schema.category,
                    "rating": schema.rating,
                    "schema_name": definition.name.value,
                    "schema_type": schema_type,
                    "schema_str": print_ast(definition),
                    "schema_ast": definition,
                }
            )
            tables[object_schema.key] = object_schema
        return tables
