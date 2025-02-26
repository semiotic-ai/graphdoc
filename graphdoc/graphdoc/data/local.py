# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Type, Callable, Dict

# internal packages
from .helper import check_directory_path, check_file_path
from .schema import (
    SchemaCategoryRatingMapping,
    SchemaObject,
    SchemaCategory,
    SchemaRating,
    SchemaCategoryPath,
)
from .parser import Parser

# external packages
from datasets import Dataset, concatenate_datasets

# logging
log = logging.getLogger(__name__)


# TODO: we can make this a base class to enable better separation of our enum values and set up a factory pattern so that everything can be defined at the config level
# check out how pytorch etc. handles loading in something like imagenet
class LocalDataHelper:
    """
    A helper class for loading data from a directory.

    :param schema_directory_path: The path to the directory containing the schemas
    :type schema_directory_path: Union[str, Path]. Defaults to the path to the schemas in the graphdoc package.
    :param categories: The categories of the schemas. Defaults to SchemaCategory.
    :type categories: Type[Enum]
    :param ratings: The ratings of the schemas. Defaults to SchemaRating.
    :type ratings: Type[Enum]
    :param categories_ratings: A callable that maps categories to ratings. Defaults to SchemaCategoryRatingMapping.get_rating.
    """

    def __init__(
        self,
        schema_directory_path: Union[str, Path] = Path(__file__).parent
        / "assets"
        / "schemas",
        categories: Type[Enum] = SchemaCategory,
        ratings: Type[Enum] = SchemaRating,
        categories_ratings: Callable = SchemaCategoryRatingMapping.get_rating,
        # TODO: potentially add a category_path object here (defaulting to SchemaCategoryPath)
    ):
        if schema_directory_path:
            check_directory_path(schema_directory_path)
            self.schema_directory_path = schema_directory_path
            self.package_directory_path = False
        else:
            check_directory_path(self.schema_directory_path)
            self.package_directory_path = True

        self.categories = categories
        self.ratings = ratings
        self.categories_ratings = categories_ratings

    # def _categories # we will get rid of this

    # def _blank_schema_folder(self) -> Path: # we will get rid of this

    # def _folder_paths(self) -> dict: # we will get rid of this

    # def _category_ratings(self) -> dict: # we will get rid of this

    # def _check_category_validity(self, category: str) -> bool: # we will get rid of this

    # def _check_node_type(self, node: Node) -> str: # this is now in the parser.py file

    # def _parse_objects_from_full_schema_object(self, schema: SchemaObject) -> dict[str, SchemaObject]: # this is now in the parser.py file

    # def schemas_folder
    def schema_objects_from_folder(
        self, category: str, rating: int, folder_path: Union[str, Path]
    ) -> dict[str, SchemaObject]:
        """
        Load schemas from a folder, keeping the difficulty tag.

        :param folder_path: The path to the folder containing the schemas
        :type folder_path: Union[str, Path]
        :param category: The category of the schemas
        :type category: str
        :param rating: The rating of the schemas
        :type rating: int
        :return: A dictionary of schemas
        :rtype: dict[str, SchemaObject]
        """
        check_directory_path(folder_path)
        schemas = {}
        for schema_file in Path(folder_path).iterdir():
            check_file_path(schema_file)
            try:
                schema_object = Parser.schema_object_from_file(
                    schema_file, category=category, rating=rating
                )
                schemas[schema_file] = schema_object
            except Exception as e:
                log.warning(f"Error parsing schema file {schema_file}: {e}")
                continue
        return schemas

    def schema_objects_from_folder_of_folders(
        self,
        folder_paths: Optional[Type[Enum]] = SchemaCategoryPath,
    ) -> Union[Dict[str, SchemaObject], None]:
        """
        Load a folder of folders containing schemas, keeping the difficulty tag.

        :param folder_paths: Enum class defining folder paths, defaults to SchemaCategoryPath. Must have a get_path method.
        :type folder_paths: Optional[Type[Enum]]
        :return: Dictionary of loaded schemas
        :rtype: Union[Dict[str, SchemaObject], None]
        """
        schemas = {}

        # iterate through categories defined in self.categories
        for category in self.categories:
            try:
                category_enum = self.categories(category)
                rating = self.categories_ratings(category_enum)

                # get path using provided folder_paths enum
                if not hasattr(folder_paths, "get_path"):
                    raise AttributeError(
                        f"folder_paths enum must have a get_path method. Received: {folder_paths}"
                    )
                path = folder_paths.get_path(category_enum, self.schema_directory_path)  # type: ignore # since we know that the enum has a get_path method
                if not path:
                    log.warning(f"No path found for category: {category}")
                    continue

                try:
                    folder_schemas = self.schema_objects_from_folder(
                        category=category.value, rating=rating.value, folder_path=path
                    )
                    schemas.update(folder_schemas)
                except Exception as e:
                    log.warning(f"Error loading schemas from {path}: {e}")
                    continue

            except ValueError as e:
                log.warning(f"Invalid category {category}: {e}")
                continue

        return schemas if schemas else None

    def folder_to_dataset(
        self,
        category: str,
        folder_path: Union[str, Path],
        parse_objects: bool = True,
        type_mapping: Optional[dict[type, str]] = None,
    ) -> Dataset:
        """
        Load a folder of schemas, keeping the difficulty tag.

        :param category: The category of the schemas
        :type category: str
        :param folder_path: The path to the folder containing the schemas
        :type folder_path: Union[str, Path]
        :param parse_objects: Whether to parse the objects from the schemas
        :type parse_objects: bool
        :param type_mapping: A dictionary mapping types to strings
        :type type_mapping: Optional[dict[type, str]]
        """
        objects = []
        rating = self.categories_ratings(self.categories(category))
        schema_objects = self.schema_objects_from_folder(
            category=category, rating=rating, folder_path=folder_path
        )

        for schema_object in schema_objects.values():
            if parse_objects:
                parsed_objects = Parser.parse_objects_from_full_schema_object(
                    schema=schema_object, type_mapping=type_mapping
                )
                if parsed_objects:
                    for parsed_object in parsed_objects.values():
                        objects.append(parsed_object)
            objects.append(schema_object)

        return concatenate_datasets(
            [schema_object.to_dataset() for schema_object in objects]
        )

    def folder_of_folders_to_dataset(
        self,
        folder_paths: Type[Enum] = SchemaCategoryPath,
        parse_objects: bool = True,
        type_mapping: Optional[dict[type, str]] = None,
    ) -> Dataset:
        """
        Load a folder of folders containing schemas, keeping the difficulty tag.

        :param folder_paths: Enum class defining folder paths, defaults to SchemaCategoryPath. Must have a get_path method.
        :type folder_paths: Type[Enum]
        :param parse_objects: Whether to parse the objects from the schemas
        :type parse_objects: bool
        :param type_mapping: A dictionary mapping graphql-ast node values to strings
        """
        schema_objects = self.schema_objects_from_folder_of_folders(
            folder_paths=folder_paths
        )
        if schema_objects is None:
            raise ValueError("No schema objects found")
        objects = []
        for schema_object in schema_objects.values():
            if parse_objects:
                parsed_objects = Parser.parse_objects_from_full_schema_object(
                    schema=schema_object, type_mapping=type_mapping
                )
                if parsed_objects:
                    for parsed_object in parsed_objects.values():
                        objects.append(parsed_object)
            objects.append(schema_object)
        return concatenate_datasets(
            [schema_object.to_dataset() for schema_object in objects]
        )

    # def _get_graph_doc_columns # we should move this to a huggingface file

    # def _get_empty_graphdoc_data # we should move this to a huggingface file

    # def _check_graph_doc_data_dict # we should move this to a huggingface file

    # def _create_graph_doc_dataset # we should move this to a huggingface file

    # def _load_from_hf # we should move this to a huggingface file

    # def _check_graph_doc_dataset_format # we should move this to a huggingface file

    # def _add_to_graph_doc_dataset # we should move this to a huggingface file

    # def _drop_dataset_duplicates # we should move this to a huggingface file

    # def _upload_to_hf # we should move this to a huggingface file

    # def _upload_repo_card_to_hf # we should move this to a huggingface file

    # def _create_repo_card # we should move this to a huggingface file

    # def _create_and_upload_repo_card # we should move this to a huggingface file

    # def _create_graph_doc_example_trainset # we should move this to a dspy file

    # def _create_doc_generator_example_trainset # we should move this to a dspy file

    # def create_graph_doc_example_trainset # we should move this to a dspy file

    # def blank_trainset # we should move this to a dspy file
