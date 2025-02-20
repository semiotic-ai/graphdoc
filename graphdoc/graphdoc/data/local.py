# this file is for loading data from a local directory

# system packages
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Type, Callable

# internal packages
from .helper import check_directory_path, check_file_path
from .schema import (
    SchemaCategoryRatingMapping,
    SchemaObject,
    SchemaCategory,
    SchemaRating,
)
from .parser import Parser

# external packages

# logging
log = logging.getLogger(__name__)


class LocalDataHelper:
    """
    A helper class for loading data from a local directory.
    """

    def __init__(
        self,
        schema_directory_path: Optional[Union[str, Path]] = None,
        categories: Optional[Type[Enum]] = SchemaCategory,
        ratings: Optional[Type[Enum]] = SchemaRating,
        categories_ratings: Optional[Callable] = SchemaCategoryRatingMapping.get_rating,
    ):
        if schema_directory_path:
            check_directory_path(schema_directory_path)
            self.schema_directory_path = schema_directory_path
            self.package_directory_path = False
        else:
            self.schema_directory_path = Path(__file__).parent / "assets" / "schemas"
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

    # def _load_folder_schemas

    # def _load_folder_of_folders

    # def _schema_objects_to_dict # we should move this to schema.py

    # _schema_objects_to_dataset # we should move this to schema.py

    # def _folder_to_dataset

    # def _folder_of_folders_to_dataset

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
