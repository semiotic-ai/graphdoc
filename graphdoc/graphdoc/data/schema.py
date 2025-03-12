# Copyright 2025-, Semiotic AI, Inc.
# SPDX-License-Identifier: Apache-2.0

# system packages
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Type, Union

from datasets import Dataset, Features, Value, concatenate_datasets

# external packages
from graphql import Node

# internal packages


# logging
log = logging.getLogger(__name__)


class SchemaCategory(str, Enum):
    """Schema quality categories enumeration.

    :no-index:

    """

    PERFECT = "perfect"
    ALMOST_PERFECT = "almost perfect"
    POOR_BUT_CORRECT = "poor but correct"
    INCORRECT = "incorrect"
    BLANK = "blank"

    @classmethod
    def from_str(cls, value: str) -> Optional["SchemaCategory"]:
        try:
            return cls(value)
        except ValueError:
            return None


class SchemaRating(str, Enum):
    """Schema quality ratings enumeration.

    :no-index:

    """

    FOUR = "4"
    THREE = "3"
    TWO = "2"
    ONE = "1"
    ZERO = "0"

    @classmethod
    def from_value(cls, value: Union[str, int]) -> Optional["SchemaRating"]:
        if isinstance(value, int):
            value = str(value)
        try:
            return cls(value)
        except ValueError:
            return None


class SchemaCategoryRatingMapping:
    """Mapping between schema categories and ratings.

    :no-index:

    """

    @staticmethod
    def get_rating(category: SchemaCategory) -> SchemaRating:
        """Get the corresponding rating for a given schema category.

        :param category: The schema category
        :return: The corresponding rating

        """
        mapping = {
            SchemaCategory.PERFECT: SchemaRating.FOUR,
            SchemaCategory.ALMOST_PERFECT: SchemaRating.THREE,
            SchemaCategory.POOR_BUT_CORRECT: SchemaRating.TWO,
            SchemaCategory.INCORRECT: SchemaRating.ONE,
            SchemaCategory.BLANK: SchemaRating.ZERO,
        }
        return mapping.get(category, SchemaRating.ZERO)

    @staticmethod
    def get_category(rating: SchemaRating) -> SchemaCategory:
        """Get the corresponding category for a given schema rating.

        :param rating: The schema rating
        :return: The corresponding category

        """
        mapping = {
            SchemaRating.FOUR: SchemaCategory.PERFECT,
            SchemaRating.THREE: SchemaCategory.ALMOST_PERFECT,
            SchemaRating.TWO: SchemaCategory.POOR_BUT_CORRECT,
            SchemaRating.ONE: SchemaCategory.INCORRECT,
            SchemaRating.ZERO: SchemaCategory.BLANK,
        }
        return mapping.get(rating, SchemaCategory.BLANK)


class SchemaType(str, Enum):
    """Schema type enumeration.

    :no-index:

    """

    FULL_SCHEMA = "full schema"
    TABLE_SCHEMA = "table schema"
    ENUM_SCHEMA = "enum schema"

    @classmethod
    def from_str(cls, value: str) -> Optional["SchemaType"]:
        try:
            return cls(value)
        except ValueError:
            return None


class SchemaCategoryPath(str, Enum):
    """Maps schema categories to their folder names.

    :no-index:

    """

    PERFECT = "perfect"
    ALMOST_PERFECT = "almost_perfect"
    POOR_BUT_CORRECT = "poor_but_correct"
    INCORRECT = "incorrect"
    BLANK = "blank"

    @classmethod
    def get_path(
        cls, category: SchemaCategory, folder_path: Union[str, Path]
    ) -> Optional[Path]:
        """Get the folder path for a given schema category and folder path.

        :param category: The schema category
        :return: The corresponding folder path

        """
        mapping = {
            SchemaCategory.PERFECT: Path(folder_path) / cls.PERFECT,
            SchemaCategory.ALMOST_PERFECT: Path(folder_path) / cls.ALMOST_PERFECT,
            SchemaCategory.POOR_BUT_CORRECT: Path(folder_path) / cls.POOR_BUT_CORRECT,
            SchemaCategory.INCORRECT: Path(folder_path) / cls.INCORRECT,
            SchemaCategory.BLANK: Path(folder_path) / cls.BLANK,
        }
        return mapping.get(category)


@dataclass
class SchemaObject:
    """Schema object containing schema data and metadata.

    :no-index:

    """

    key: str
    category: Optional[Enum] = None
    rating: Optional[Enum] = None
    schema_name: Optional[str] = None
    schema_type: Optional[Enum] = None
    schema_str: Optional[str] = None
    schema_ast: Optional[Node] = None

    @classmethod
    def from_dict(
        cls,
        data: dict,
        category_enum: Type[Enum] = SchemaCategory,
        rating_enum: Type[Enum] = SchemaRating,
        type_enum: Type[Enum] = SchemaType,
    ) -> "SchemaObject":
        """Create SchemaObject from dictionary with validation.

        :param data: The data dictionary
        :param category_enum: Custom Enum class for categories
        :param rating_enum: Custom Enum class for ratings
        :param type_enum: Custom Enum class for schema types

        """
        if "key" not in data:
            raise ValueError("Missing required field: key")

        category = None
        if data.get("category"):
            try:
                category = category_enum(data["category"])
            except ValueError:
                raise ValueError(
                    f"Invalid category. Must be one of: {[e.value for e in category_enum]}"
                )

        rating = None
        if data.get("rating"):
            try:
                if hasattr(rating_enum, "from_value"):
                    # we ignore the type because we know that the from_value method exists
                    rating = rating_enum.from_value(data["rating"])  # type: ignore
                else:
                    rating = rating_enum(data["rating"])
            except ValueError:
                raise ValueError(
                    f"Invalid rating. Must be one of: {[e.value for e in rating_enum]}"
                )

        schema_type = None
        if data.get("schema_type"):
            try:
                schema_type = type_enum(data["schema_type"])
            except ValueError:
                raise ValueError(
                    f"Invalid schema type. Must be one of: {[e.value for e in type_enum]}"
                )

        return cls(
            key=data["key"],
            category=category,
            rating=rating,
            schema_name=data.get("schema_name"),
            schema_type=schema_type,
            schema_str=data.get("schema_str"),
            schema_ast=data.get("schema_ast"),
        )

    def to_dict(self) -> dict:
        """Convert the SchemaObject to a dictionary, excluding the key field.

        :return: Dictionary representation of the SchemaObject without the key
        :rtype: dict

        """
        return {
            "category": self.category.value if self.category else None,
            "rating": self.rating.value if self.rating else None,
            "schema_name": self.schema_name,
            "schema_type": self.schema_type.value if self.schema_type else None,
            "schema_str": self.schema_str,
            "schema_ast": self.schema_ast,
        }

    @staticmethod
    def _hf_schema_object_columns() -> Features:
        """Return the columns for the graph_doc dataset, based on the SchemaObject
        fields.

        :return: The columns for the graph_doc dataset
        :rtype: Features

        """
        return Features(
            {
                "category": Value("string"),
                "rating": Value("string"),
                "schema_name": Value("string"),
                "schema_type": Value("string"),
                "schema_str": Value("string"),
            }
        )

    def to_dataset(self) -> Dataset:
        """Convert the SchemaObject to a Hugging Face Dataset.

        :return: The Hugging Face Dataset
        :rtype: Dataset

        """
        dictionary = {
            "category": [self.category.value if self.category else None],
            "rating": [self.rating.value if self.rating else None],
            "schema_name": [self.schema_name],
            "schema_type": [self.schema_type.value if self.schema_type else None],
            "schema_str": [self.schema_str],
        }
        return Dataset.from_dict(
            dictionary, features=SchemaObject._hf_schema_object_columns()
        )


# TODO: we may end up wanting to both abstract and/or move this elsewhere
def schema_objects_to_dataset(schema_objects: List[SchemaObject]) -> Dataset:
    """Convert a list of SchemaObjects to a Hugging Face Dataset.

    :param schema_objects: The list of SchemaObjects
    :return: The Hugging Face Dataset

    """
    return concatenate_datasets(
        [schema_object.to_dataset() for schema_object in schema_objects]
    )
