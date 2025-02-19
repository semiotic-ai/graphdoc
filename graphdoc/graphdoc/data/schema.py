# This file is used for defining our schema data class.

# system packages
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union, Type

# internal packages

# external packages
from graphql import Node

# logging
log = logging.getLogger(__name__)


class SchemaCategory(str, Enum):
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
    """Maps SchemaCategory to SchemaRating."""

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
    FULL_SCHEMA = "full schema"
    TABLE_SCHEMA = "table schema"
    ENUM_SCHEMA = "enum schema"

    @classmethod
    def from_str(cls, value: str) -> Optional["SchemaType"]:
        try:
            return cls(value)
        except ValueError:
            return None


@dataclass
class SchemaObject:
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
            except ValueError as e:
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
            except ValueError as e:
                raise ValueError(
                    f"Invalid rating. Must be one of: {[e.value for e in rating_enum]}"
                )

        schema_type = None
        if data.get("schema_type"):
            try:
                schema_type = type_enum(data["schema_type"])
            except ValueError as e:
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
