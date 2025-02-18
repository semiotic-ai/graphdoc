# This file is used for defining our schema data class.

# system packages
import logging
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Union

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
    category: Optional[SchemaCategory] = None
    rating: Optional[SchemaRating] = None
    schema_name: Optional[str] = None
    schema_type: Optional[SchemaType] = None
    schema_str: Optional[str] = None
    schema_ast: Optional[Node] = None
