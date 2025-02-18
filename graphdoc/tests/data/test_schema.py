# system packages
import logging

# internal packages
from graphdoc import SchemaCategory, SchemaRating, SchemaType, SchemaObject

# external packages

# logging
log = logging.getLogger(__name__)


class TestSchema:
    def test_schema_category_enum(self):
        assert SchemaCategory.PERFECT == "perfect"
        assert SchemaCategory.ALMOST_PERFECT == "almost perfect"
        assert SchemaCategory.POOR_BUT_CORRECT == "poor but correct"
        assert SchemaCategory.INCORRECT == "incorrect"
        assert SchemaCategory.BLANK == "blank"

    def test_schema_category_from_str(self):
        assert SchemaCategory.from_str("perfect") == SchemaCategory.PERFECT
        assert (
            SchemaCategory.from_str("almost perfect") == SchemaCategory.ALMOST_PERFECT
        )
        assert (
            SchemaCategory.from_str("poor but correct")
            == SchemaCategory.POOR_BUT_CORRECT
        )
        assert SchemaCategory.from_str("incorrect") == SchemaCategory.INCORRECT
        assert SchemaCategory.from_str("blank") == SchemaCategory.BLANK
        assert SchemaCategory.from_str("invalid") is None

    def test_schema_rating_enum(self):
        assert SchemaRating.FOUR == "4"
        assert SchemaRating.THREE == "3"
        assert SchemaRating.TWO == "2"
        assert SchemaRating.ONE == "1"
        assert SchemaRating.ZERO == "0"

    def test_schema_rating_from_value(self):
        assert SchemaRating.from_value("4") == SchemaRating.FOUR
        assert SchemaRating.from_value(3) == SchemaRating.THREE
        assert SchemaRating.from_value(2) == SchemaRating.TWO
        assert SchemaRating.from_value(1) == SchemaRating.ONE
        assert SchemaRating.from_value(0) == SchemaRating.ZERO
        assert SchemaRating.from_value("invalid") is None

    def test_schema_type_enum(self):
        assert SchemaType.FULL_SCHEMA == "full schema"
        assert SchemaType.TABLE_SCHEMA == "table schema"
        assert SchemaType.ENUM_SCHEMA == "enum schema"

    def test_schema_type_from_str(self):
        assert SchemaType.from_str("full schema") == SchemaType.FULL_SCHEMA
        assert SchemaType.from_str("table schema") == SchemaType.TABLE_SCHEMA
        assert SchemaType.from_str("enum schema") == SchemaType.ENUM_SCHEMA
        assert SchemaType.from_str("invalid") is None
