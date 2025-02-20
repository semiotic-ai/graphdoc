# system packages
import logging

# internal packages
from graphdoc import (
    SchemaCategory,
    SchemaRating,
    SchemaType,
    SchemaObject,
    SchemaCategoryRatingMapping,
)

# external packages
from graphql import parse
from datasets import Dataset

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

    def test_schema_category_rating_mapping(self):
        assert (
            SchemaCategoryRatingMapping.get_rating(SchemaCategory.PERFECT)
            == SchemaRating.FOUR
        )
        assert (
            SchemaCategoryRatingMapping.get_rating(SchemaCategory.ALMOST_PERFECT)
            == SchemaRating.THREE
        )
        assert (
            SchemaCategoryRatingMapping.get_rating(SchemaCategory.POOR_BUT_CORRECT)
            == SchemaRating.TWO
        )
        assert (
            SchemaCategoryRatingMapping.get_rating(SchemaCategory.INCORRECT)
            == SchemaRating.ONE
        )
        assert (
            SchemaCategoryRatingMapping.get_rating(SchemaCategory.BLANK)
            == SchemaRating.ZERO
        )

    def test_schema_object_from_dict(self):
        schema_ast = parse("type Account @entity { id: Bytes! }")
        data = {
            "key": "test",
            "category": SchemaCategory.PERFECT,
            "rating": SchemaRating.FOUR,
            "schema_name": "test",
            "schema_type": SchemaType.FULL_SCHEMA,
            "schema_str": "test",
            "schema_ast": schema_ast,
        }
        schema_object = SchemaObject.from_dict(data)
        assert schema_object.key == "test"
        assert schema_object.category == SchemaCategory.PERFECT
        assert schema_object.rating == SchemaRating.FOUR
        assert schema_object.schema_name == "test"
        assert schema_object.schema_type == SchemaType.FULL_SCHEMA
        assert schema_object.schema_str == "test"
        assert schema_object.schema_ast == schema_ast

    def test_schema_object_to_dict(self):
        schema_ast = parse("type Account @entity { id: Bytes! }")
        schema_object = SchemaObject(
            key="test",
            category=SchemaCategory.PERFECT,
            rating=SchemaRating.FOUR,
            schema_name="test",
            schema_type=SchemaType.FULL_SCHEMA,
            schema_str="test",
            schema_ast=schema_ast,
        )

        assert schema_object.to_dict() == {
            "category": "perfect",
            "rating": "4",
            "schema_name": "test",
            "schema_type": "full schema",
            "schema_str": "test",
            "schema_ast": schema_ast,
        }

    def test_schema_object_to_dataset(self):
        schema_ast = parse("type Account @entity { id: Bytes! }")
        schema_object = SchemaObject(
            key="test",
            category=SchemaCategory.PERFECT,
            rating=SchemaRating.FOUR,
            schema_name="test",
            schema_type=SchemaType.FULL_SCHEMA,
            schema_str="test",
            schema_ast=schema_ast,
        )

        dataset = schema_object.to_dataset()
        assert dataset.num_rows == 1
        assert isinstance(dataset, Dataset)
