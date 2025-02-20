# system packages
import logging
from pathlib import Path

# internal packages
from graphdoc import LocalDataHelper
from graphdoc import SchemaObject

# external packages
from graphql import DocumentNode

# logging
log = logging.getLogger(__name__)

# global variables
BASE_DIR = Path(__file__).resolve().parent.parent.parent
SCHEMA_DIR = BASE_DIR / "tests" / "assets" / "schemas"


class TestLocalDataHelper:
    def test_schema_objects_from_folder(self, default_ldh: LocalDataHelper):
        folder_path = Path(SCHEMA_DIR) / "perfect"
        ldh = default_ldh
        schema_objects = ldh.schema_objects_from_folder(
            category="perfect", rating=4, folder_path=folder_path
        )
        assert len(schema_objects) == 1
        for value in schema_objects.values():
            assert isinstance(value, SchemaObject)

    def test_schema_objects_from_folder_of_folders(self, default_ldh: LocalDataHelper):
        ldh = default_ldh
        ldh.schema_directory_path = SCHEMA_DIR
        schema_objects = ldh.schema_objects_from_folder_of_folders()
        assert schema_objects is not None
        assert len(schema_objects) == 4
