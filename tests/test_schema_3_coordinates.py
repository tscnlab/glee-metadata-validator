import os
import unittest

from gleam_validator import validate_against_json_schema
from tests.test_schema_3_file_datetime import dataset_record


SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "schemas",
    "3.0.0",
    "dataset.schema.json",
)


class Schema3CoordinateTests(unittest.TestCase):
    def errors_for_location(self, location):
        record = dataset_record(
            {
                "dataset_file_datetime_source": "collection",
                "dataset_file_datetime_date": "2026-07-13",
                "dataset_file_datetime_dateformat": "YYYY-MM-DD",
                "dataset_file_datetime_time": None,
                "dataset_file_datetime_timeformat": None,
            }
        )
        record["dataset_location"] = location
        return validate_against_json_schema([record], SCHEMA_PATH, "datasets")

    def test_accepts_numeric_coordinates(self):
        self.assertEqual(self.errors_for_location([48.5216, 9.0576]), [])

    def test_rejects_string_coordinates(self):
        errors = self.errors_for_location(["48.5216", "9.0576"])

        self.assertEqual(len(errors), 2)
        self.assertTrue(all("is not of type 'number'" in error["message"] for error in errors))

    def test_requires_exactly_two_coordinates(self):
        errors = self.errors_for_location([48.5216])

        self.assertTrue(any("is too short" in error["message"] for error in errors))


if __name__ == "__main__":
    unittest.main()
