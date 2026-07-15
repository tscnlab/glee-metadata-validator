import os
import unittest

from gleam_validator import validate_against_json_schema, validate_primary_variables_subset
from test_schema_3_file_datetime import dataset_record


SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "schemas",
    "3.0.0",
    "dataset.schema.json",
)


def collection_datetime():
    return {
        "dataset_file_datetime_source": "collection",
        "dataset_file_datetime_date": "2026-07-14",
        "dataset_file_datetime_dateformat": "YYYY-MM-DD",
        "dataset_file_datetime_time": None,
        "dataset_file_datetime_timeformat": None,
    }


class Schema3FileRoleProcessingTests(unittest.TestCase):
    def errors_for(self, record):
        return validate_against_json_schema([record], SCHEMA_PATH, "datasets")

    def test_primary_role_requires_primary_variables(self):
        record = dataset_record(collection_datetime())
        del record["dataset_file"][0]["primary_variables"]

        errors = self.errors_for(record)

        self.assertTrue(any("primary_variables" in error["message"] for error in errors))

        record["dataset_file"][0]["primary_variables"] = None
        self.assertTrue(self.errors_for(record))

    def test_supporting_role_allows_optional_primary_variables(self):
        with_primary = dataset_record(collection_datetime())
        with_primary["dataset_file"][0]["dataset_file_role"] = "supporting"
        without_primary = dataset_record(collection_datetime())
        without_primary["dataset_file"][0]["dataset_file_role"] = "supporting"
        del without_primary["dataset_file"][0]["primary_variables"]

        self.assertEqual(self.errors_for(with_primary), [])
        self.assertEqual(self.errors_for(without_primary), [])
        self.assertEqual(validate_primary_variables_subset([with_primary]), [])

    def test_raw_requires_false_preprocessing_status_and_no_description(self):
        record = dataset_record(collection_datetime())
        preprocessing = record["dataset_file"][0]["dataset_file_preprocessing"]
        preprocessing["dataset_file_preprocessing_bol"] = True
        preprocessing["dataset_file_preprocessing_desc"] = ["Filtered"]

        errors = self.errors_for(record)

        self.assertTrue(errors)

    def test_processed_requires_true_status_and_non_empty_description(self):
        record = dataset_record(collection_datetime())
        file_group = record["dataset_file"][0]
        file_group["dataset_file_data_state"] = "processed"
        preprocessing = file_group["dataset_file_preprocessing"]
        preprocessing["dataset_file_preprocessing_bol"] = True
        preprocessing["dataset_file_preprocessing_desc"] = ["Applied quality filtering"]

        self.assertEqual(self.errors_for(record), [])

        preprocessing["dataset_file_preprocessing_desc"] = []
        self.assertTrue(self.errors_for(record))

    def test_rejects_removed_auxiliary_field(self):
        record = dataset_record(collection_datetime())
        record["dataset_file"][0]["dataset_file_auxiliary"] = False

        errors = self.errors_for(record)

        self.assertTrue(any("Additional properties" in error["message"] for error in errors))


if __name__ == "__main__":
    unittest.main()
