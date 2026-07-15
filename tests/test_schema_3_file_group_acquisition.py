import os
import unittest

from gleam_validator import validate_against_json_schema, validate_dataset_device_ids
from test_schema_3_file_datetime import dataset_record


SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "schemas",
    "3.0.0",
    "dataset.schema.json",
)


class Schema3FileGroupAcquisitionTests(unittest.TestCase):
    def record(self):
        return dataset_record(
            {
                "dataset_file_datetime_source": "collection",
                "dataset_file_datetime_date": "2026-07-13",
                "dataset_file_datetime_dateformat": "YYYY-MM-DD",
                "dataset_file_datetime_time": None,
                "dataset_file_datetime_timeformat": None,
            }
        )

    def test_requires_acquisition_metadata_in_each_file_group(self):
        record = self.record()
        del record["dataset_file"][0]["dataset_file_temporal_resolution"]

        errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")

        self.assertTrue(any("dataset_file_temporal_resolution" in error["message"] for error in errors))

    def test_rejects_old_dataset_level_acquisition_fields(self):
        record = self.record()
        record["dataset_sampling_interval"] = 60

        errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")

        self.assertTrue(any("Additional properties" in error["message"] for error in errors))

    def test_validates_each_file_group_device_reference(self):
        record = self.record()
        second_group = dict(record["dataset_file"][0])
        second_group["dataset_file_crossref_device_id"] = "UNKNOWN"
        record["dataset_file"].append(second_group)

        errors = validate_dataset_device_ids(
            [record],
            [{"device_internal_id": "D001"}],
        )

        self.assertEqual(len(errors), 1)
        self.assertEqual(
            errors[0]["path"],
            ["dataset_file", 1, "dataset_file_crossref_device_id"],
        )

    def test_rejects_obsolete_location_type(self):
        record = self.record()
        record["dataset_file"][0]["dataset_file_device_location_type"] = "other"

        errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")

        self.assertTrue(errors)

    def test_accepts_environmental_location(self):
        record = self.record()
        file_group = record["dataset_file"][0]
        file_group["dataset_file_device_location_type"] = "environmental"
        file_group["dataset_file_device_location"] = "bedroom"

        self.assertEqual(validate_against_json_schema([record], SCHEMA_PATH, "datasets"), [])

    def test_accepts_participant_proximal_location(self):
        record = self.record()
        file_group = record["dataset_file"][0]
        file_group["dataset_file_device_location_type"] = "participant_proximal"
        file_group["dataset_file_device_location"] = "bedside table"

        self.assertEqual(validate_against_json_schema([record], SCHEMA_PATH, "datasets"), [])

    def test_requires_value_and_unit_for_fixed_interval(self):
        record = self.record()
        record["dataset_file"][0]["dataset_file_temporal_resolution"] = {
            "resolution_type": "fixed_interval"
        }

        errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")

        messages = [error["message"] for error in errors]
        self.assertTrue(any("value" in message for message in messages))
        self.assertTrue(any("unit" in message for message in messages))

    def test_accepts_event_based_without_value_or_unit(self):
        record = self.record()
        record["dataset_file"][0]["dataset_file_temporal_resolution"] = {
            "resolution_type": "event_based"
        }

        self.assertEqual(validate_against_json_schema([record], SCHEMA_PATH, "datasets"), [])

    def test_rejects_value_or_unit_for_event_based(self):
        record = self.record()
        record["dataset_file"][0]["dataset_file_temporal_resolution"] = {
            "resolution_type": "event_based",
            "value": 1,
            "unit": "day",
        }

        self.assertTrue(validate_against_json_schema([record], SCHEMA_PATH, "datasets"))


if __name__ == "__main__":
    unittest.main()
