import os
import unittest

from glc_validator import validate_against_json_schema
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
        "dataset_file_datetime_date": "2026-07-15",
        "dataset_file_datetime_dateformat": "YYYY-MM-DD",
        "dataset_file_datetime_time": None,
        "dataset_file_datetime_timeformat": None,
    }


class Schema3FileModalityTests(unittest.TestCase):
    def record(self):
        return dataset_record(collection_datetime())

    def errors(self, record):
        return validate_against_json_schema([record], SCHEMA_PATH, "datasets")

    def test_accepts_multiple_sensor_modalities(self):
        record = self.record()
        record["dataset_file"][0]["dataset_file_modality"] = [
            "light", "accelerometry", "temperature"
        ]

        self.assertEqual(self.errors(record), [])

    def test_rejects_sensor_and_non_device_modalities_together(self):
        record = self.record()
        record["dataset_file"][0]["dataset_file_modality"] = ["light", "questionnaire"]

        self.assertTrue(self.errors(record))

    def test_non_device_modality_omits_device_metadata(self):
        record = self.record()
        file_group = record["dataset_file"][0]
        file_group["dataset_file_modality"] = ["questionnaire"]
        file_group["dataset_file_instrument"] = {
            "instrument_type": "questionnaire",
            "instrument_name": "MCTQ",
            "collection_method": "paper",
            "recorded_by": "participant",
        }
        for field in (
            "dataset_file_crossref_device_id",
            "dataset_file_device_location",
            "dataset_file_device_location_type",
        ):
            del file_group[field]

        self.assertEqual(self.errors(record), [])

        file_group["dataset_file_crossref_device_id"] = "D001"
        self.assertTrue(self.errors(record))

        file_group["dataset_file_device_location"] = "non-dominant wrist"
        file_group["dataset_file_device_location_type"] = "body_worn"
        self.assertEqual(self.errors(record), [])

    def test_sensor_modality_requires_device_metadata(self):
        record = self.record()
        del record["dataset_file"][0]["dataset_file_crossref_device_id"]

        self.assertTrue(self.errors(record))

    def test_other_must_be_classified_and_only_combined_with_compatible_modalities(self):
        record = self.record()
        file_group = record["dataset_file"][0]
        file_group["dataset_file_modality"] = ["other"]
        file_group["dataset_file_modality_other"] = "Electrodermal activity"
        file_group["dataset_file_modality_other_type"] = "sensor"
        self.assertEqual(self.errors(record), [])

        file_group["dataset_file_modality"].append("light")
        self.assertEqual(self.errors(record), [])

        file_group["dataset_file_modality"].append("questionnaire")
        self.assertTrue(self.errors(record))

        file_group["dataset_file_modality"] = ["other"]
        file_group["dataset_file_modality_other_type"] = "non_device"
        for field in (
            "dataset_file_crossref_device_id",
            "dataset_file_device_location",
            "dataset_file_device_location_type",
        ):
            del file_group[field]
        self.assertEqual(self.errors(record), [])

    def test_wear_log_allows_no_device_but_rejects_partial_device_metadata(self):
        record = self.record()
        file_group = record["dataset_file"][0]
        file_group["dataset_file_modality"] = ["wear_log"]
        file_group["dataset_file_instrument"] = {
            "instrument_type": "wear_log",
            "instrument_name": "Study wear log",
            "collection_method": "software",
            "software_name": "REDCap",
            "recorded_by": "participant",
        }
        self.assertEqual(self.errors(record), [])

        for field in (
            "dataset_file_crossref_device_id",
            "dataset_file_device_location",
            "dataset_file_device_location_type",
        ):
            del file_group[field]
        self.assertEqual(self.errors(record), [])

        file_group["dataset_file_device_location"] = "head"
        self.assertTrue(self.errors(record))

    def test_non_sensor_instrument_method_rules(self):
        record = self.record()
        file_group = record["dataset_file"][0]
        file_group["dataset_file_modality"] = ["diary"]
        file_group["dataset_file_instrument"] = {
            "instrument_type": "diary",
            "instrument_name": "Sleep diary",
            "collection_method": "software",
            "recorded_by": "participant",
        }
        for field in (
            "dataset_file_crossref_device_id",
            "dataset_file_device_location",
            "dataset_file_device_location_type",
        ):
            del file_group[field]
        self.assertTrue(self.errors(record))

        file_group["dataset_file_instrument"]["software_name"] = "REDCap"
        self.assertEqual(self.errors(record), [])

        file_group["dataset_file_instrument"]["recorded_by"] = "other"
        self.assertTrue(self.errors(record))

        file_group["dataset_file_instrument"]["recorded_by_other"] = "Teacher"
        self.assertEqual(self.errors(record), [])

        del file_group["dataset_file_instrument"]["recorded_by_other"]
        del file_group["dataset_file_instrument"]["recorded_by"]
        self.assertTrue(self.errors(record))

    def test_rejects_multiple_non_sensor_modalities(self):
        record = self.record()
        file_group = record["dataset_file"][0]
        file_group["dataset_file_modality"] = ["questionnaire", "diary"]
        file_group["dataset_file_instrument"] = {
            "instrument_type": "questionnaire",
            "instrument_name": "Combined form",
            "collection_method": "paper",
            "recorded_by": "study_staff",
        }
        self.assertTrue(self.errors(record))

    def test_rejects_duplicate_or_empty_modality_array(self):
        record = self.record()
        record["dataset_file"][0]["dataset_file_modality"] = ["light", "light"]
        self.assertTrue(self.errors(record))

        record["dataset_file"][0]["dataset_file_modality"] = []
        self.assertTrue(self.errors(record))


if __name__ == "__main__":
    unittest.main()
