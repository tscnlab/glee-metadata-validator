import os
import unittest

from gleam_validator import (
    parse_collection_timestamp,
    parse_timestamps_from_rows,
    validate_against_json_schema,
)


SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "schemas",
    "3.0.0",
    "dataset.schema.json",
)


def dataset_record(datetime_metadata):
    return {
        "schema_version": "3.0.0",
        "dataset_internal_id": "DS001",
        "dataset_participant_associated": True,
        "dataset_crossref": {
            "dataset_crossref_study_id": "STUDY001",
            "dataset_crossref_participant_id": "P001",
        },
        "dataset_timezone": "Europe/Berlin",
        "dataset_location": [48.5216, 9.0576],
        "dataset_variable_terms": [{"term": "other", "label": "Other"}],
        "dataset_file": [
            {
                "dataset_file_modality": ["light"],
                "dataset_file_crossref_device_id": "D001",
                "dataset_file_device_location": "non-dominant wrist",
                "dataset_file_device_location_type": "body_worn",
                "dataset_file_temporal_resolution": {
                    "resolution_type": "fixed_interval",
                    "value": 60,
                    "unit": "second",
                },
                "dataset_file_instructions": "Complete the recording protocol.",
                "dataset_file_names": ["data/datasets/light.csv"],
                "dataset_file_format": "csv",
                "dataset_file_encoding": ["UTF-8"],
                "dataset_file_timezone": "Europe/Berlin",
                "dataset_file_datetime": datetime_metadata,
                "dataset_file_role": "primary",
                "dataset_file_data_state": "raw",
                "dataset_file_preprocessing": {
                    "dataset_file_preprocessing_bol": False,
                    "dataset_file_preprocessing_desc": None,
                },
                "dataset_file_variables": [
                    {
                        "dataset_file_variables_name": "timestamp",
                        "dataset_file_variables_labels": "Timestamp",
                        "dataset_file_variables_units": "ISO 8601",
                        "dataset_file_variables_type": "string",
                        "dataset_file_variables_calibration": None,
                        "dataset_file_variables_term": {
                            "variable_term": "other",
                            "variable_name": "timestamp",
                        },
                    }
                ],
                "primary_variables": ["timestamp"],
            }
        ],
    }


class Schema3FileDatetimeTests(unittest.TestCase):
    def schema_errors(self, datetime_metadata):
        return validate_against_json_schema(
            [dataset_record(datetime_metadata)],
            SCHEMA_PATH,
            "datasets",
        )

    def test_accepts_column_datetime_metadata(self):
        errors = self.schema_errors(
            {
                "dataset_file_datetime_source": "column",
                "dataset_file_datetime_date": "timestamp",
                "dataset_file_datetime_dateformat": "YYYY-MM-DD HH:mm:ss",
                "dataset_file_datetime_time": None,
                "dataset_file_datetime_timeformat": None,
            }
        )

        self.assertEqual(errors, [])

    def test_accepts_collection_datetime_metadata(self):
        errors = self.schema_errors(
            {
                "dataset_file_datetime_source": "collection",
                "dataset_file_datetime_date": "2026-07-13",
                "dataset_file_datetime_dateformat": "YYYY-MM-DD",
                "dataset_file_datetime_time": None,
                "dataset_file_datetime_timeformat": None,
            }
        )

        self.assertEqual(errors, [])

    def test_rejects_missing_datetime_source(self):
        metadata = {
            "dataset_file_datetime_date": "timestamp",
            "dataset_file_datetime_dateformat": "YYYY-MM-DD HH:mm:ss",
        }

        errors = self.schema_errors(metadata)

        self.assertTrue(any("dataset_file_datetime_source" in error["message"] for error in errors))

    def test_rejects_time_column_for_collection_source(self):
        metadata = {
            "dataset_file_datetime_source": "collection",
            "dataset_file_datetime_date": "2026-07-13",
            "dataset_file_datetime_dateformat": "YYYY-MM-DD",
            "dataset_file_datetime_time": "time",
            "dataset_file_datetime_timeformat": "HH:mm:ss",
        }

        errors = self.schema_errors(metadata)

        self.assertTrue(any("is not of type 'null'" in error["message"] for error in errors))

    def test_parses_column_and_collection_sources(self):
        column_timestamps, column_errors, checked, config_errors = parse_timestamps_from_rows(
            [{"timestamp": "2026-07-13 09:30:00"}],
            {
                "dataset_file_datetime_date": "timestamp",
                "dataset_file_datetime_dateformat": "YYYY-MM-DD HH:mm:ss",
            },
            field_prefix="dataset_file_datetime",
        )
        collection_timestamps, collection_errors, collection_checked, collection_config = parse_collection_timestamp(
            {
                "dataset_file_datetime_date": "2026-07-13",
                "dataset_file_datetime_dateformat": "YYYY-MM-DD",
            }
        )

        self.assertEqual(len(column_timestamps), 1)
        self.assertEqual((column_errors, checked, config_errors), (0, 1, []))
        self.assertEqual(len(collection_timestamps), 1)
        self.assertEqual((collection_errors, collection_checked, collection_config), (0, 1, []))


if __name__ == "__main__":
    unittest.main()
