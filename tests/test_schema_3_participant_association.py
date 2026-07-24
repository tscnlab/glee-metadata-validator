import os
import unittest

from glc_validator import validate_against_json_schema
from test_schema_3_file_datetime import dataset_record
from test_schema_3_file_modality import collection_datetime


SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "schemas",
    "3.0.0",
    "dataset.schema.json",
)


class Schema3ParticipantAssociationTests(unittest.TestCase):
    def errors(self, record):
        return validate_against_json_schema([record], SCHEMA_PATH, "datasets")

    def record(self):
        return dataset_record(collection_datetime())

    def test_participant_associated_requires_participant_id(self):
        record = self.record()
        record["dataset_crossref"]["dataset_crossref_participant_id"] = None

        self.assertTrue(self.errors(record))

    def test_non_participant_requires_null_participant_id(self):
        record = self.record()
        record["dataset_participant_associated"] = False

        self.assertTrue(self.errors(record))

        record["dataset_crossref"]["dataset_crossref_participant_id"] = None
        record["dataset_file"][0]["dataset_file_device_location_type"] = "environmental"
        record["dataset_file"][0]["dataset_file_device_location"] = "building rooftop"
        self.assertEqual(self.errors(record), [])

    def test_non_participant_rejects_body_worn_or_proximal_location(self):
        for location_type in ("body_worn", "participant_proximal"):
            with self.subTest(location_type=location_type):
                record = self.record()
                record["dataset_participant_associated"] = False
                record["dataset_crossref"]["dataset_crossref_participant_id"] = None
                record["dataset_file"][0]["dataset_file_device_location_type"] = location_type

                self.assertTrue(self.errors(record))

    def test_participant_associated_questionnaire_needs_no_device(self):
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

    def test_participant_association_field_is_required(self):
        record = self.record()
        del record["dataset_participant_associated"]

        self.assertTrue(self.errors(record))


if __name__ == "__main__":
    unittest.main()
