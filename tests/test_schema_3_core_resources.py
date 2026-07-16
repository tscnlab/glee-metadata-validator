import json
import os
import unittest

from gleam_validator import validate_against_json_schema


ROOT = os.path.dirname(os.path.dirname(__file__))
SCHEMAS = os.path.join(ROOT, "schemas", "3.0.0")
FIXTURE_DATA = os.path.join(ROOT, "tests", "fixtures", "schema-3.0.0", "pass", "data")


class Schema3CoreResourceTests(unittest.TestCase):
    def load_json(self, filename):
        with open(os.path.join(FIXTURE_DATA, filename), encoding="utf-8") as fixture_file:
            return json.load(fixture_file)

    def schema_errors(self, data, schema_name, label):
        return validate_against_json_schema(
            data,
            os.path.join(SCHEMAS, schema_name),
            label,
        )

    def test_comprehensive_study_with_contributor_passes(self):
        studies = self.load_json("study.json")
        self.assertEqual(self.schema_errors(studies, "study.schema.json", "study"), [])

    def test_study_allows_nullable_optional_sections(self):
        study = self.load_json("study.json")[0]
        for field in (
            "study_preregistration",
            "study_ethics",
            "study_registration",
            "study_groups",
            "study_intervention",
            "study_contributors",
            "study_type",
            "study_funding_sources",
            "study_keywords",
        ):
            study[field] = None
        self.assertEqual(self.schema_errors([study], "study.schema.json", "study"), [])

    def test_contributor_institution_requires_name_and_country(self):
        study = self.load_json("study.json")[0]
        del study["study_contributors"][0]["contributor_institution"]["contributor_institution_country"]
        errors = self.schema_errors([study], "study.schema.json", "study")
        self.assertTrue(any("contributor_institution_country" in error["message"] for error in errors))

    def test_device_optional_firmware_and_sensors_may_be_null(self):
        device = self.load_json("devices.json")[0]
        device["device_firmware_version"] = None
        device["device_sensors"] = None
        self.assertEqual(self.schema_errors([device], "device.schema.json", "devices"), [])

    def test_device_sensor_requires_type_and_rejects_unknown_fields(self):
        device = self.load_json("devices.json")[0]
        device["device_sensors"] = [{"device_sensor_datasheet_id": None}]
        errors = self.schema_errors([device], "device.schema.json", "devices")
        self.assertTrue(any("device_sensor_type" in error["message"] for error in errors))

        device["device_sensors"] = [{"device_sensor_type": "light", "unknown": "value"}]
        errors = self.schema_errors([device], "device.schema.json", "devices")
        self.assertTrue(any("Additional properties" in error["message"] for error in errors))

    def test_device_calibration_date_accepts_null_and_rejects_bad_shape(self):
        device = self.load_json("devices.json")[0]
        device["device_calibration_date"] = None
        self.assertEqual(self.schema_errors([device], "device.schema.json", "devices"), [])

        device["device_calibration_date"] = "16-07-2026"
        self.assertTrue(self.schema_errors([device], "device.schema.json", "devices"))


if __name__ == "__main__":
    unittest.main()
