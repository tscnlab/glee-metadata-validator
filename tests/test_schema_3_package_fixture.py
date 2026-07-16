import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import gleam_validator


FIXTURE_ROOT = os.path.join(
    os.path.dirname(__file__),
    "fixtures",
    "schema-3.0.0",
    "pass",
)


class Schema3PackageFixtureTests(unittest.TestCase):
    def validate_fixture(self, mutate=None):
        with tempfile.TemporaryDirectory() as directory:
            package_root = os.path.join(directory, "package")
            shutil.copytree(FIXTURE_ROOT, package_root)
            if mutate:
                mutate(package_root)

            report_path = os.path.join(directory, "validation.json")
            manifest_path = os.path.join(directory, "manifest.json")
            environment = {
                "VALIDATION_JSON": report_path,
                "VALIDATION_MANIFEST": manifest_path,
            }
            with patch.dict(os.environ, environment, clear=False):
                with contextlib.redirect_stdout(io.StringIO()):
                    exit_code = gleam_validator.validate_crossrefs(
                        os.path.join(package_root, "datapackage.json")
                    )

            with open(report_path, encoding="utf-8") as report_file:
                report = json.load(report_file)
            with open(manifest_path, encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            return exit_code, report, manifest

    def test_complete_schema_3_fixture_passes(self):
        exit_code, report, manifest = self.validate_fixture()

        self.assertEqual(exit_code, 0, report["errors"])
        self.assertEqual(report["status"], "pass")
        self.assertEqual(report["schema_version"], "3.0.0")
        self.assertEqual(report["error_count"], 0)
        self.assertNotIn("column_check_mode", report)
        self.assertEqual(manifest["schema_version"], "1.0")
        self.assertTrue(
            any(file["path"] == "data/datasets/light.csv" and file["sha256"] for file in manifest["files"])
        )

    def test_invalid_schema_3_fixture_fails_with_specific_errors(self):
        def make_invalid(package_root):
            datasets_path = os.path.join(package_root, "data", "datasets.json")
            with open(datasets_path, encoding="utf-8") as datasets_file:
                datasets = json.load(datasets_file)
            datasets[0]["dataset_timezone"] = "Europe/NotAPlace"
            datasets[0]["dataset_location"] = ["52.52", "13.405"]
            del datasets[0]["dataset_file"][0]["dataset_file_datetime"]["dataset_file_datetime_source"]
            with open(datasets_path, "w", encoding="utf-8") as datasets_file:
                json.dump(datasets, datasets_file, indent=2)

        exit_code, report, _ = self.validate_fixture(mutate=make_invalid)

        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "fail")
        messages = [error["message"] for error in report["errors"]]
        self.assertTrue(any("Invalid IANA timezone" in message for message in messages))
        self.assertTrue(any("is not of type 'number'" in message for message in messages))
        self.assertTrue(any("dataset_file_datetime_source" in message for message in messages))

    def test_data_file_columns_types_and_warnings_are_checked(self):
        def add_extra_column(package_root):
            data_path = os.path.join(package_root, "data", "datasets", "light.csv")
            with open(data_path, "w", encoding="utf-8") as data_file:
                data_file.write(
                    "timestamp,illuminance,undeclared\n"
                    "2026-07-14 08:00:00,12.5,x\n"
                    "2026-07-14 08:01:00,13.0,y\n"
                )

        exit_code, report, _ = self.validate_fixture(mutate=add_extra_column)
        self.assertEqual(exit_code, 0, report["errors"])
        self.assertTrue(any("undeclared extra columns" in warning["message"] for warning in report["warnings"]))

        def invalidate_file(package_root):
            data_path = os.path.join(package_root, "data", "datasets", "light.csv")
            with open(data_path, "w", encoding="utf-8") as data_file:
                data_file.write(
                    "timestamp,wrong_column\n"
                    "not-a-date,not-a-number\n"
                )

        exit_code, report, _ = self.validate_fixture(mutate=invalidate_file)
        messages = [error["message"] for error in report["errors"]]
        self.assertEqual(exit_code, 1)
        self.assertTrue(any("missing declared columns" in message for message in messages))
        self.assertTrue(any("datetime values that do not match" in message for message in messages))

    def test_cross_resource_ids_are_checked_end_to_end(self):
        def invalidate_crossrefs(package_root):
            datasets_path = os.path.join(package_root, "data", "datasets.json")
            with open(datasets_path, encoding="utf-8") as datasets_file:
                datasets = json.load(datasets_file)
            datasets[0]["dataset_crossref"]["dataset_crossref_study_id"] = "UNKNOWN-STUDY"
            datasets[0]["dataset_crossref"]["dataset_crossref_participant_id"] = "UNKNOWN-PARTICIPANT"
            datasets[0]["dataset_file"][0]["dataset_file_crossref_device_id"] = "UNKNOWN-DEVICE"
            with open(datasets_path, "w", encoding="utf-8") as datasets_file:
                json.dump(datasets, datasets_file, indent=2)

        exit_code, report, _ = self.validate_fixture(mutate=invalidate_crossrefs)
        messages = [error["message"] for error in report["errors"]]
        self.assertEqual(exit_code, 1)
        self.assertTrue(any("Invalid study ID" in message for message in messages))
        self.assertTrue(any("Invalid participant ID" in message for message in messages))
        self.assertTrue(any("Invalid device ID" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
