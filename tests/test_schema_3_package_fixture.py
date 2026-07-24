import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import glc_validator


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
                    exit_code = glc_validator.validate_crossrefs(
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
        self.assertTrue(
            any(
                file["path"] == "data/participant_characteristics.csv" and file["sha256"]
                for file in manifest["files"]
            )
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
                    "Device export,Lumitech LT-100\n"
                    "Generated for schema test,2026-07-16\n"
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
                    "Device export,Lumitech LT-100\n"
                    "Generated for schema test,2026-07-16\n"
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

    def test_participant_characteristic_participant_id_is_checked_end_to_end(self):
        def invalidate_characteristic_crossref(package_root):
            characteristics_path = os.path.join(
                package_root,
                "data",
                "participant_characteristics.csv",
            )
            with open(characteristics_path, "w", encoding="utf-8") as characteristics_file:
                characteristics_file.write(
                    "participant_internal_id,participant_characteristic_name,"
                    "participant_characteristic_value,participant_characteristic_unit,"
                    "participant_characteristic_description\n"
                    "UNKNOWN,chronotype,intermediate,category,Invalid participant link\n"
                )

        exit_code, report, _ = self.validate_fixture(mutate=invalidate_characteristic_crossref)
        messages = [error["message"] for error in report["errors"]]
        self.assertEqual(exit_code, 1)
        self.assertTrue(any("Unknown participant ID" in message for message in messages))

    def test_optional_participant_characteristics_resource_may_be_absent(self):
        def remove_characteristics_resource(package_root):
            datapackage_path = os.path.join(package_root, "datapackage.json")
            with open(datapackage_path, encoding="utf-8") as datapackage_file:
                datapackage = json.load(datapackage_file)
            datapackage["resources"] = [
                resource
                for resource in datapackage["resources"]
                if resource.get("name") != "participant_characteristics"
            ]
            with open(datapackage_path, "w", encoding="utf-8") as datapackage_file:
                json.dump(datapackage, datapackage_file, indent=2)

        exit_code, report, _ = self.validate_fixture(mutate=remove_characteristics_resource)
        self.assertEqual(exit_code, 0, report["errors"])

    def test_study_contributor_orcid_is_optional_end_to_end(self):
        def remove_contributor_orcid(package_root):
            study_path = os.path.join(package_root, "data", "study.json")
            with open(study_path, encoding="utf-8") as study_file:
                studies = json.load(study_file)
            del studies[0]["study_contributors"][0]["contributor_orcid"]
            with open(study_path, "w", encoding="utf-8") as study_file:
                json.dump(studies, study_file, indent=2)

        exit_code, report, _ = self.validate_fixture(mutate=remove_contributor_orcid)
        self.assertEqual(exit_code, 0, report["errors"])

    def test_study_contributor_orcid_must_be_string_when_supplied(self):
        def set_invalid_contributor_orcid(package_root):
            study_path = os.path.join(package_root, "data", "study.json")
            with open(study_path, encoding="utf-8") as study_file:
                studies = json.load(study_file)
            studies[0]["study_contributors"][0]["contributor_orcid"] = 123
            with open(study_path, "w", encoding="utf-8") as study_file:
                json.dump(studies, study_file, indent=2)

        exit_code, report, _ = self.validate_fixture(
            mutate=set_invalid_contributor_orcid
        )
        self.assertEqual(exit_code, 1)
        self.assertTrue(
            any(
                error.get("path", [])[-1:] == ["contributor_orcid"]
                for error in report["errors"]
            )
        )

    def test_participant_constraints_are_checked_end_to_end(self):
        def set_invalid_age(package_root):
            participants_path = os.path.join(package_root, "data", "participants.csv")
            with open(participants_path, "w", encoding="utf-8") as participants_file:
                participants_file.write(
                    "participant_internal_id,participant_age,participant_sex,participant_gender\n"
                    "P001,121,female,woman\n"
                )

        exit_code, report, _ = self.validate_fixture(mutate=set_invalid_age)
        messages = [error["message"] for error in report["errors"]]
        self.assertEqual(exit_code, 1)
        self.assertTrue(any("participant_age" in message or "maximum" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
