import contextlib
import io
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from glc_validator import CORE_REQUIRED, validate_crossrefs


class MissingResourceTests(unittest.TestCase):
    def run_validator(self, directory, descriptor):
        datapackage_path = os.path.join(directory, "datapackage.json")
        report_path = os.path.join(directory, "validation.json")
        manifest_path = os.path.join(directory, "manifest.json")

        with open(datapackage_path, "w", encoding="utf-8") as package_file:
            json.dump(descriptor, package_file)

        environment = {
            "VALIDATION_JSON": report_path,
            "VALIDATION_MANIFEST": manifest_path,
        }
        with patch.dict(os.environ, environment, clear=False):
            with contextlib.redirect_stdout(io.StringIO()):
                exit_code = validate_crossrefs(datapackage_path)

        with open(report_path, encoding="utf-8") as report_file:
            return exit_code, json.load(report_file)

    def test_absent_core_resources_each_produce_one_high_level_error(self):
        with tempfile.TemporaryDirectory() as directory:
            exit_code, report = self.run_validator(
                directory,
                {
                    "name": "missing-resources",
                    "profile": "data-package",
                    "schema_version": "2.0.0",
                    "resources": [],
                },
            )

        self.assertEqual(exit_code, 1)
        missing_errors = [
            error
            for error in report["errors"]
            if error["message"].endswith("Required resource is missing")
        ]
        self.assertEqual(len(missing_errors), len(CORE_REQUIRED))
        self.assertEqual(
            {
                error["message"].split("]", 1)[0].lstrip("[")
                for error in missing_errors
            },
            CORE_REQUIRED,
        )
        self.assertFalse(any(error["message"].startswith("[profile]") for error in report["errors"]))

    def test_missing_resource_file_does_not_trigger_error_cascade(self):
        with tempfile.TemporaryDirectory() as directory:
            for filename in ["study.json", "datasets.json", "devices.json", "device_datasheets.json"]:
                with open(os.path.join(directory, filename), "w", encoding="utf-8") as resource_file:
                    json.dump([], resource_file)

            json_resource = lambda name, path: {
                "name": name,
                "path": path,
                "profile": "json-entity-resource.json",
                "mediatype": "application/json",
                "jsonSchema": f"schemas/2.0.0/{name.rstrip('s')}.schema.json",
            }
            descriptor = {
                "name": "missing-participants-file",
                "profile": "data-package",
                "schema_version": "2.0.0",
                "resources": [
                    json_resource("study", "study.json"),
                    {
                        "name": "participants",
                        "path": "participants.csv",
                        "profile": "tabular-data-resource",
                        "mediatype": "text/csv",
                        "schema": {
                            "fields": [
                                {"name": "participant_internal_id", "type": "string"},
                            ]
                        },
                    },
                    json_resource("datasets", "datasets.json"),
                    json_resource("devices", "devices.json"),
                    json_resource("device_datasheets", "device_datasheets.json"),
                ],
            }

            exit_code, report = self.run_validator(directory, descriptor)

        self.assertEqual(exit_code, 1)
        participant_file_errors = [
            error
            for error in report["errors"]
            if error["message"].startswith("[participants] Data path does not exist:")
        ]
        self.assertEqual(len(participant_file_errors), 1, report["errors"])
        self.assertFalse(
            any(error["message"].startswith("[validator] Failed during validation pipeline") for error in report["errors"])
        )


if __name__ == "__main__":
    unittest.main()
