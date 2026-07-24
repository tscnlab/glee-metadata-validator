import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfoNotFoundError

from glc_validator import validate_dataset_timezones


class Schema3TimezoneTests(unittest.TestCase):
    def dataset(self, dataset_timezone="Europe/Berlin", file_timezone="UTC"):
        return {
            "dataset_internal_id": "DS001",
            "dataset_timezone": dataset_timezone,
            "dataset_file": [
                {"dataset_file_timezone": file_timezone},
            ],
        }

    def test_accepts_iana_names_and_utc(self):
        errors = validate_dataset_timezones([self.dataset()], "3.0.0")

        self.assertEqual(errors, [])

    def test_rejects_invalid_dataset_and_file_timezones(self):
        errors = validate_dataset_timezones(
            [self.dataset("Europe/NotAPlace", "Mars/Olympus_Mons")],
            "3.0.0",
        )

        self.assertEqual(len(errors), 2)
        self.assertEqual(errors[0]["path"], ["dataset_timezone"])
        self.assertEqual(errors[1]["path"], ["dataset_file", 0, "dataset_file_timezone"])

    def test_skips_iana_check_for_legacy_schema_versions(self):
        errors = validate_dataset_timezones(
            [self.dataset("Not/AZone", "Still/NotAZone")],
            "2.0.0",
        )

        self.assertEqual(errors, [])

    def test_reports_missing_runtime_timezone_database_once(self):
        with patch("glc_validator.ZoneInfo", side_effect=ZoneInfoNotFoundError):
            errors = validate_dataset_timezones([self.dataset()], "3.0.0")

        self.assertEqual(len(errors), 1)
        self.assertIn("database is unavailable", errors[0]["message"])


if __name__ == "__main__":
    unittest.main()
