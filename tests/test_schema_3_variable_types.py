import os
import unittest

from glc_validator import validate_against_json_schema, validate_declared_column_values
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


class Schema3VariableTypeTests(unittest.TestCase):
    def variable(self, variable_type, factor_levels=None):
        variable = {
            "dataset_file_variables_name": "value",
            "dataset_file_variables_type": variable_type,
        }
        if factor_levels is not None:
            variable["dataset_file_variables_factor_levels"] = factor_levels
        return variable

    def content_errors(self, rows, variable):
        warnings = []
        errors = validate_declared_column_values(
            rows,
            [variable],
            ["value"],
            "values.csv",
            "[dataset DS001 file 1]",
            ["dataset_file", 0, "dataset_file_variables"],
            warnings,
        )
        return errors, warnings

    def test_schema_requires_variable_type(self):
        record = dataset_record(collection_datetime())
        del record["dataset_file"][0]["dataset_file_variables"][0]["dataset_file_variables_type"]

        errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")

        self.assertTrue(any("dataset_file_variables_type" in error["message"] for error in errors))

    def test_schema_accepts_optional_nonempty_variable_description(self):
        record = dataset_record(collection_datetime())
        variable = record["dataset_file"][0]["dataset_file_variables"][0]
        variable["dataset_file_variables_description"] = "What time did you get into bed?"

        valid_errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")
        self.assertEqual(valid_errors, [])

        variable["dataset_file_variables_description"] = ""
        empty_errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")
        self.assertTrue(empty_errors)
        self.assertTrue(any("dataset_file_variables_description" in error["path"] for error in empty_errors))

    def test_schema_requires_factor_levels_only_for_factor(self):
        record = dataset_record(collection_datetime())
        variable = record["dataset_file"][0]["dataset_file_variables"][0]
        variable["dataset_file_variables_type"] = "factor"
        missing_errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")
        self.assertTrue(any("dataset_file_variables_factor_levels" in error["message"] for error in missing_errors))

        variable["dataset_file_variables_type"] = "string"
        variable["dataset_file_variables_factor_levels"] = [{"value": "a", "label": "A"}]
        prohibited_errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")
        self.assertTrue(prohibited_errors)

    def test_schema_requires_units_only_for_numeric_and_integer_variables(self):
        record = dataset_record(collection_datetime())
        variable = record["dataset_file"][0]["dataset_file_variables"][0]
        variable["dataset_file_variables_type"] = "factor"
        variable["dataset_file_variables_factor_levels"] = [{"value": "a", "label": "A"}]

        factor_errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")
        self.assertEqual(factor_errors, [])

        variable["dataset_file_variables_units"] = "N/A"
        factor_with_units_errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")
        self.assertTrue(factor_with_units_errors)

        variable["dataset_file_variables_type"] = "string"
        variable.pop("dataset_file_variables_factor_levels")
        variable.pop("dataset_file_variables_units")
        string_errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")
        self.assertEqual(string_errors, [])

        variable["dataset_file_variables_type"] = "numeric"
        missing_units_errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")
        self.assertTrue(any("dataset_file_variables_units" in error["message"] for error in missing_units_errors))

        variable["dataset_file_variables_units"] = "Unknown"
        placeholder_errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")
        self.assertTrue(placeholder_errors)

        variable["dataset_file_variables_units"] = "lx"
        numeric_errors = validate_against_json_schema([record], SCHEMA_PATH, "datasets")
        self.assertEqual(numeric_errors, [])

    def test_validates_numeric_integer_and_boolean_values(self):
        numeric_errors, _ = self.content_errors(
            [{"value": "1.5"}, {"value": "bad"}], self.variable("numeric")
        )
        integer_errors, _ = self.content_errors(
            [{"value": "2"}, {"value": "2.0"}], self.variable("integer")
        )
        boolean_errors, _ = self.content_errors(
            [{"value": "TRUE"}, {"value": "0"}, {"value": "1"}, {"value": "no"}],
            self.variable("boolean"),
        )

        self.assertEqual(len(numeric_errors), 1)
        self.assertEqual(len(integer_errors), 1)
        self.assertEqual(len(boolean_errors), 1)

    def test_warns_for_empty_values_without_type_error(self):
        errors, warnings = self.content_errors(
            [{"value": "1"}, {"value": ""}, {"value": "   "}], self.variable("numeric")
        )

        self.assertEqual(errors, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("2/3 empty values", warnings[0]["message"])

    def test_validates_string_coded_factor_values_and_duplicates(self):
        valid = self.variable(
            "factor",
            [{"value": "0", "label": "No"}, {"value": "1", "label": "Yes"}],
        )
        errors, _ = self.content_errors([{"value": "0"}, {"value": "2"}], valid)
        self.assertEqual(len(errors), 1)
        self.assertIn("invalid non-empty values", errors[0]["message"])

        duplicate = self.variable(
            "factor",
            [{"value": "1", "label": "Yes"}, {"value": "1", "label": "Also yes"}],
        )
        duplicate_errors, _ = self.content_errors([{"value": "1"}], duplicate)
        self.assertEqual(len(duplicate_errors), 1)
        self.assertIn("duplicate", duplicate_errors[0]["message"])


if __name__ == "__main__":
    unittest.main()
