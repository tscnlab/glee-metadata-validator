import os
import unittest

from gleam_validator import validate_against_json_schema


SCHEMA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "schemas",
    "3.0.0",
    "device_datasheet.schema.json",
)


def light_datasheet():
    return {
        "schema_version": "3.0.0",
        "datasheet_id": "lumitech-lt100-v1.0",
        "datasheet_version": "1.0",
        "datasheet_manufacturer": "Lumitech",
        "datasheet_type": "Wearable light sensor",
        "datasheet_sensor_modality": ["light"],
        "datasheet_sensor_modality_other": None,
        "datasheet_model": "LT-100",
        "datasheet_calibration_interval": 365,
        "datasheet_calibration_method": "Reference lamp comparison",
        "datasheet_calibration_accuracy": "+/-2%",
        "datasheet_calibration_range": "0-100,000 lux",
        "datasheet_calibration_notes": None,
        "datasheet_calibration_parameters": None,
        "datasheet_calibration_spectral_sensitivity": [
            {
                "datasheet_calibration_spectral_sensitivity_wavelength": 555,
                "datasheet_calibration_spectral_sensitivity_relative": 1.0,
            }
        ],
        "datasheet_calibration_linearity": "+/-2% over the measurement range",
        "datasheet_calibration_directional_response": "Cosine corrected",
        "datasheet_channel": [
            {
                "datasheet_channel_nr": 1,
                "datasheet_channel_name": "photopic",
                "datasheet_channel_unit": "lux",
                "datasheet_channel_description": "Photopic illuminance",
            }
        ],
    }


class Schema3DeviceDatasheetTests(unittest.TestCase):
    def errors(self, datasheet):
        return validate_against_json_schema([datasheet], SCHEMA_PATH, "device_datasheets")

    def test_accepts_complete_light_datasheet(self):
        self.assertEqual(self.errors(light_datasheet()), [])

    def test_requires_light_specific_calibration_fields(self):
        datasheet = light_datasheet()
        del datasheet["datasheet_calibration_spectral_sensitivity"]
        del datasheet["datasheet_calibration_range"]

        errors = self.errors(datasheet)

        self.assertTrue(any("datasheet_calibration_spectral_sensitivity" in error["message"] for error in errors))
        self.assertTrue(any("datasheet_calibration_range" in error["message"] for error in errors))

    def test_accepts_non_light_datasheet_with_general_parameters(self):
        datasheet = light_datasheet()
        datasheet["datasheet_id"] = "example-accelerometer-v1.0"
        datasheet["datasheet_type"] = "Tri-axial accelerometer"
        datasheet["datasheet_sensor_modality"] = ["accelerometer"]
        datasheet["datasheet_calibration_interval"] = None
        datasheet["datasheet_calibration_parameters"] = [
            {
                "parameter_name": "measurement_range",
                "parameter_value": 8,
                "parameter_unit": "g",
                "parameter_description": "+/-8 g range",
            }
        ]
        for field in [
            "datasheet_calibration_spectral_sensitivity",
            "datasheet_calibration_linearity",
            "datasheet_calibration_directional_response",
            "datasheet_calibration_range",
        ]:
            del datasheet[field]

        self.assertEqual(self.errors(datasheet), [])

    def test_rejects_light_only_fields_for_non_light_modality(self):
        datasheet = light_datasheet()
        datasheet["datasheet_sensor_modality"] = ["accelerometer"]

        errors = self.errors(datasheet)

        self.assertTrue(any("should not be valid" in error["message"] for error in errors))

    def test_requires_description_for_other_modality(self):
        datasheet = light_datasheet()
        datasheet["datasheet_sensor_modality"] = ["other"]
        del datasheet["datasheet_sensor_modality_other"]
        for field in [
            "datasheet_calibration_spectral_sensitivity",
            "datasheet_calibration_linearity",
            "datasheet_calibration_directional_response",
            "datasheet_calibration_range",
        ]:
            del datasheet[field]

        errors = self.errors(datasheet)

        self.assertTrue(any("datasheet_sensor_modality_other" in error["message"] for error in errors))

    def test_rejects_null_description_for_other_modality(self):
        datasheet = light_datasheet()
        datasheet["datasheet_sensor_modality"] = ["other"]
        for field in [
            "datasheet_calibration_spectral_sensitivity",
            "datasheet_calibration_linearity",
            "datasheet_calibration_directional_response",
            "datasheet_calibration_range",
        ]:
            del datasheet[field]

        self.assertTrue(self.errors(datasheet))

    def test_accepts_mixed_light_and_non_light_modalities(self):
        datasheet = light_datasheet()
        datasheet["datasheet_sensor_modality"] = ["light", "accelerometer"]
        datasheet["datasheet_calibration_parameters"] = [
            {
                "parameter_name": "measurement_range",
                "parameter_value": 8,
                "parameter_unit": "g",
            }
        ]

        self.assertEqual(self.errors(datasheet), [])

    def test_requires_light_fields_when_light_is_one_of_multiple_modalities(self):
        datasheet = light_datasheet()
        datasheet["datasheet_sensor_modality"] = ["temperature", "light"]
        del datasheet["datasheet_calibration_linearity"]

        errors = self.errors(datasheet)

        self.assertTrue(any("datasheet_calibration_linearity" in error["message"] for error in errors))

    def test_rejects_duplicate_or_empty_modalities(self):
        duplicate = light_datasheet()
        duplicate["datasheet_sensor_modality"] = ["light", "light"]
        empty = light_datasheet()
        empty["datasheet_sensor_modality"] = []

        self.assertTrue(self.errors(duplicate))
        self.assertTrue(self.errors(empty))

    def test_rejects_data_constructs_as_sensor_modalities(self):
        for invalid_modality in ["activity", "sleep", "questionnaire"]:
            with self.subTest(modality=invalid_modality):
                datasheet = light_datasheet()
                datasheet["datasheet_sensor_modality"] = [invalid_modality]
                self.assertTrue(self.errors(datasheet))


if __name__ == "__main__":
    unittest.main()
