import os
import tempfile
import unittest

from gleam_validator import read_tabular_file


class TabularHeaderDetectionTests(unittest.TestCase):
    def make_file(self, text):
        temporary = tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8")
        temporary.write(text)
        temporary.close()
        self.addCleanup(lambda: os.path.exists(temporary.name) and os.unlink(temporary.name))
        return temporary.name

    def test_explicit_header_row_skips_preamble(self):
        path = self.make_file(
            "Device export\nGenerated 2026-07-16\ntimestamp,illuminance\n2026-07-16 08:00:00,10.5\n"
        )

        headers, rows, error = read_tabular_file(
            path,
            "csv",
            "utf-8",
            header_row_hint=3,
            declared_headers=["timestamp", "illuminance"],
        )

        self.assertIsNone(error)
        self.assertEqual(headers, ["timestamp", "illuminance"])
        self.assertEqual(rows[0]["illuminance"], "10.5")

    def test_declared_columns_detect_header_after_preamble(self):
        path = self.make_file(
            "Device,ABC123\nOperator,Researcher\ntimestamp,illuminance\n2026-07-16 08:00:00,10.5\n"
        )

        headers, rows, error = read_tabular_file(
            path,
            "csv",
            "utf-8",
            declared_headers=["timestamp", "illuminance"],
        )

        self.assertIsNone(error)
        self.assertEqual(headers, ["timestamp", "illuminance"])
        self.assertEqual(len(rows), 1)

    def test_out_of_range_header_row_is_reported(self):
        path = self.make_file("timestamp,illuminance\n2026-07-16 08:00:00,10.5\n")

        _, _, error = read_tabular_file(path, "csv", "utf-8", header_row_hint=5)

        self.assertIn("out of range", error)


if __name__ == "__main__":
    unittest.main()
