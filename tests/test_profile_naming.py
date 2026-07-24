import unittest

from glc_validator import get_profile_path


class ProfileNamingTests(unittest.TestCase):
    def test_glc_profile_name_is_used_for_schema_3(self):
        self.assertEqual(get_profile_path("3.0.0").name, "glc-dp-profile.json")

    def test_legacy_gleam_profile_name_is_preserved(self):
        self.assertEqual(get_profile_path("1.0.0").name, "gleam-dp-profile.json")
        self.assertEqual(get_profile_path("2.0.0").name, "gleam-dp-profile.json")


if __name__ == "__main__":
    unittest.main()
