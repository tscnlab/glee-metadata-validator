import unittest

import petl

from gleam_validator import warn_unreferenced_participant_ids


class ParticipantDatasetCoverageTests(unittest.TestCase):
    def participant_table(self, *participant_ids):
        return petl.fromdicts(
            {"participant_internal_id": participant_id}
            for participant_id in participant_ids
        )

    def dataset(self, participant_id):
        return {
            "dataset_crossref": {
                "dataset_crossref_participant_id": participant_id,
            }
        }

    def test_warns_for_declared_participants_without_datasets(self):
        warnings = warn_unreferenced_participant_ids(
            [self.dataset("P001")],
            self.participant_table("P001", "P002", "P003"),
        )

        self.assertEqual(len(warnings), 1)
        self.assertIn("2 declared participant IDs", warnings[0]["message"])
        self.assertIn("'P002', 'P003'", warnings[0]["message"])

    def test_returns_no_warning_when_every_participant_is_referenced(self):
        warnings = warn_unreferenced_participant_ids(
            [self.dataset("P001"), self.dataset("P002")],
            self.participant_table("P001", "P002"),
        )

        self.assertEqual(warnings, [])

    def test_accepts_multi_participant_references(self):
        warnings = warn_unreferenced_participant_ids(
            [self.dataset(["P001", "P002"])],
            self.participant_table("P001", "P002"),
        )

        self.assertEqual(warnings, [])


if __name__ == "__main__":
    unittest.main()
