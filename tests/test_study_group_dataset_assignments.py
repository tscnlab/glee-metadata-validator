import unittest

from glc_validator import validate_study_group_datasets_link


class StudyGroupDatasetAssignmentTests(unittest.TestCase):
    def test_rejects_dataset_assigned_to_multiple_groups(self):
        study = {
            "study_internal_id": "STUDY001",
            "study_groups": [
                {
                    "study_group_name": "control",
                    "study_group_datasets": ["DS001"],
                },
                {
                    "study_group_name": "intervention",
                    "study_group_datasets": ["DS001"],
                },
            ],
        }

        errors = validate_study_group_datasets_link(study, {"DS001"})

        self.assertEqual(len(errors), 1)
        self.assertIn("assigned to multiple study groups", errors[0]["message"])
        self.assertIn("'control' and 'intervention'", errors[0]["message"])

    def test_accepts_distinct_dataset_assignments(self):
        study = {
            "study_internal_id": "STUDY001",
            "study_groups": [
                {
                    "study_group_name": "control",
                    "study_group_datasets": ["DS001"],
                },
                {
                    "study_group_name": "intervention",
                    "study_group_datasets": ["DS002"],
                },
            ],
        }

        errors = validate_study_group_datasets_link(study, {"DS001", "DS002"})

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
