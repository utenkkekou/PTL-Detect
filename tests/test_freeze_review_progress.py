import unittest

from scripts.freeze_review_progress import ATTESTATION, freeze


class FreezeReviewProgressTests(unittest.TestCase):
    def _manifest(self):
        return [
            {
                "id": "a",
                "requires_change": True,
                "boxes": [
                    {"cls": 3, "cx": 0.2, "cy": 0.2, "w": 0.1, "h": 0.1, "conf": 0.5}
                ],
            }
        ]

    def test_strips_untrusted_box_metadata_and_recomputes_changed(self):
        progress = {
            "a": {
                "status": "reviewed",
                "boxes": [
                    {
                        "cls": 0,
                        "cx": 0.5,
                        "cy": 0.5,
                        "w": 0.1,
                        "h": 0.2,
                        "conf": 0.9,
                        "candidate_sources": ["seed"],
                        "untrusted_seed": True,
                    }
                ],
            }
        }
        frozen, report = freeze(
            manifest=self._manifest(),
            progress=progress,
            reviewer="manual+qa",
            attestation=ATTESTATION,
        )
        self.assertEqual(
            set(frozen["a"]["boxes"][0]), {"cls", "cx", "cy", "w", "h"}
        )
        self.assertTrue(frozen["a"]["changed"])
        self.assertEqual(report["class_counts"]["pedestrian_signal_red"], 1)

    def test_rejects_requires_change_seed_that_was_not_changed(self):
        progress = {
            "a": {
                "status": "reviewed",
                "boxes": self._manifest()[0]["boxes"],
            }
        }
        with self.assertRaisesRegex(ValueError, "requires_change"):
            freeze(
                manifest=self._manifest(),
                progress=progress,
                reviewer="manual+qa",
                attestation=ATTESTATION,
            )

    def test_rejects_strong_overlap_unless_explicitly_allowed(self):
        manifest = [{"id": "a", "boxes": []}]
        progress = {
            "a": {
                "status": "reviewed",
                "boxes": [
                    {"cls": 0, "cx": 0.5, "cy": 0.5, "w": 0.2, "h": 0.2},
                    {"cls": 3, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.1},
                ],
            }
        }
        with self.assertRaisesRegex(ValueError, "strong overlaps"):
            freeze(
                manifest=manifest,
                progress=progress,
                reviewer="manual+qa",
                attestation=ATTESTATION,
            )
        _, report = freeze(
            manifest=manifest,
            progress=progress,
            reviewer="manual+qa",
            attestation=ATTESTATION,
            allowed_overlap_ids={"a"},
        )
        self.assertEqual(len(report["strong_overlap_warnings"]), 1)

    def test_rejects_coercible_non_integer_class(self):
        progress = {
            "a": {
                "status": "reviewed",
                "boxes": [
                    {"cls": 0.9, "cx": 0.5, "cy": 0.5, "w": 0.1, "h": 0.2}
                ],
            }
        }
        with self.assertRaisesRegex(ValueError, "invalid box"):
            freeze(
                manifest=self._manifest(),
                progress=progress,
                reviewer="manual+qa",
                attestation=ATTESTATION,
            )


if __name__ == "__main__":
    unittest.main()
