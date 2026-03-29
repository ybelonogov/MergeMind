"""Context processing tests."""

import unittest

from src.context.processing import enrich_example, parse_diff
from src.data.schema import MRExample


class ContextProcessingTests(unittest.TestCase):
    def test_parse_diff_extracts_changed_file_and_identifiers(self) -> None:
        diff = (
            "diff --git a/src/example.py b/src/example.py\n"
            "--- a/src/example.py\n"
            "+++ b/src/example.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-    return default_value\n"
            "+    return line_items[0].sku\n"
        )
        repository_files = {
            "src/example.py": (
                "def first_sku(line_items):\n"
                "    if not line_items:\n"
                "        return \"\"\n"
                "    return line_items[0].sku\n"
            )
        }

        changed_files = parse_diff(diff, repository_files)

        self.assertEqual(len(changed_files), 1)
        self.assertEqual(changed_files[0].path, "src/example.py")
        self.assertIn("line_items", changed_files[0].changed_identifiers)
        self.assertIn("first_sku", changed_files[0].structural_symbols)

    def test_enrich_example_builds_repository_context(self) -> None:
        example = MRExample(
            source_dataset="CodeReviewer",
            example_id="demo",
            split="demo",
            repo="repo/demo",
            title="Demo",
            description="Demo change",
            diff=(
                "diff --git a/src/example.py b/src/example.py\n"
                "--- a/src/example.py\n"
                "+++ b/src/example.py\n"
                "@@ -1,3 +1,3 @@\n"
                "-    return 0\n"
                "+    return discount / total\n"
            ),
            repository_files={
                "src/example.py": (
                    "def compute_discount_ratio(total, discount):\n"
                    "    if total is None:\n"
                    "        return 0\n"
                    "    return discount / total\n"
                )
            },
        )

        enriched = enrich_example(example)

        self.assertTrue(enriched.changed_files)
        self.assertIn("File: src/example.py", enriched.repository_context)
        self.assertIn("compute_discount_ratio", enriched.repository_context)


if __name__ == "__main__":
    unittest.main()
