
import unittest
import yaml
from pathlib import Path
import sys
import os

# Add the script's directory to the Python path to allow importing
sys.path.append(str(Path(__file__).parent.parent.parent / 'scripts' / 'preparation'))

from create_manifest import TextNormalizer

class TestTextNormalizer(unittest.TestCase):

    def setUp(self):
        """Create a temporary normalization rules file for testing."""
        self.rules_content = {
            'digit_map': {
                '0': 'không', '1': 'một', '2': 'hai',
                '3': 'ba', '4': 'bốn', '5': 'năm',
                '6': 'sáu', '7': 'bảy', '8': 'tám', '9': 'chín'
            },
            'regex_rules': [
                {'pattern': '[?!.]+', 'replace': '.', 'description': 'Collapse punctuation.'},
                {'pattern': '[^\\w\\sÀ-ỹà-ỹ.]', 'replace': '', 'description': 'Remove invalid chars.'},
                {'pattern': '\\s+', 'replace': ' ', 'description': 'Collapse whitespace.'}
            ],
            'max_length': 25
        }
        self.rules_path = Path("test_rules.yaml")
        with self.rules_path.open("w", encoding="utf-8") as f:
            yaml.dump(self.rules_content, f)
        
        self.normalizer = TextNormalizer(self.rules_path)

    def tearDown(self):
        """Remove the temporary rules file."""
        if self.rules_path.exists():
            self.rules_path.unlink()

    def test_digit_to_word_conversion(self):
        """Tests if digits are correctly converted to words."""
        self.assertEqual(self.normalizer.normalize("mã 1900"), "mã một chín không không")
        self.assertEqual(self.normalizer.normalize("năm 2025"), "năm hai không hai năm")

    def test_regex_rules_application(self):
        """Tests if regex rules are applied correctly."""
        self.assertEqual(self.normalizer.normalize("Xin chào!!!"), "Xin chào.")
        self.assertEqual(self.normalizer.normalize("  cảm   ơn  "), "cảm ơn")
        self.assertEqual(self.normalizer.normalize("đây là@#$test"), "đây làtest")

    def test_max_length_enforcement(self):
        """Tests if text is correctly truncated to max_length."""
        long_string = "đây là một câu rất dài và không có ý nghĩa gì cả"
        self.assertEqual(self.normalizer.normalize(long_string), "đây là một câu rất dài v") # Truncated to 25

    def test_combined_normalization(self):
        """Tests a combination of all normalization rules."""
        raw_text = "  Mã của bạn là: 1234??  "
        expected = "Mã của bạn là. một hai ba bốn."
        self.assertEqual(self.normalizer.normalize(raw_text), expected)

    def test_empty_and_none_input(self):
        """Tests handling of empty or None input."""
        self.assertEqual(self.normalizer.normalize(""), "")
        self.assertEqual(self.normalizer.normalize(None), "")

if __name__ == '__main__':
    unittest.main()
