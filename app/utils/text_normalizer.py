"""
Text Normalization utilities.

This module provides a class to normalize raw text from the STT service,
making it more suitable for NLP and TTS processing. It handles things like
number-to-word conversion, currency expansion, and other custom rules defined
in a YAML file.
"""
import yaml
import structlog
import re
import os
from pathlib import Path

log = structlog.get_logger()

class TextNormalizer:
    """
    Normalizes text using a set of rules loaded from a YAML file.
    """
    def __init__(self, rules_path: str = "config/normalization_rules.yaml"):
        """
        Initializes the normalizer by loading rules.
        Args:
            rules_path: Path to the normalization rules YAML file.
        """
        self.rules = {}
        try:
            if os.path.exists(rules_path):
                with open(rules_path, 'r', encoding='utf-8') as f:
                    self.rules = yaml.safe_load(f) or {}
                log.info("TextNormalizer: Loaded rules", count=len(self.rules.get('replacements', [])), path=rules_path)
            else:
                log.warning("TextNormalizer: Rules file not found.", path=rules_path)
        except Exception as e:
            log.error("TextNormalizer: Error loading rules", path=rules_path, exc_info=e)

    def _number_to_vietnamese_words(self, number_str: str) -> str:
        """Converts a number string to Vietnamese words."""
        try:
            num = int(number_str)
        except (ValueError, TypeError):
            return number_str

        if num == 0: return "không"
        if num < 0: return f"âm {self._number_to_vietnamese_words(str(abs(num)))}"

        parts = []
        billions = num // 1_000_000_000
        millions = (num % 1_000_000_000) // 1_000_000
        thousands = (num % 1_000_000) // 1_000
        remainder = num % 1_000

        if billions > 0:
            parts.append(f"{self._number_to_vietnamese_words(str(billions))} tỷ")
        if millions > 0:
            parts.append(f"{self._number_to_vietnamese_words(str(millions))} triệu")
        if thousands > 0:
            parts.append(f"{self._number_to_vietnamese_words(str(thousands))} nghìn")
        
        if remainder > 0:
            rem_words = []
            hundreds = remainder // 100
            tens_units = remainder % 100
            rem_words.append(f"{self._digit_to_word(hundreds)} trăm")

            if tens_units > 0:
                if hundreds > 0 and tens_units < 10:
                    rem_words.append("linh")
                
                tens = tens_units // 10
                units = tens_units % 10

                if tens > 1:
                    rem_words.append(f"{self._digit_to_word(tens)} mươi")
                elif tens == 1:
                    rem_words.append("mười")
                
                if units > 0:
                    if tens > 1 and units == 1:
                        rem_words.append("mốt")
                    elif tens > 0 and units == 5:
                        rem_words.append("lăm")
                    else:
                        rem_words.append(self._digit_to_word(units))
            parts.append(" ".join(filter(None, rem_words)))

        return " ".join(filter(None, parts))

    def _digit_to_word(self, digit: int) -> str:
        return ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"][digit]

    def normalize(self, text: str) -> str:
        """Applies all configured normalization rules to the input text."""
        if not self.rules and not text:
            return text

        normalized_text = f' {text.lower()} '

        # Apply simple replacements from YAML
        for rule in self.rules.get('replacements', []):
            for key, value in rule.items():
                normalized_text = normalized_text.replace(f' {key} ', f' {value} ')

        # Normalize currency
        def currency_replacer(match):
            num = match.group(1)
            unit = match.group(2).lower()
            num_words = self._number_to_vietnamese_words(num)
            if unit == 'k': return f'{num_words} nghìn'
            if unit == 'đ': return f'{num_words} đồng'
            return match.group(0)
        
        normalized_text = re.sub(r'\b(\d+)([kđ])\b', currency_replacer, normalized_text)

        # Normalize numbers
        normalized_text = re.sub(r'\b(\d+)\b', lambda m: self._number_to_vietnamese_words(m.group(1)), normalized_text)

        return normalized_text.strip()
