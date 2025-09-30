"""
Guardrails and content safety utilities for the AI agent.

This module provides functions to:
- Redact Personally Identifiable Information (PII) from user input before sending
  it to the LLM.
- Un-redact PII from the LLM's response.
- Detect prohibited keywords and jailbreak attempts in bot responses.

This logic is critical for maintaining a safe and secure user experience.
"""
import re
import structlog
from typing import Dict, List, Tuple

log = structlog.get_logger()

# --- PII Detection Patterns ---
PHONE_REGEX = re.compile(r'(\b(0?)(3[2-9]|5[6|8|9]|7[0|6-9]|8[1-6|8-9]|9[0-4|6-9])[0-9]{7}\b)')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

# --- Content Safety Patterns ---
PROHIBITED_KEYWORDS = {
    "bạo lực", "tự sát", "ma túy", "bom", "tấn công", "hack", "xâm nhập", "giết", "suicide"
}
JAILBREAK_PATTERNS = [
    re.compile(r"(?i)ignore previous instructions"),
    re.compile(r"(?i)act as a malicious"),
    re.compile(r"(?i)you are now in service mode"),
]


def redact_pii(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Finds and replaces PII (phone numbers, emails) in text with placeholders.

    Args:
        text: The input string.

    Returns:
        A tuple containing the redacted text and a map from placeholders to original PII.
    """
    pii_map: Dict[str, str] = {}

    def replacer(match, pii_type: str):
        pii_value = match.group(0)
        placeholder = f"[{pii_type}_{len(pii_map)}_]"
        pii_map[placeholder] = pii_value
        log.debug("Guardrails: Redacted PII", value=pii_value, placeholder=placeholder)
        return placeholder

    redacted_text = PHONE_REGEX.sub(lambda m: replacer(m, 'PHONE'), text)
    redacted_text = EMAIL_REGEX.sub(lambda m: replacer(m, 'EMAIL'), redacted_text)
    
    return redacted_text, pii_map


def unredact_pii(text: str, pii_map: Dict[str, str]) -> str:
    """
    Restores original PII in a text string using the provided map.

    Args:
        text: The text with PII placeholders.
        pii_map: The map from placeholders back to original PII.

    Returns:
        The text with PII restored.
    """
    for placeholder, original_value in pii_map.items():
        text = text.replace(placeholder, original_value)
    return text


def is_content_safe(text: str) -> Tuple[bool, List[str]]:
    """
    Checks if the given text violates any content safety policies.

    Args:
        text: The text to check (typically an LLM response).

    Returns:
        A tuple containing a boolean (True if safe) and a list of violation reasons.
    """
    violations = []
    lower_text = text.lower()

    # Check for prohibited keywords
    found_keywords = [kw for kw in PROHIBITED_KEYWORDS if kw in lower_text]
    if found_keywords:
        violations.append(f"prohibited_keywords_found: {found_keywords}")

    # Check for jailbreak patterns
    for pattern in JAILBREAK_PATTERNS:
        if pattern.search(text):
            violations.append(f"jailbreak_pattern_detected: '{pattern.pattern}'")

    is_safe = not violations
    if not is_safe:
        log.warning("Guardrails: Unsafe content detected", violations=violations, text=text)
        
    return is_safe, violations
