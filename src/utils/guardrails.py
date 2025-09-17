import re
import logging
from typing import Dict, List, Tuple

PHONE_REGEX = re.compile(r'(\b(0?)(3[2-9]|5[6|8|9]|7[0|6-9]|8[1-6|8-9]|9[0-4|6-9])[0-9]{7}\b)')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
PROHIBITED_KEYWORDS = {
    "bạo lực", "tự sát", "ma túy", "bom", "tấn công", "hack", "xâm nhập", "kill", "suicide"
}
JAILBREAK_PATTERNS = [
    re.compile(r"(?i)(ignore previous instructions)", re.IGNORECASE),
    re.compile(r"(?i)act as a malicious", re.IGNORECASE),
]


def redact_pii(text: str) -> Tuple[str, Dict[str, str]]:
    pii_map: Dict[str, str] = {}

    def phone_replacer(match):
        phone_number = match.group(0)
        placeholder = f"[PHONE_{len(pii_map) + 1}]"
        pii_map[placeholder] = phone_number
        logging.debug("Guardrails: masked phone '%s' as '%s'", phone_number, placeholder)
        return placeholder

    def email_replacer(match):
        email = match.group(0)
        placeholder = f"[EMAIL_{len(pii_map) + 1}]"
        pii_map[placeholder] = email
        logging.debug("Guardrails: masked email '%s' as '%s'", email, placeholder)
        return placeholder

    redacted_text = PHONE_REGEX.sub(phone_replacer, text)
    redacted_text = EMAIL_REGEX.sub(email_replacer, redacted_text)
    return redacted_text, pii_map


def unredact_pii(text: str, pii_map: Dict[str, str]) -> str:
    for placeholder, original_value in pii_map.items():
        text = text.replace(placeholder, original_value)
    return text


def detect_prohibited_keywords(text: str) -> List[str]:
    lower = text.lower()
    return [kw for kw in PROHIBITED_KEYWORDS if kw in lower]


def detect_jailbreak_patterns(text: str) -> List[str]:
    matches = []
    for pattern in JAILBREAK_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def sanitize_response(text: str) -> str:
    # hiện tại chỉ cắt bỏ khoảng trắng dư; có thể mở rộng nếu cần
    return text.strip()


def evaluate_output_guardrails(text: str) -> Dict[str, List[str]]:
    violations = {
        "prohibited_keywords": detect_prohibited_keywords(text),
        "jailbreak": detect_jailbreak_patterns(text),
    }
    return {k: v for k, v in violations.items() if v}


def is_response_safe(text: str) -> Tuple[bool, Dict[str, List[str]]]:
    violations = evaluate_output_guardrails(text)
    is_safe = not violations
    return is_safe, violations
