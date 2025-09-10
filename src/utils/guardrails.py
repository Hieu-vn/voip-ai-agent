import re
import logging
from typing import Tuple, Dict

# Các biểu thức chính quy (regex) để phát hiện PII
# Đây là các mẫu cơ bản, có thể cần cải tiến để chính xác hơn
PHONE_REGEX = re.compile(r'(\b(0?)(3[2-9]|5[6|8|9]|7[0|6-9]|8[1-6|8-9]|9[0-4|6-9])[0-9]{7}\b)')
EMAIL_REGEX = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')

def redact_pii(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Tìm và che PII trong văn bản.
    Trả về văn bản đã che và một dictionary để ánh xạ ngược.
    """
    pii_map = {}
    
    def phone_replacer(match):
        phone_number = match.group(0)
        placeholder = f"[PHONE_{len(pii_map) + 1}]"
        pii_map[placeholder] = phone_number
        logging.debug(f"Guardrails: Đã che số điện thoại '{phone_number}' bằng '{placeholder}'")
        return placeholder

    def email_replacer(match):
        email = match.group(0)
        placeholder = f"[EMAIL_{len(pii_map) + 1}]"
        pii_map[placeholder] = email
        logging.debug(f"Guardrails: Đã che email '{email}' bằng '{placeholder}'")
        return placeholder

    redacted_text = PHONE_REGEX.sub(phone_replacer, text)
    redacted_text = EMAIL_REGEX.sub(email_replacer, redacted_text)
    
    return redacted_text, pii_map

def unredact_pii(text: str, pii_map: Dict[str, str]) -> str:
    """
    Khôi phục PII trong văn bản từ dictionary ánh xạ.
    """
    for placeholder, original_value in pii_map.items():
        text = text.replace(placeholder, original_value)
    return text
