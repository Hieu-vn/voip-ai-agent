import asyncio
import structlog
import re

log = structlog.get_logger()

# Basic sanitization regex for alphanumeric and hyphens
ALPHANUMERIC_HYPHEN_REGEX = re.compile(r"^[a-zA-Z0-9-]*$")
NUMERIC_REGEX = re.compile(r"^[0-9]*$")

def _sanitize_input(input_str: str, pattern: re.Pattern, max_len: int = 50) -> str:
    if not isinstance(input_str, str):
        return ""
    cleaned = input_str.strip()[:max_len]
    if not pattern.fullmatch(cleaned):
        log.warning("Input sanitization failed: Invalid characters detected.", input=input_str, cleaned=cleaned)
        return ""
    return cleaned

async def crm_lookup(slots: dict) -> dict:
    log.info("CRM Lookup initiated", slots=slots)
    order_id = slots.get("order_id")
    phone = slots.get("phone")

    if order_id:
        sanitized_order_id = _sanitize_input(order_id, ALPHANUMERIC_HYPHEN_REGEX, 20)
        if not sanitized_order_id:
            return {"status": "error", "message": "Mã đơn hàng không hợp lệ.", "data": None}
        # Use sanitized_order_id for CRM lookup
        log.info("Performing CRM lookup with sanitized order_id", order_id=sanitized_order_id)
    
    if phone:
        sanitized_phone = _sanitize_input(phone, NUMERIC_REGEX, 15)
        if not sanitized_phone:
            return {"status": "error", "message": "Số điện thoại không hợp lệ.", "data": None}
        # Use sanitized_phone for CRM lookup
        log.info("Performing CRM lookup with sanitized phone", phone=sanitized_phone)

    await asyncio.sleep(0.1) # Simulate network latency
    result = {"status": "success", "data": {"customer_name": "Nguyen Van A", "order_status": "pending"}}
    log.info("CRM Lookup completed", result=result)
    return result

async def crm_update(slots: dict) -> dict:
    log.info("CRM Update initiated", slots=slots)
    # Example: sanitize a 'case_id' if it were present in slots
    case_id = slots.get("case_id")
    if case_id:
        sanitized_case_id = _sanitize_input(case_id, ALPHANUMERIC_HYPHEN_REGEX, 20)
        if not sanitized_case_id:
            return {"status": "error", "message": "Mã case không hợp lệ.", "data": None}
        log.info("Performing CRM update with sanitized case_id", case_id=sanitized_case_id)

    await asyncio.sleep(0.1) # Simulate network latency
    result = {"status": "success", "message": "Customer data updated"}
    log.info("CRM Update completed", result=result)
    return result