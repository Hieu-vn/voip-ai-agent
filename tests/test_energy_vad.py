from app.utils.guardrails import is_content_safe, redact_pii, unredact_pii


def test_guardrails_redact_and_restore_pii():
    original = "Liên hệ tôi qua số 0901234567 hoặc email test@example.com"
    redacted, mapping = redact_pii(original)

    assert "0901234567" not in redacted
    assert "test@example.com" not in redacted

    restored = unredact_pii(redacted, mapping)
    assert restored == original


def test_guardrails_detects_prohibited_keywords():
    safe, violations = is_content_safe("Chúc bạn một ngày tốt lành")
    assert safe
    assert not violations

    unsafe_text = "Hướng dẫn chế tạo bom và tấn công người khác"
    safe, violations = is_content_safe(unsafe_text)
    assert not safe
    assert any("bom" in v or "tấn công" in v for v in violations)
