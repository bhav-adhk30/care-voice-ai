"""The bedside emergency button: P1 always, and logged like any other call."""

from src.audit import AuditLog
from src.pipeline import emergency_button_result
from src.rules import CONFIG, EMERGENCY_BUTTON, emergency_button


def test_button_is_p1_with_reason():
    d = emergency_button()
    assert d.final_priority == CONFIG["P_EMERGENCY_BUTTON"] == 1
    assert d.rules_fired == ["R9_emergency_button"] and "emergency button" in d.reason


def test_button_call_is_logged(tmp_path):
    r = emergency_button_result("12")
    assert r["request_class"] == EMERGENCY_BUTTON and r["final_priority"] == 1
    log = AuditLog(tmp_path / "a.db")
    i = log.append(r, bed="12", audio_name="emergency_button", audio_bytes=b"", mode="button")
    row = log.recent(1)[0]
    assert row["id"] == i and row["final_priority"] == 1 and row["mode"] == "button"
