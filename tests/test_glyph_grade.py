"""v0.4 §26: grading a forecast with a measurement."""
import json

import pytest

from umwelt.glyph import Grade, ReceiptError, grade, parse


def test_forecast_confirmed():
    g = grade("[0.8~]☀", "[0.83]☀", tol=0.05)
    assert g.sentence == "🧪 ⟨[0.83]☀|[0.8~]☀⟩ → |✅⟩" and g.verdict == "✅"
    assert g.max_abs_err == pytest.approx(0.03)
    assert parse("[0.83]☀") is not None


def test_forecast_refuted():
    g = grade("[0.8~]☀", "[0.5]☀", tol=0.05)
    assert g.sentence == "🧪 ⟨[0.5]☀|[0.8~]☀⟩ → |❌⟩" and g.verdict == "❌"


def test_tolerance_is_inclusive_and_default_is_tight():
    assert grade("☀!0.25~", "☀!0.30", tol=0.05 + 1e-12).verdict == "✅"
    assert grade("☀!0.25~", "☀!0.30", tol=0.04).verdict == "❌"
    assert grade("[0.8~]☀", "[0.8]☀").verdict == "✅"
    assert grade("[0.8~]☀", "[0.8001]☀").verdict == "❌"


def test_every_slot_must_be_within_tolerance_mixed_vector():
    f = "[0.9,0.3~,1.2~]☀🌙"
    assert grade(f, "[0.9,0.31,1.19]☀🌙", tol=0.02).verdict == "✅"
    assert grade(f, "[0.9,0.31,1.5]☀🌙", tol=0.02).verdict == "❌"
    assert grade(f, "[0.8,0.3,1.2]☀🌙", tol=0.02).verdict == "❌"      # the straight slot is a claim too


def test_dark_measurement_grades_nothing():
    g = grade("[0.9,0.3,1.2~]☀🌙", "[0.9,0.3,·]☀🌙")
    assert g.verdict is None and g.sentence.endswith("→ ✅?")


def test_record_is_tape_ready_and_appended():
    tape = []
    g = grade("[0.8~]☀", "[0.83]☀", tol=0.05, tape=tape)
    assert tape == [g.to_record()] and json.loads(json.dumps(tape[0])) == tape[0]
    assert tape[0]["kind"] == "grade" and tape[0]["verdict"] == "✅" and tape[0]["tol"] == 0.05
    assert tape[0]["forecast"] == "[0.8~]☀" and tape[0]["measured"] == "[0.83]☀"
    assert isinstance(g, Grade) and g.to_record()["sentence"] == g.sentence


def test_refusals():
    with pytest.raises(ReceiptError, match="nothing to grade"):
        grade("[0.8]☀", "[0.8]☀")                                   # a straight number is not a forecast
    with pytest.raises(ReceiptError, match="itself a forecast"):
        grade("[0.8~]☀", "[0.8~]☀")
    with pytest.raises(ReceiptError, match="not the same statement"):
        grade("[0.8~]☀", "[0.8]🌙")
    with pytest.raises(ReceiptError, match="not the same statement"):
        grade("[0.8~]☀", "☀!0.8")
    with pytest.raises(ReceiptError):
        grade("[0.8~]☀", "[0.8]☀", tol=-1)
