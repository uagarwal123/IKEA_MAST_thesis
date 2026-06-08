import re
from fractions import Fraction


def _normalize(answer: str) -> str | None:
    """Try to reduce an answer string to a canonical numeric form."""
    s = answer.strip().rstrip(".")
    # Remove percentage sign
    if s.endswith("%"):
        s = s[:-1].strip()
    # Try fraction (e.g. "1/2")
    try:
        return str(float(Fraction(s)))
    except (ValueError, ZeroDivisionError):
        pass
    # Try direct float
    try:
        return str(float(s))
    except ValueError:
        pass
    return None


def answers_match(predicted: str | None, expected: str | None) -> bool:
    """Return True if predicted and expected represent the same answer."""
    if predicted is None or expected is None:
        return False
    if predicted.strip() == expected.strip():
        return True
    norm_pred = _normalize(predicted)
    norm_exp = _normalize(expected)
    if norm_pred is not None and norm_exp is not None:
        return norm_pred == norm_exp
    return False
