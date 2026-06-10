import re
from fractions import Fraction

_UNANSWERABLE_PATTERNS = re.compile(
    r"cannot be determined|not enough information|insufficient information|"
    r"cannot be solved|no solution|not possible to determine|"
    r"cannot be calculated|information (is )?not (given|provided|sufficient)|"
    r"cannot determine",
    re.IGNORECASE,
)


def is_unanswerable_response(content: str) -> bool:
    """Return True if the model explicitly states the problem cannot be solved."""
    return bool(_UNANSWERABLE_PATTERNS.search(content))


def _clean(s: str) -> str:
    """Remove LaTeX spacing artifacts and normalize whitespace around commas."""
    s = re.sub(r'\\ ', '', s)                    # \ space (LaTeX medium/thin space)
    s = re.sub(r'\\,', '', s)                    # \, (LaTeX thin space)
    s = re.sub(r'\s*,\s*', ',', s)               # spaces around commas
    s = re.sub(r'\^?\{?\\circ\}?|°', '', s)      # degree symbol: °, \circ, ^{\circ}
    s = re.sub(r'\\(?:left|right)\s*', '', s)    # \left and \right sizing commands
    return s.strip()


def _strip_assignment(s: str) -> str:
    """Strip leading variable assignment like 'x = ' or 'x='."""
    return re.sub(r'^[a-zA-Z]\s*=\s*', '', s).strip()


def _set_match(predicted: str, expected: str) -> bool:
    """Order-invariant comparison of comma-separated multi-value answers."""
    pred_parts = [p.strip() for p in predicted.split(',') if p.strip()]
    exp_parts = [p.strip() for p in expected.split(',') if p.strip()]
    if len(pred_parts) != len(exp_parts) or len(pred_parts) < 2:
        return False

    def norm(p):
        v = _normalize(p)
        return v if v is not None else p.strip()

    return sorted(norm(p) for p in pred_parts) == sorted(norm(p) for p in exp_parts)


def _normalize(answer: str) -> str | None:
    """Try to reduce an answer string to a canonical numeric form."""
    s = answer.strip().rstrip(".").strip("$").strip()
    # Handle LaTeX fractions: \frac or \dfrac
    frac_m = re.match(r"(-?)\\d?frac\{([^}]+)\}\{([^}]+)\}", s)
    if frac_m:
        try:
            sign = -1 if frac_m.group(1) == "-" else 1
            num = float(Fraction(frac_m.group(2).strip()))
            den = float(Fraction(frac_m.group(3).strip()))
            if den != 0:
                return str(sign * num / den)
        except (ValueError, ZeroDivisionError):
            pass
    # Strip LaTeX commands like \text{...}, \dfrac{...}{...}, etc.
    s = re.sub(r"\\{1,2}[a-zA-Z]+\{[^}]*\}?", "", s)
    # Strip lone backslash sequences (e.g. \\ spacing artifacts)
    s = re.sub(r"\\+\s*", " ", s).strip().rstrip(".")
    # Remove currency/percentage/LaTeX math delimiters
    if s.endswith("%"):
        s = s[:-1].strip()
    s = s.strip("$").strip()
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


def _latex_to_sympy_str(s: str) -> str:
    """Convert common LaTeX math to a sympy-parseable string."""
    s = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', s)
    s = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'((\1)/(\2))', s)
    s = re.sub(r'\^\{([^}]+)\}', r'**(\1)', s)
    s = re.sub(r'\{([^}]+)\}', r'(\1)', s)
    s = re.sub(r'\\cdot|\\times', '*', s)
    s = re.sub(r'\\[a-zA-Z]+', '', s)
    # Insert * for implicit multiplication: 17sqrt → 17*sqrt, ) ( → )*(
    s = re.sub(r'(\d)\s*(sqrt)', r'\1*\2', s)
    s = re.sub(r'(\d|\))\s+\(', r'\1*(', s)
    return s.strip()


def _symbolic_match(predicted: str, expected: str) -> bool:
    """Use sympy to check symbolic equivalence between two math expressions."""
    try:
        import sympy
        from sympy import simplify
        from sympy.parsing.latex import parse_latex

        def _parse(s: str):
            s = s.strip()
            try:
                return parse_latex(s)
            except Exception:
                pass
            try:
                return sympy.sympify(s, evaluate=True)
            except Exception:
                pass
            try:
                return sympy.sympify(_latex_to_sympy_str(s), evaluate=True)
            except Exception:
                pass
            return None

        pred_expr = _parse(predicted)
        exp_expr = _parse(expected)
        if pred_expr is None or exp_expr is None:
            return False
        if simplify(pred_expr - exp_expr) == 0:
            return True
        # Numeric fallback: compare floating-point values
        try:
            pred_val = complex(pred_expr.evalf())
            exp_val = complex(exp_expr.evalf())
            if abs(pred_val.imag) < 1e-6 and abs(exp_val.imag) < 1e-6:
                tol = max(1e-4, 1e-4 * abs(exp_val.real))
                return abs(pred_val.real - exp_val.real) < tol
        except Exception:
            pass
        return False
    except Exception:
        return False


def answers_match(predicted: str | None, expected: str | None, symbolic: bool = False) -> bool:
    """Return True if predicted and expected represent the same answer.

    When symbolic=True (e.g. for competition math), falls back to sympy-based
    equivalence check after numeric normalization fails. Also handles
    semicolon-separated multi-answer expected strings (any match → True).
    """
    if predicted is None or expected is None:
        return False

    # Handle semicolon-separated multi-answer expected (e.g. OlympiadBench)
    variants = [v.strip() for v in expected.split(";")]

    pred_clean = _clean(_strip_assignment(predicted))

    for exp in variants:
        exp_clean = _clean(exp)

        if predicted.strip() == exp or pred_clean == exp_clean:
            return True
        norm_pred = _normalize(pred_clean)
        norm_exp = _normalize(exp_clean)
        if norm_pred is not None and norm_exp is not None and norm_pred == norm_exp:
            return True
        if symbolic and _symbolic_match(pred_clean, exp_clean):
            return True
        if _set_match(pred_clean, exp_clean):
            return True

    return False
