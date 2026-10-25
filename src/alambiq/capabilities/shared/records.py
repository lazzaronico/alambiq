"""Small record helpers shared by more than one family.

The predecessor defined a canonical digest in 23 separate places. There is one
here, and it is the only one.
"""

from __future__ import annotations

import hashlib
import json
import re
from fractions import Fraction


def digest(data):
    return hashlib.sha256(
        json.dumps(
            data, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def label(value, name, maximum=200):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(ord(c) < 32 for c in value)
    ):
        raise ValueError(
            f"{name} requires a nonempty single-line string, at most {maximum} characters"
        )


def coefficient(value):
    if type(value) not in (int, str):
        raise ValueError(
            "Coefficient requires a positive integer or exact rational/decimal string"
        )
    if isinstance(value, str) and (
        len(value) > 32
        or not re.fullmatch(r"\+?(?:[0-9]+(?:/[0-9]+|\.[0-9]*)?|\.[0-9]+)", value)
    ):
        raise ValueError(
            "Coefficient strings require at most 32 characters in integer, fraction or decimal notation"
        )
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Coefficient requires a positive rational number") from exc
    if not 0 < result <= 10000 or result.denominator > 1000000:
        raise ValueError(
            "Coefficient must be in (0,10000], with denominator at most 1000000"
        )
    return result


def signed_coefficient(value):
    if type(value) is int:
        return (-1 if value < 0 else 1) * coefficient(abs(value))
    if isinstance(value, str):
        return (-1 if value.startswith("-") else 1) * coefficient(
            value.removeprefix("-")
        )
    raise ValueError(
        "Stoichiometry requires nonzero signed integers or rational strings"
    )
