import re
from datetime import timedelta

interval_units = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}


def parse_interval(interval_str: str) -> timedelta:
    re_match = re.fullmatch(r"([0-9]+)([smhdw])", interval_str)
    if not re_match:
        return timedelta(0)
    value = int(re_match.group(1))
    unit = interval_units[re_match.group(2)]
    return timedelta(**{unit: value})
