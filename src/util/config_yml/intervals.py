import re
from datetime import timedelta

# Kept in sync with the `interval` and `freq_max` patterns in .config.schema.yaml.
INTERVAL_PATTERN = r"^[0-9]+[smhdw]$"

interval_units = {
    "s": "seconds",
    "m": "minutes",
    "h": "hours",
    "d": "days",
    "w": "weeks",
}


def parse_interval(interval_str: str) -> timedelta:
    """Parse an interval such as "3h" into a timedelta.

    Raises ValueError on anything malformed. It used to return timedelta(0),
    which silently disabled rate limiting: a zero-length window means every
    queued timestamp is already outside it, so the queue drained on every call
    and no user was ever limited.

    Callers reach this only through fields that carry INTERVAL_PATTERN, so a bad
    value is rejected when config.yml is loaded rather than here.
    """
    re_match = re.fullmatch(r"([0-9]+)([smhdw])", interval_str)
    if not re_match:
        raise ValueError(
            f"malformed interval {interval_str!r}: expected a number followed by "
            "one of s/m/h/d/w, e.g. '30s', '3h', '7d'"
        )
    value = int(re_match.group(1))
    unit = interval_units[re_match.group(2)]
    return timedelta(**{unit: value})
