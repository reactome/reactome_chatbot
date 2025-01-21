import re
from datetime import datetime, timedelta
from enum import StrEnum, auto
from fnmatch import fnmatch

from pydantic import BaseModel

from util.logging import logging

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


class TriggerEvent(StrEnum):
    on_chat_start = auto()
    on_chat_end = auto()
    on_chat_resume = auto()
    on_message = auto()


class Trigger(BaseModel):
    event: TriggerEvent | None = None
    after_messages: int | None = None
    start: datetime | None = None
    end: datetime | None = None
    freq_max: str | None = None

    def match_trigger(
        self,
        event: TriggerEvent | None = None,
        after_messages: int | None = None,
        last_message: str | None = None,
    ):
        now = datetime.now()
        if self.event and self.event != event:
            return False
        if self.after_messages and self.after_messages != after_messages:
            return False
        if self.start and self.start.replace(tzinfo=None) > now:
            return False
        if self.end and self.end.replace(tzinfo=None) < now:
            return False
        if (
            self.freq_max
            and last_message
            and (
                parse_interval(self.freq_max)
                > now - datetime.fromisoformat(last_message)
            )
        ):
            return False
        return True


class Message(BaseModel):
    message: str
    enabled: bool = True
    recipients: list[str] | None = None
    trigger: Trigger

    def match_recipient(self, user_id: str | None) -> bool:
        if not self.recipients:
            return True
        for entry in self.recipients:
            logging.warning(entry)
            if entry == "all":
                return True
            if user_id is None:
                if entry == "guests":
                    return True
            else:
                if entry == "logged_in":
                    return True
                elif entry[0] == "/" and entry[-1] == "/":
                    if re.search(entry[1:-1], user_id):
                        return True
                else:
                    if fnmatch(user_id, entry):
                        return True
        return False
