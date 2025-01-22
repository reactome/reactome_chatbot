from datetime import datetime
from enum import StrEnum, auto

from pydantic import BaseModel

from util.config_yml.intervals import parse_interval


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
