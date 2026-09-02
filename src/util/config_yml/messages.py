from datetime import UTC, datetime
from enum import StrEnum, auto

from pydantic import BaseModel, Field

from util.config_yml.intervals import INTERVAL_PATTERN, parse_interval


class TriggerEvent(StrEnum):
    on_chat_start = auto()
    on_chat_end = auto()
    on_chat_resume = auto()
    on_message = auto()


def _as_utc(value: datetime) -> datetime:
    """Treat a naive datetime as UTC; convert an aware one to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class Trigger(BaseModel):
    event: TriggerEvent | None = None
    after_messages: int | None = None
    start: datetime | None = None
    end: datetime | None = None
    freq_max: str | None = Field(default=None, pattern=INTERVAL_PATTERN)

    def match_trigger(
        self,
        event: TriggerEvent | None = None,
        after_messages: int | None = None,
        last_message: str | None = None,
    ) -> bool:
        if self.event and self.event != event:
            return False
        if self.after_messages and self.after_messages != after_messages:
            return False
        # start/end come from config.yml and are usually written with an offset
        # ("2025-01-01T00:00:00Z"). These used to be compared by stripping tzinfo,
        # which discards the offset instead of converting, shifting the window by
        # the host's UTC offset.
        now_utc = datetime.now(UTC)
        if self.start and _as_utc(self.start) > now_utc:
            return False
        if self.end and _as_utc(self.end) < now_utc:
            return False
        # last_message is written by chainlit_helpers as a naive local
        # datetime.now().isoformat(), so it keeps its own naive local clock.
        return not (
            self.freq_max
            and last_message
            and (
                parse_interval(self.freq_max)
                > datetime.now() - datetime.fromisoformat(last_message)
            )
        )


class Message(BaseModel):
    message: str
    enabled: bool = True
    recipients: list[str] | None = None
    trigger: Trigger
