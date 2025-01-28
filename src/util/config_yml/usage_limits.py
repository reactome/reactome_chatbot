from datetime import datetime

from pydantic import BaseModel

from util.config_yml.intervals import parse_interval


class MessageRate(BaseModel):
    users: list[str]
    max_messages: int
    interval: str

    def check_rate(self, message_times_queue: list[str]):
        now = datetime.now()
        while len(message_times_queue) > 0:
            if now - datetime.fromisoformat(message_times_queue[0]) > parse_interval(
                self.interval
            ):
                message_times_queue.pop(0)
            else:
                break
        if len(message_times_queue) < self.max_messages:
            message_times_queue.append(now.isoformat())
            return False  # not rate limited
        else:
            return True


class UsageLimits(BaseModel):
    message_rates: list[MessageRate]
