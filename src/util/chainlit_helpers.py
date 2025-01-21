from datetime import datetime
from typing import Iterable

import chainlit as cl

from util.config_yml import Config, TriggerEvent


def get_user_id() -> str | None:
    user: cl.User | None = cl.user_session.get("user")
    return user.identifier if user else None


def is_feature_enabled(config: Config | None, feature_id: str) -> bool:
    if not config:
        return True
    user_id: str | None = get_user_id()
    return config.get_feature(feature_id, user_id)


async def send_messages(messages: Iterable[str]):
    for message in messages:
        await cl.Message(content=message).send()


async def static_messages(
    config: Config | None,
    event: TriggerEvent | None = None,
    after_messages: int | None = None,
) -> None:
    if not config:
        return
    user_id: str | None = get_user_id()
    last_static_messages: dict[str, str] = cl.user_session.get(
        "last_static_messages", {}
    )
    messages: dict[str, str] = config.get_messages(
        user_id, event, after_messages, last_static_messages
    )
    now: str = datetime.now().isoformat()
    for message_id in messages:
        last_static_messages[message_id] = now
    cl.user_session.set("last_static_messages", last_static_messages)

    chat_profile: str = cl.user_session.get("chat_profile")

    messages_formatted: Iterable[str] = map(
        lambda msg: msg.format(
            chat_profile=chat_profile,
            user_id=user_id,
        ),
        messages.values(),
    )
    await send_messages(messages_formatted)
