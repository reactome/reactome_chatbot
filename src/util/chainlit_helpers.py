from datetime import datetime
from typing import Any, Iterable

import chainlit as cl
import chainlit.data as cl_data
from langchain_community.callbacks import OpenAICallbackHandler

from util.config_yml import Config, TriggerEvent

guest_user_metadata: dict[str, Any] = {}


def get_user_id() -> str | None:
    user: cl.User | None = cl.user_session.get("user")
    return user.identifier if user else None


def get_user_metadata(
    key: Any,
    default: Any | None = None,
    use_guest: bool = True,
) -> Any:
    user: cl.User | None = cl.user_session.get("user")
    if user:
        return user.metadata.get(key, default)
    elif use_guest:
        return guest_user_metadata.get(key, default)
    else:
        return default


def is_feature_enabled(config: Config | None, feature_id: str) -> bool:
    if not config:
        return True
    user_id: str | None = get_user_id()
    return config.get_feature(feature_id, user_id)


def save_openai_metrics(message_id: str, openai_cb: OpenAICallbackHandler) -> None:
    openai_metrics: dict[str, dict] = cl.user_session.get("openai_metrics", {})
    openai_metrics[message_id] = {
        prop: openai_cb.__dict__.get(prop, None)
        for prop in [
            "completion_tokens",
            "prompt_tokens",
            "prompt_tokens_cached",
            "reasoning_tokens",
            "successful_requests",
            "total_cost",
            "total_tokens",
        ]
    }
    cl.user_session.set("openai_metrics", openai_metrics)


def set_user_metadata(key: Any, value: Any, use_guest: bool = True) -> None:
    global guest_user_metadata  # not ideal, but works for now
    user: cl.User | None = cl.user_session.get("user")
    if user:
        user.metadata[key] = value
    elif use_guest:
        guest_user_metadata[key] = value


async def message_rate_limited(config: Config | None) -> bool:
    if not config:
        return False
    user_id: str | None = get_user_id()
    message_times_queue: list[str] = get_user_metadata("message_times_queue", [])
    is_limited: bool = config.get_message_rate_usage_limited(
        user_id, message_times_queue
    )
    if is_limited:
        if user_id:
            await send_messages(
                ["User messages quota reached. Please try again later."]
            )
        else:
            await send_messages(
                ["Public messages quota reached. Please try again later."]
            )
    set_user_metadata("message_times_queue", message_times_queue)
    return is_limited


async def send_messages(messages: Iterable[str]) -> None:
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


async def update_search_results(
    search_results: list[dict[str, str]],
    message: cl.Message,
) -> None:
    search_results_element = cl.CustomElement(
        name="SearchResults",
        props={"results": search_results},
    )
    message.elements = [search_results_element]  # type: ignore
    await message.update()


async def update_user() -> None:
    user: cl.User | None = cl.user_session.get("user")
    if user:
        await cl_data.get_data_layer().create_user(user)
