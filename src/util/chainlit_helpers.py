import os
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Iterable

import chainlit as cl
from chainlit.data import get_data_layer
from chainlit.data.storage_clients.s3 import S3StorageClient
from langchain_community.callbacks import OpenAICallbackHandler

from util.config_yml import Config, TriggerEvent
from util.config_yml.usage_limits import MessageRate

guest_user_metadata: dict[str, Any] = {}


class PrefixedS3StorageClient(S3StorageClient):
    def __init__(self, bucket: str, prefix: str, **kwargs: Any) -> None:
        super().__init__(bucket, **kwargs)
        self._prefix = PurePosixPath(prefix)

    async def upload_file(
        self,
        object_key: str,
        data: bytes | str,
        mime: str = "application/octet-stream",
        overwrite: bool = True,
    ) -> dict[str, Any]:
        object_key = str(self._prefix / object_key)
        return await super().upload_file(object_key, data, mime, overwrite)

    async def delete_file(self, object_key: str) -> bool:
        object_key = str(self._prefix / object_key)
        return await super().delete_file(object_key)

    async def get_read_url(self, object_key: str) -> str:
        object_key = str(self._prefix / object_key)
        return await super().get_read_url(object_key)


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
    rate_limit: MessageRate | None = config.get_message_rate_usage_limited(
        user_id, message_times_queue
    )
    if rate_limit:
        quota_message: str
        if user_id:
            quota_message = (
                "User messages quota reached. "
                f"You are allowed a maximum of {rate_limit.max_messages} messages every {rate_limit.interval}."
            )
        else:
            quota_message = "Our servers are currently overloaded.\n"
            login_uri: str | None = os.getenv("CHAINLIT_URI_LOGIN", "")
            if login_uri:
                quota_message += f"Please [log in]({login_uri}) to continue chatting and enjoy features like saved chat history and fewer limits."
            else:
                quota_message += "Please try again later."
        await send_messages([quota_message])
    set_user_metadata("message_times_queue", message_times_queue)
    await update_user()
    return rate_limit is not None


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
    message.elements.append(search_results_element)  # type: ignore[arg-type]
    await message.update()


async def update_user() -> None:
    user: cl.User | None = cl.user_session.get("user")
    if user:
        await get_data_layer().create_user(user)
