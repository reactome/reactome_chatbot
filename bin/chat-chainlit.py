import os

import chainlit as cl
from chainlit.data.base import BaseDataLayer
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.oauth_providers import providers
from chainlit.types import ThreadDict
from dotenv import load_dotenv
from langchain_community.callbacks import OpenAICallbackHandler

from agent.graph import AgentGraph
from agent.profiles import ProfileName, get_chat_profiles
from agent.profiles.base import OutputState
from util.chainlit_helpers import (
    PrefixedS3StorageClient,
    is_feature_enabled,
    message_rate_limited,
    save_openai_metrics,
    static_messages,
    update_search_results,
)
from util.config_yml import Config, TriggerEvent
from util.logging import logging
from util.orcid_provider import ORCIDOAuthProvider
from util.secrets import (
    SECRET_NAMES,
    get_db_uri,
    load_secrets_to_environ,
    mounted_secrets,
)

load_dotenv()
# Before anything reads os.environ. Docker secrets, where mounted, take
# precedence over .env; where not mounted, nothing changes.
_mounted = mounted_secrets(SECRET_NAMES)
load_secrets_to_environ(SECRET_NAMES)
if _mounted:
    # A count, not the names. mounted_secrets never opens a file, so the names
    # are not secret values -- but they are strings like OPENAI_API_KEY, and
    # CodeQL's clear-text-logging rule matches on that shape whatever their
    # provenance. Rather than dismiss a security alert to keep a nicety, this
    # logs the number; `ls /run/secrets` answers which, for anyone who needs it.
    logging.info(
        f"{len(_mounted)} of {len(SECRET_NAMES)} secrets supplied as Docker secrets"
    )

config: Config | None = Config.from_yaml()

profiles: list[ProfileName] = config.profiles if config else [ProfileName.React_to_Me]
llm_graph = AgentGraph(profiles, llm_config=config.llm if config else None)

POSTGRES_CHAINLIT_DB = os.getenv("POSTGRES_CHAINLIT_DB")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_CHAINLIT_PREFIX = os.getenv("S3_CHAINLIT_PREFIX")

# Once, not per call. Under Vault every call to get_db_uri mints a fresh
# short-lived credential, so calling it inside get_data_layer -- which chainlit
# invokes per session -- would issue a new lease for every visitor and leave
# them all outstanding until they expire. SQLAlchemy needs its dialect named.
CHAINLIT_DB_URI = get_db_uri(POSTGRES_CHAINLIT_DB, driver="psycopg")

if CHAINLIT_DB_URI:
    # A local so the narrowing survives into get_data_layer below: mypy does not
    # carry a module global's narrowed type into a nested function.
    _chainlit_db_uri: str = CHAINLIT_DB_URI

    storage_client: PrefixedS3StorageClient | None
    if S3_BUCKET and S3_CHAINLIT_PREFIX:
        storage_client = PrefixedS3StorageClient(S3_BUCKET, S3_CHAINLIT_PREFIX)
    else:
        storage_client = None

    @cl.data_layer
    def get_data_layer() -> BaseDataLayer:
        return SQLAlchemyDataLayer(
            conninfo=_chainlit_db_uri,
            storage_provider=storage_client,
        )

else:
    logging.warning("POSTGRES_CHAINLIT_DB undefined; Chainlit persistence disabled.")

if os.getenv("OAUTH_ORCID_CLIENT_ID") and not any(p.id == "orcid" for p in providers):
    providers.append(ORCIDOAuthProvider())

if os.getenv("CHAINLIT_AUTH_SECRET"):
    # chainlit 2.1 made this async and added the OIDC id_token as a fifth
    # argument. Neither is optional: chainlit awaits the result and calls it with
    # five arguments, so the old sync four-argument version would have failed at
    # login rather than at import.
    @cl.oauth_callback
    async def oauth_callback(
        provider_id: str,
        token: str,
        raw_user_data: dict[str, str],
        default_user: cl.User,
        id_token: str | None = None,
    ) -> cl.User | None:
        return default_user


# chainlit 2.1 passes the current user, so a deployment can vary the profile
# list per user. This one does not, but the parameter is required.
@cl.set_chat_profiles
async def chat_profiles(user: cl.User | None = None) -> list[cl.ChatProfile]:
    return [
        cl.ChatProfile(
            name=profile.name,
            markdown_description=profile.description,
        )
        for profile in get_chat_profiles(profiles)
    ]


@cl.on_chat_start
async def start() -> None:
    if cl.user_session.get("thread_id") is None:
        cl.user_session.set("thread_id", cl.user_session.get("id"))
    await static_messages(config, TriggerEvent.on_chat_start)


@cl.on_chat_resume
async def resume(thread: ThreadDict) -> None:
    await static_messages(config, TriggerEvent.on_chat_resume)


@cl.on_chat_end
async def end() -> None:
    await static_messages(config, TriggerEvent.on_chat_end)


@cl.on_message
async def main(message: cl.Message) -> None:
    if await message_rate_limited(config):
        return

    await static_messages(config, TriggerEvent.on_message)

    message_count: int = cl.user_session.get("message_count", 0) + 1
    cl.user_session.set("message_count", message_count)

    chat_profile: str = cl.user_session.get("chat_profile")

    thread_id: str = cl.user_session.get("thread_id")

    chainlit_cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True,
        force_stream_final_answer=True,  # we're not using prefix tokens
    )
    openai_cb = OpenAICallbackHandler()

    enable_postprocess: bool = is_feature_enabled(config, "postprocessing")
    result: OutputState = await llm_graph.ainvoke(
        message.content,
        chat_profile.lower(),
        callbacks=[chainlit_cb, openai_cb],
        thread_id=thread_id,
        enable_postprocess=enable_postprocess,
    )
    assistant_message: cl.Message | None = chainlit_cb.final_stream

    if (
        enable_postprocess
        and assistant_message
        and len(result["additional_content"]["search_results"]) > 0
    ):
        await update_search_results(
            result["additional_content"]["search_results"],
            assistant_message,
        )

    await static_messages(config, after_messages=message_count)

    save_openai_metrics(message.id, openai_cb)
