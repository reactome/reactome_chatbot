# mypy: disallow-untyped-calls=False
# chainlit ships no annotations for cl.user_session.get/set, cl.Message.send
# or get_data_layer. Not fixable here; it needs stubs upstream. The file is
# named with a hyphen, so it cannot be listed in [[tool.mypy.overrides]].
import os
from pathlib import Path

import chainlit as cl
from chainlit.data.base import BaseDataLayer
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.oauth_providers import providers
from chainlit.types import ThreadDict
from dotenv import load_dotenv
from langchain_community.callbacks import OpenAICallbackHandler
from langchain_core.messages import AIMessage, HumanMessage

from agent.profile_names import ProfileName
from agent.profiles import get_chat_profiles
from agent.profiles.base import OutputState
from agent.registry import get_graph
from analysis import gene_list
from analysis.client import (
    MAX_SUBMITTED_IDENTIFIERS,
    fetch_not_found,
    pathway_browser_url,
    submit_identifiers,
)
from gsa.chainlit_flow import (
    Attachment,
    matrix_attachment,
    result_file_kwargs,
    run_analysis,
)
from gsa.chat import HOW_TO_RUN_GSA, asks_to_run_gsa
from handoff import seed
from handoff.store import AnalysisHandoff, handoffs
from handoff.window import acknowledgement, claimed_id
from util.chainlit_helpers import (
    PrefixedS3StorageClient,
    is_feature_enabled,
    message_rate_limited,
    save_openai_metrics,
    static_messages,
    update_search_results,
)
from util.config_yml import Config
from util.config_yml.messages import TriggerEvent
from util.logging import logging
from util.orcid_provider import ORCIDOAuthProvider
from util.secrets import (
    SECRET_NAMES,
    get_db_uri,
    load_secrets_to_environ,
    mounted_secrets,
)

logger = logging.getLogger(__name__)

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
# Not built here any more. Building at module scope meant importing this file
# -- which mount_chainlit does -- constructed the graph and every BM25 index,
# 85 seconds before the app could serve or a test could load it. The FastAPI
# lifespan builds it now, and get_graph() falls back to building on demand so
# `chainlit run bin/chat-chainlit.py` still works.

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


async def run_gsa_analysis(attachment: Attachment) -> None:
    """Drive `gsa.chainlit_flow` with this session's chat operations.

    The flow takes these four as arguments so it can be tested without a
    browser; this is the only place that knows they are Chainlit.
    """
    # Created on the first update, not up front.
    #
    # It used to be sent before anything else, so it was the first message
    # in the thread -- and Chainlit scrolls the latest user message to the
    # top, which put the progress line above the fold. It updated faithfully
    # for minutes where nobody could see it; the user saw "Started." and then
    # nothing. Created lazily, it lands below "Started.", where they are
    # looking.
    progress: cl.Message | None = None

    async def ask_for_grouping() -> str | None:
        answer = await cl.AskUserMessage(
            content="Which group is each sample in?", timeout=600
        ).send()
        return (answer or {}).get("output") if answer else None

    async def send(text: str) -> None:
        await cl.Message(content=text).send()

    async def update_progress(text: str) -> None:
        nonlocal progress
        if progress is None:
            progress = cl.Message(content=text)
            await progress.send()
        else:
            progress.content = text
            await progress.update()

    async def send_file(path: Path) -> None:
        await cl.Message(
            content="", elements=[cl.File(**result_file_kwargs(path))]
        ).send()

    # The progress line is removed rather than marked "Done".
    #
    # A `finally` that sets "Done." runs on the failure paths too, so a user
    # whose analysis died would have been told it finished, one line above
    # the message explaining that it had not. `run_analysis` says what
    # happened on every path; this only has to stop the spinner.
    try:
        await run_analysis(
            attachment,
            ask_for_grouping=ask_for_grouping,
            send=send,
            update_progress=update_progress,
            send_file=send_file,
        )
    finally:
        if progress is not None:
            await progress.remove()


async def continue_from_handoff(handoff_id: str) -> None:
    """Start this thread from the summary the reader clicked from.

    The model gets the summary as its own previous turn plus the data it was
    built from, at the tier the reader chose (FR-003); the reader sees their
    summary verbatim. An unknown or expired handoff says so (FR-009), rather
    than leaving an empty chat the reader will assume has the context.
    """
    handoff = handoffs.get(handoff_id)
    if handoff is None:
        # Expired, never issued, or minted by another process -- the guest
        # and logged-in chats do not share this in-memory store.
        logger.info("handoff unavailable")
        await cl.Message(content=seed.UNAVAILABLE).send()
        return

    profile: str = (cl.user_session.get("chat_profile") or "").lower()
    # `on_chat_start` sets `thread_id` from the session id. A claim arriving
    # before it has run would otherwise seed a thread called "None".
    thread_id: str = cl.user_session.get("thread_id") or cl.user_session.get("id")
    data = None
    try:
        if isinstance(handoff, AnalysisHandoff):
            data = await seed.analysis_data(handoff)
        seeded = await get_graph().seed_history(
            profile, thread_id=thread_id, messages=seed.seeded_turn(handoff, data)
        )
    except Exception:
        # A failed fetch or graph update must not leave the reader in a chat
        # that silently lacks the context they came for.
        logger.exception("handoff seeding failed")
        seeded = False
    if not seeded:
        logger.warning("handoff not seeded", extra={"profile": profile})
        await cl.Message(content=seed.UNAVAILABLE).send()
        return

    logger.info(
        "handoff claimed",
        extra={
            "kind": handoff.kind,
            "tier": getattr(handoff, "tier", None),
            "with_data": data is not None,
        },
    )
    await cl.Message(content=seed.shown_to_reader(handoff)).send()


async def run_gene_list_analysis(text: str, identifiers: list[str]) -> None:
    """Over-representation on a pasted gene list, and seed it for follow-ups."""
    submitted = identifiers[:MAX_SUBMITTED_IDENTIFIERS]
    result = await submit_identifiers(submitted)
    if result is None:
        await cl.Message(content=gene_list.FAILED).send()
        return
    not_found = result.result.get("identifiersNotFound")
    unmatched = (
        await fetch_not_found(result.token)
        if isinstance(not_found, int) and not_found > 0
        else None
    )
    reply = gene_list.describe_overrepresentation(
        submitted,
        result.result,
        pathway_browser_url(result.token),
        unmatched,
        truncated=len(identifiers) > len(submitted),
    )
    await cl.Message(content=reply.text).send()
    logger.info(
        "gene list analysed",
        extra={"submitted": len(submitted), "has_pathways": reply.has_pathways},
    )
    if not reply.has_pathways:
        return
    # The exchange becomes the thread's previous turn, so "which of these
    # involve TP53?" is answered from this result. The reader typed the list
    # into this chat, whose every message goes to the model anyway.
    profile: str = (cl.user_session.get("chat_profile") or "").lower()
    thread_id: str = cl.user_session.get("thread_id") or cl.user_session.get("id")
    try:
        await get_graph().seed_history(
            profile,
            thread_id=thread_id,
            messages=[HumanMessage(content=text), AIMessage(content=reply.text)],
        )
    except Exception:
        # The reader has their result; only follow-ups lose it.
        logger.exception("gene list seeding failed")


@cl.on_window_message
async def on_window_message(message: object) -> None:
    """Claim a "Continue in chat" handoff posted by this tab (spec 013).

    Chainlit forwards every message posted in the page, including this
    server's own acknowledgement, so anything that is not a well-formed claim
    is ignored. The tab retries until acknowledged, so one claim can arrive
    several times; it is redeemed once per session and acknowledged every
    time.
    """
    handoff_id = claimed_id(message)
    if handoff_id is None:
        return

    claimed: set[str] = cl.user_session.get("handoff_claimed") or set()
    if handoff_id not in claimed:
        claimed.add(handoff_id)
        cl.user_session.set("handoff_claimed", claimed)
        await continue_from_handoff(handoff_id)

    await cl.send_window_message(acknowledgement(handoff_id))


@cl.on_message
async def main(message: cl.Message) -> None:
    if await message_rate_limited(config):
        return

    await static_messages(config, TriggerEvent.on_message)

    # An attached matrix routes to the analysis flow instead of the graph.
    #
    # Not a tool the agent calls: the run takes minutes, which is longer
    # than a chat turn, and the matrix is over a megabyte, which must never
    # enter the model's context. A tool call would put the model in the
    # middle of both problems.
    attachment = matrix_attachment(getattr(message, "elements", None))
    if attachment is not None:
        await run_gsa_analysis(attachment)
        return

    # Asked in words rather than by attaching a file. The answer path is
    # grounded in the user guide, which describes the website's GSA page, so
    # it said this chat could not do it. Answered here instead -- chat only,
    # because the same answer path serves the search page, which cannot.
    # A gene list is not a matrix: run what Reactome runs on a list,
    # over-representation, rather than explaining how to upload a matrix.
    # Before the GSA check, because the request that prompted this said "gsa".
    identifiers = gene_list.gene_list_request(message.content or "")
    if identifiers is not None:
        await run_gene_list_analysis(message.content, identifiers)
        return

    if asks_to_run_gsa(message.content or ""):
        await cl.Message(content=HOW_TO_RUN_GSA).send()
        return

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
    result: OutputState = await get_graph().ainvoke(
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
