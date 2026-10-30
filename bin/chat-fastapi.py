import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlsplit

from dotenv import load_dotenv

from util.secrets import SECRET_NAMES, get_secret, load_secrets_to_environ

# Before anything imports chainlit. `chainlit.utils` loads
# `chainlit.oauth_providers`, which reads OAUTH_*_CLIENT_SECRET once, at import:
# loaded after it, an OAuth secret supplied as a Docker secret was never seen,
# and logged-in chat failed at the OAuth callback (review, area 1b).
load_dotenv()
# The same list chat-chainlit uses. This was a second, hand-maintained copy
# that had already drifted from it in both directions.
load_secrets_to_environ(SECRET_NAMES)

import httpx
from chainlit.utils import mount_chainlit
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from agent.registry import build_graph, set_graph
from api.analysis_summary import router as analysis_summary_router
from api.answer import router as answer_router
from api.handoff import router as handoff_router
from util import captcha_cookie
from util.branding import Branding, branding_for
from util.caller_token import load_verifying_key
from util.captcha_scope import is_captcha_exempt
from util.config_yml import Config
from util.embedding_environment import EmbeddingEnvironment
from util.landing_page import render_error_page, render_landing_page
from util.logging import logging


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Build the one shared graph before serving, and close its pool after.

    At startup rather than at import: building it at import cost 85 seconds
    before the module finished loading, which is why the container needs a
    three-minute startup wait and why nothing could load this app in a test.
    Here the cost is the same but it is paid where a startup cost belongs, and
    both surfaces -- Chainlit and the answer endpoint -- get the same instance.
    """
    # At INFO, so it is silent where an operator has set LOG_LEVEL=error -- which
    # is the local default in .env, and is why this line did not appear the first
    # time it was checked. beta runs LOG_LEVEL=info and does print it.
    started = time.monotonic()
    # Before the graph: a missing verifying key must stop the process, and
    # spending 52 seconds building a graph first only delays the failure.
    _app.state.caller_token_key = load_verifying_key()
    # The release the served answers are built from, for FR-007 cache
    # invalidation. Read here rather than per request because the graph below is
    # built from these same bundles, so this value describes what is served even
    # if the pointer file changes underneath a running process.
    _app.state.release = EmbeddingEnvironment.get_release("reactome")
    graph = build_graph()
    set_graph(graph)
    logging.info("Agent graph ready in %.1fs", time.monotonic() - started)
    try:
        yield
    finally:
        await graph.close_pool()


app = FastAPI(lifespan=lifespan)

# Empty means unset. Compose turns an unset variable into "", and an empty
# CHAINLIT_URI left the captcha page redirecting to itself (review, 1b).
CHAINLIT_URI = os.getenv("CHAINLIT_URI") or None

# Defined after CHAINLIT_URI, which it is built from. The endpoint lives under
# the Chainlit mount point so one nginx location covers both.
API_PREFIX = f"{CHAINLIT_URI}/api" if CHAINLIT_URI else "/chat/api"
app.include_router(answer_router, prefix=API_PREFIX)
# Same prefix, so the captcha exemption below covers both. It verifies its own
# caller with a signed token and additionally requires evidence that a person
# is present -- a stricter bar than the answer endpoint's, because it discloses
# the user's own uploaded analysis rather than public pathway text.
app.include_router(analysis_summary_router, prefix=API_PREFIX)
app.include_router(handoff_router, prefix=API_PREFIX)
CHAINLIT_URL = os.getenv("CHAINLIT_URL")

CLOUDFLARE_SECRET_KEY = get_secret("CLOUDFLARE_SECRET_KEY")

# A human check on the chat is required unless a deployment says otherwise, and
# saying otherwise is explicit. Absence used to mean "no captcha to enforce", so
# a deployment that simply had no key served an ungated chat and reported
# nothing -- satisfied on paper, off in practice.
#
# Refusing to start rather than serving ungated is the same trade the answer
# endpoint's verifying key already makes: losing the feature is the correct
# failure, serving it unprotected is not.
CHAT_REQUIRES_HUMAN = os.getenv("CHAT_REQUIRES_HUMAN", "1").strip() not in {
    "0",
    "false",
    "no",
}
if CHAT_REQUIRES_HUMAN and not CLOUDFLARE_SECRET_KEY:
    raise RuntimeError(
        "CHAT_REQUIRES_HUMAN is on and no CLOUDFLARE_SECRET_KEY is configured, "
        "so the chat would be served with no human check at all. Provide the "
        "Turnstile keys, or set CHAT_REQUIRES_HUMAN=0 to say deliberately that "
        "this deployment does not want one."
    )
CLOUDFLARE_SITE_KEY = os.getenv("CLOUDFLARE_SITE_KEY")
if CHAT_REQUIRES_HUMAN and not CLOUDFLARE_SITE_KEY:
    # Without it the check page renders data-sitekey="None", which no one can
    # pass, while startup reported healthy (review, area 1b).
    raise RuntimeError(
        "CHAT_REQUIRES_HUMAN is on and no CLOUDFLARE_SITE_KEY is configured, "
        "so the human check could not be shown."
    )

# Which knowledgebase this deployment is. `profiles:` in config.yml already
# decides which graph answers questions and which chat profiles are offered, and
# config.yml is the one file a Plant Reactome deployment replaces (see
# compose.yaml), so the pages around the chat follow it rather than a second
# switch that could disagree with the first. No config.yml means
# config_default.yml, which names React-to-Me, so Reactome does nothing to keep
# what it has.
#
# Read once, at startup, like the release in `lifespan`. A page that changed brand
# because config.yml was edited under a running process -- while the graph kept
# answering from the old one -- would be worse than a stale one.
config: Config | None = Config.from_yaml()
BRANDING: Branding = branding_for(config.profiles if config else [])

HEADER_DONT_CACHE = {"Cache-Control": "no-store"}


def _error_page(error_title: str) -> Response:
    """What a failed human check ends on, in this deployment's brand."""
    return Response(
        content=render_error_page(BRANDING, error_title, CHAINLIT_URI or "/"),
        status_code=400,
        headers=HEADER_DONT_CACHE,
        media_type="text/html",
    )


def _gated(path: str) -> bool:
    """Whether the human check applies to this path."""
    return not is_captcha_exempt(
        path,
        chainlit_uri=CHAINLIT_URI,
        # The value resolved through get_secret, not os.environ. get_secret
        # prefers a mounted Docker secret file, so a deployment that mounts the
        # key rather than exporting it used to land here with the env var unset
        # and skip the captcha entirely -- switching off a protection the
        # operator had configured, silently.
        captcha_configured=bool(CLOUDFLARE_SECRET_KEY),
        # The website API verifies its own caller, with a signed token rather
        # than a captcha, so redirecting it to a captcha page would break it. This
        # is a deliberate hole in an authentication boundary, which is why the
        # rule was pinned by tests before it was widened. With the slash: the
        # bare prefix exempted `/chat/guest/api-anything` too (review, 1b).
        extra_prefixes=[f"{API_PREFIX}/"],
    )


def _set_pass(response: Response) -> None:
    response.set_cookie(
        key=captcha_cookie.COOKIE_NAME,
        value=captcha_cookie.mint(CLOUDFLARE_SECRET_KEY or "", time.time()),
        max_age=captcha_cookie.MAX_AGE_SECONDS,
        secure=True,  # HTTPS only
        httponly=True,  # inaccessible to client side JS
    )


class WebsocketGate:
    """The human check, for websocket connections.

    `@app.middleware("http")` never sees a websocket. Browsers met the gate only
    because socket.io starts on HTTP polling; a client opening the websocket
    directly got the whole chat with no captcha, and -- choosing its own session
    id -- no per-person limit either (review, area 1b, reproduced).

    Refused before the handshake completes, which the server sends as a 403.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] == "websocket" and _gated(scope.get("path", "")):
            headers = dict(scope.get("headers") or [])
            cookie = captcha_cookie.from_cookie_header(
                headers.get(b"cookie", b"").decode("latin-1")
            )
            verdict = captcha_cookie.check(
                cookie, CLOUDFLARE_SECRET_KEY or "", time.time()
            )
            if not verdict.ok:
                await receive()  # websocket.connect
                await send({"type": "websocket.close", "code": 1008})
                return
        await self.app(scope, receive, send)


app.add_middleware(WebsocketGate)


@app.middleware("http")
async def verify_captcha_middleware(
    request: Request, call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    path = request.url.path
    if CHAINLIT_URI and path == CHAINLIT_URI and not path.endswith("/"):
        # Safety: ensure the path is a clean, simple relative path with no
        # scheme/host/dot-segments before echoing it back.
        clean_path = urlsplit(path).path
        if ".." not in clean_path:
            return RedirectResponse(url=f"{clean_path}/")

    if not _gated(path):
        return await call_next(request)

    # There was a check here that refused any request whose Referer was an
    # http: page. The Referer is the page the reader came from, so it turned
    # away readers following a link from any plain-http site, and stopped no
    # one: a script omits the header (review, area 1b). TLS is Apache's job,
    # and the cookie is Secure.

    verdict = captcha_cookie.check(
        request.cookies.get(captcha_cookie.COOKIE_NAME),
        CLOUDFLARE_SECRET_KEY or "",
        time.time(),
    )
    if not verdict.ok:
        return RedirectResponse(url=f"{CHAINLIT_URI}/verify_captcha_page")

    response = await call_next(request)
    if verdict.renew:
        _set_pass(response)
    return response


# Serve the CAPTCHA verification page (basic HTML form)
@app.get(f"{CHAINLIT_URI}/verify_captcha_page")
async def captcha_page() -> Response:
    html_content = f"""
    <html>
        <head>
            <script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
        </head>
        <body>
            <form id="captcha-form" action="{CHAINLIT_URI}/verify_captcha" method="post">
                <div class="cf-turnstile" data-sitekey="{CLOUDFLARE_SITE_KEY}" data-callback="onSubmit"></div>
            </form>
            <script>
                // Continue in chat (spec 013): keep a handoff across this page.
                // A visitor without the captcha cookie arrives here from
                // /chat/guest/#handoff=<id>; the redirect keeps the fragment,
                // but the form POST below cannot carry it, so after the
                // redirect back the chat would open without its context.
                // sessionStorage is per tab and never sent to a server, so the
                // handoff stays tab-bound and out of logs; custom.js restores it.
                try {{
                    if (/(^|[#&])handoff=/.test(window.location.hash)) {{
                        sessionStorage.setItem('reactome-handoff-fragment', window.location.hash);
                    }}
                }} catch (e) {{}}
                let formSubmitted = false;
                function onSubmit(token) {{
                    if (!formSubmitted) {{
                        formSubmitted = true;
                        document.getElementById('captcha-form').submit();
                    }}
                }}
            </script>
        </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")


@app.post(f"{CHAINLIT_URI}/verify_captcha")
async def verify_captcha(request: Request) -> Response:
    # Bounded: anyone can reach this route. max_files=0 refuses file parts
    # outright -- they were spooled to the disk the whole host shares (review,
    # area 1b) -- and max_part_size holds the one field, a Turnstile token a
    # couple of hundred bytes long, to 8 KiB rather than Starlette's 1 MiB
    # default.
    #
    # max_part_size needs Starlette >= 0.44, which added it to Request.form().
    # The published image shipped 0.41.3, and starlette arrives transitively via
    # chainlit with nothing pinning the floor, so this raised
    # `TypeError: Request.form() got an unexpected keyword argument
    # 'max_part_size'` -- a 500 on the entry path, for every visitor, and only
    # on the image. A transitive version is not a floor; the floor lives in
    # pyproject now. Starlette 1.3.1 is where these limits are actually enforced
    # in FormParser, which is the floor declared there.
    form_data = await request.form(max_files=0, max_fields=4, max_part_size=8192)
    cf_turnstile_response = form_data.get("cf-turnstile-response")
    if not isinstance(cf_turnstile_response, str):
        return _error_page("CAPTCHA response is invalid")

    client_ip: str
    if request.client:
        client_ip = request.client.host
    elif "X-Forwarded-For" in request.headers:
        client_ip = request.headers["X-Forwarded-For"].split(",")[0]
    else:
        return _error_page("Could not determine client host")

    # Verify the CAPTCHA with Cloudflare
    url = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    data = {
        # Same reason: os.environ can be empty while the secret is mounted, and
        # sending Cloudflare a null secret fails in a way that reads as a captcha
        # problem rather than a configuration one.
        "secret": CLOUDFLARE_SECRET_KEY,
        "response": cf_turnstile_response,
        "remoteip": client_ip,
    }

    # Asynchronous: the blocking `requests.post` held the event loop every
    # session shares for a Cloudflare round trip on each anonymous POST, and
    # an unreachable or non-JSON reply was an unhandled 500 (review, 1b).
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            reply = await client.post(url, data=data)
        result = reply.json()
    except (httpx.HTTPError, ValueError):
        logging.warning("Turnstile siteverify failed; treating as not verified")
        result = {}
    if not isinstance(result, dict):
        result = {}

    # If CAPTCHA validation fails, return an error
    if not result.get("success"):
        return _error_page("CAPTCHA verification failed")

    redirect_response = RedirectResponse(
        url=f"{CHAINLIT_URI}/", status_code=302, headers=HEADER_DONT_CACHE
    )
    # A fresh nonce and issue time -- not the Turnstile token, which the old
    # cookie carried and which named nothing a limit could count.
    _set_pass(redirect_response)

    return redirect_response


@app.get("/chat/")
async def landing_page() -> HTMLResponse:
    """The page that introduces the chat, in this deployment's brand.

    Rendered in `util/landing_page.py`, where a test can render it without
    importing this module -- which mounts Chainlit and refuses to start without
    Turnstile keys. Only the branding varies; the layout is shared, so a Plant
    Reactome reader gets the same page with their knowledgebase's name, logo and
    contact on it.
    """
    # Empty rather than the literal "None" this interpolated when CHAINLIT_URL is
    # unset, which made every button read "None/chat/guest/". With nothing in
    # front of them they resolve against this origin, which is what an unset
    # CHAINLIT_URL should always have meant.
    return HTMLResponse(content=render_landing_page(BRANDING, CHAINLIT_URL or ""))


# Ensure all other endpoints remain mounted
mount_chainlit(app=app, target="bin/chat-chainlit.py", path=CHAINLIT_URI)
