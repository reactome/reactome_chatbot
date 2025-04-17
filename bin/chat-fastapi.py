import hashlib
import hmac
import os
from string import Template

import requests
from chainlit.utils import mount_chainlit
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from util.secrets import get_secret, load_secrets_to_environ

load_dotenv()
load_secrets_to_environ([
    "CHAINLIT_AUTH_SECRET",
    "OAUTH_AUTH0_CLIENT_SECRET",
    "OAUTH_GOOGLE_CLIENT_SECRET",
    "OPENAI_API_KEY",
    "TAVILY_API_KEY",
])

app = FastAPI()

CHAINLIT_URI = os.getenv("CHAINLIT_URI")
CHAINLIT_URL = os.getenv("CHAINLIT_URL")

CLOUDFLARE_SECRET_KEY = get_secret("CLOUDFLARE_SECRET_KEY")
CLOUDFLARE_SITE_KEY = os.getenv("CLOUDFLARE_SITE_KEY")

ERROR_PAGE_TEMPLATE = Template(
    f"""
<html>
    <body>
        <h1>$error_title</h1>
        <p>If you believe this to be in error, please contact the maintainers to report an issue: help@reactome.org</p>
        <form action="{CHAINLIT_URI}" method="get">
            <button type="submit">Try again</button>
        </form>
    </body>
</html>
"""
)

HEADER_DONT_CACHE = {"Cache-Control": "no-store"}


def make_signature(value: str) -> str:
    if CLOUDFLARE_SECRET_KEY is None:
        raise ValueError("CLOUDFLARE_SECRET_KEY is not set")
    return hmac.new(
        CLOUDFLARE_SECRET_KEY.encode(), value.encode(), hashlib.sha256
    ).hexdigest()


def create_secure_cookie(value: str) -> str:
    signature = make_signature(value)
    return f"{value}|{signature}"


def verify_secure_cookie(cookie_value: str) -> bool:
    try:
        value, signature = cookie_value.split("|", 1)
        expected_signature = make_signature(value)
        return hmac.compare_digest(signature, expected_signature)
    except Exception:
        return False


@app.middleware("http")
async def verify_captcha_middleware(request: Request, call_next):
    if (
        CHAINLIT_URI
        and not request.url.path.startswith(CHAINLIT_URI)
        and request.url.path[-1] != "/"
    ):
        return RedirectResponse(url=f"{request.url.path}/")
    # Allow access to CAPTCHA pages and static files
    if (
        request.url.path
        in [
            "/chat/",
            f"{CHAINLIT_URI}/verify_captcha",
            f"{CHAINLIT_URI}/verify_captcha_page",
            f"{CHAINLIT_URI}/static",
        ]
        or request.url.path.startswith("/static")
        or not os.getenv("CLOUDFLARE_SECRET_KEY")
        or (CHAINLIT_URI and not request.url.path.startswith(CHAINLIT_URI))
    ):
        response = await call_next(request)
        return response

    host = request.headers.get("referer")
    if host and host.startswith("http:"):
        error_html = ERROR_PAGE_TEMPLATE.substitute(
            error_title="HTTPS is required for accessing this site",
        )
        return Response(content=error_html, status_code=400, media_type="text/html")

    # Check if the user has completed the CAPTCHA verification
    captcha_verified = request.cookies.get("captcha_verified")

    # If CAPTCHA is not verified, block access
    if not captcha_verified or not verify_secure_cookie(captcha_verified):
        return RedirectResponse(url=f"{CHAINLIT_URI}/verify_captcha_page")

    response = await call_next(request)
    return response


# Serve the CAPTCHA verification page (basic HTML form)
@app.get(f"{CHAINLIT_URI}/verify_captcha_page")
async def captcha_page():
    html_content = f"""
    <html>
        <head>
            <script src="https://challenges.cloudflare.com/turnstile/v0/api.js" async defer></script>
        </head>
        <body>
            <form id="captcha-form" action="{CHAINLIT_URI}/verify_captcha" method="post">
                <div class="cf-turnstile" data-sitekey="{os.getenv('CLOUDFLARE_SITE_KEY')}" data-callback="onSubmit"></div>
            </form>
            <script>
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
async def verify_captcha(request: Request):
    form_data = await request.form()
    cf_turnstile_response = form_data.get("cf-turnstile-response")
    if not isinstance(cf_turnstile_response, str):
        error_html = ERROR_PAGE_TEMPLATE.substitute(
            error_title="CAPTCHA response is invalid",
        )
        return Response(
            content=error_html,
            status_code=400,
            headers=HEADER_DONT_CACHE,
            media_type="text/html",
        )

    client_ip: str
    if request.client:
        client_ip = request.client.host
    elif "X-Forwarded-For" in request.headers:
        client_ip = request.headers["X-Forwarded-For"].split(",")[0]
    else:
        error_html = ERROR_PAGE_TEMPLATE.substitute(
            error_title="Could not determine client host",
        )
        return Response(
            content=error_html,
            status_code=400,
            headers=HEADER_DONT_CACHE,
            media_type="text/html",
        )

    # Verify the CAPTCHA with Cloudflare
    url = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
    data = {
        "secret": os.getenv("CLOUDFLARE_SECRET_KEY"),
        "response": cf_turnstile_response,
        "remoteip": client_ip,
    }

    # Perform request to Cloudflare Turnstile verification endpoint
    response = requests.post(url, data=data)
    result = response.json()

    # If CAPTCHA validation fails, return an error
    if not result.get("success"):
        error_html = ERROR_PAGE_TEMPLATE.substitute(
            error_title="CAPTCHA verification failed",
        )
        return Response(
            content=error_html,
            status_code=400,
            headers=HEADER_DONT_CACHE,
            media_type="text/html",
        )

    # Set a signed cookie to mark CAPTCHA as verified
    cookie_value = create_secure_cookie(cf_turnstile_response)
    redirect_response = RedirectResponse(
        url=f"{CHAINLIT_URI}/", status_code=302, headers=HEADER_DONT_CACHE
    )
    redirect_response.set_cookie(
        key="captcha_verified",
        value=cookie_value,
        max_age=3600,  # Cookie expires in 1 hour
        secure=True,  # HTTPS only
        httponly=True,  # inaccessible to client side JS
    )

    return redirect_response


@app.get("/chat/")
async def landing_page():
    html_content = Template(
        """
    <html>
    <head>
        <link rel="icon" type="image/x-icon" href="https://reactome.org/templates/favourite/favicon.ico">
        <style>
            body {
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                background-color: #f4f7fc;
                font-family: 'Arial', sans-serif;
                padding: 20px;
            }
            .container {
                text-align: center;
                border-radius: 12px;
                padding: 2rem;
                background: white;
                max-width: 700px;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.15);
            }
            .logo {
                margin-bottom: 1rem;
            }
            .logo img {
                max-width: 180px;
                height: auto;
            }
            h1 {
                font-size: 1.8rem;
                margin-bottom: 1rem;
                color: #333;
            }
            .centered-text {
                font-size: 1rem;
                color: #444;
                line-height: 1.6;
                margin-bottom: 1.5rem;
                text-align: center;
            }
            .button-container {
                display: flex;
                justify-content: center;
                gap: 10px;
                margin-bottom: 1.5rem;
            }
            .button {
                display: inline-block;
                padding: 0.75rem 1.5rem;
                font-size: 1rem;
                font-weight: bold;
                color: white;
                background-color: #007bff;
                border: none;
                border-radius: 8px;
                cursor: pointer;
                text-decoration: none;
                transition: all 0.3s ease;
                box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
            }
            .button:hover {
                background-color: #0056b3;
                transform: translateY(-2px);
                box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
            }
            .feedback-button {
                background-color: #28a745;
            }
            .feedback-button:hover {
                background-color: #218838;
            }
            .left-justified {
                font-size: 1rem;
                color: #555;
                text-align: left;
                margin-bottom: 1rem;
            }
            .left-justified strong {
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="logo">
                <img src="https://reactome.org/templates/favourite/images/logo/logo.png" alt="Reactome Logo">
            </div>
            <h1>Meet the React-to-Me AI Chatbot!</h1>
            <p class="centered-text">Your new guide to Reactome. Whether you're looking for specific genes and pathways or just browsing, our AI Chatbot is here to assist you!</p>

            <div class="button-container">
                <a class="button" href="$CHAINLIT_URL/chat/guest/" target="_blank">Guest Access</a>
                <a class="button" href="$CHAINLIT_URL/chat/personal/" target="_blank">Log In</a>
                <a class="button feedback-button" href="https://forms.gle/Rvzb8EA73yZs7wd38" target="_blank">Feedback</a>
            </div>

            <p class="left-justified">Choose <strong>Guest Access</strong> to try the chatbot out. <strong>Log In</strong> will give an increased query allowance and securely stores your chat history so you can save and continue conversations.</p>
            <p class="left-justified">We encourage you to use the <strong>Feedback</strong> button to tell us about your experience with the chatbot and help us improve it.</p>
        </div>
    </body>
    </html>
    """
    ).substitute(CHAINLIT_URL=CHAINLIT_URL)

    return HTMLResponse(content=html_content)


# Ensure all other endpoints remain mounted
mount_chainlit(app=app, target="bin/chat-chainlit.py", path=CHAINLIT_URI)
