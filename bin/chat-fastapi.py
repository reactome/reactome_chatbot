import hashlib
import hmac
import os
from string import Template

import requests
from chainlit.utils import mount_chainlit
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Response
from fastapi.responses import RedirectResponse

load_dotenv()

app = FastAPI()

CHAINLIT_URI = os.getenv("CHAINLIT_URI")

CLOUDFLARE_SECRET_KEY = os.getenv("CLOUDFLARE_SECRET_KEY")
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
    # Allow access to CAPTCHA pages and static files
    if (
        request.url.path
        in [
            f"{CHAINLIT_URI}/verify_captcha",
            f"{CHAINLIT_URI}/verify_captcha_page",
            f"{CHAINLIT_URI}/static",
        ]
        or request.url.path.startswith("/static")
        or not os.getenv("CLOUDFLARE_SECRET_KEY")
    ):
        response = await call_next(request)
        return response

    if request.url.scheme == "http":
        url = request.url.replace(scheme="https")
        return RedirectResponse(url=str(url))

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
                // Function called when CAPTCHA is completed
                function onSubmit(token) {{
                    document.getElementById('captcha-form').submit();  // Auto-submit form once CAPTCHA is validated
                }}

                // Optional: Automatically trigger Turnstile verification when the page loads
                window.onload = function() {{
                    setTimeout(function() {{
                        turnstile.execute();
                    }}, 1000);  // Trigger after 1 second (adjust as needed)
                }};
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
        return Response(content=error_html, media_type="text/html", status_code=400)

    client_ip: str
    if request.client:
        client_ip = request.client.host
    elif "X-Forwarded-For" in request.headers:
        client_ip = request.headers["X-Forwarded-For"].split(",")[0]
    else:
        error_html = ERROR_PAGE_TEMPLATE.substitute(
            error_title="Could not determine client host",
        )
        return Response(content=error_html, media_type="text/html", status_code=400)

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
        return Response(content=error_html, media_type="text/html", status_code=400)

    # Set a signed cookie to mark CAPTCHA as verified
    cookie_value = create_secure_cookie(cf_turnstile_response)
    redirect_response = RedirectResponse(url=f"{CHAINLIT_URI}", status_code=303)
    redirect_response.set_cookie(
        key="captcha_verified",
        value=cookie_value,
        max_age=3600,  # Cookie expires in 1 hour
        secure=True,  # HTTPS only
        httponly=True,  # inaccessible to client side JS
    )

    return redirect_response


mount_chainlit(app=app, target="bin/chat-chainlit.py", path=CHAINLIT_URI)
