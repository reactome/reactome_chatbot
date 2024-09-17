import hashlib
import hmac
import os

from chainlit.utils import mount_chainlit
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse
import requests

load_dotenv()

app = FastAPI()

HCAPTCHA_SECRET_KEY = os.getenv("HCAPTCHA_SECRET_KEY")
HCAPTCHA_SITE_KEY = os.getenv("HCAPTCHA_SITE_KEY")


def make_signature(value:str) -> str:
    return hmac.new(
        HCAPTCHA_SECRET_KEY.encode(),
        value.encode(),
        hashlib.sha256
    ).hexdigest()

def create_secure_cookie(value:str) -> str:
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
        request.url.path in ["/chat/verify_captcha", "/chat/verify_captcha_page", "/chat/static"]
        or request.url.path.startswith("/static")
        or not HCAPTCHA_SECRET_KEY
    ):
        response = await call_next(request)
        return response

    # Check if the user has completed the CAPTCHA verification
    captcha_verified = request.cookies.get("captcha_verified")

    # If CAPTCHA is not verified, block access
    if (not captcha_verified) or (not verify_secure_cookie(captcha_verified)):
        return RedirectResponse(url="/chat/verify_captcha_page")

    response = await call_next(request)
    return response


# Serve the CAPTCHA verification page (basic HTML form)
@app.get("/chat/verify_captcha_page")
async def captcha_page():
    html_content = f"""
    <html>
        <head>
            <script src="https://hcaptcha.com/1/api.js" async defer></script>
        </head>
        <body>
            <form action="/chat/verify_captcha" method="post">
                <div class="h-captcha" data-sitekey=\"{HCAPTCHA_SITE_KEY}\"></div>
                <br/>
                <button type="submit">Submit</button>
            </form>
        </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")


@app.post("/chat/verify_captcha")
async def verify_captcha(request: Request):
    form_data = await request.form()
    h_captcha_response = form_data.get("h-captcha-response")

    if not h_captcha_response:
        raise HTTPException(status_code=400, detail="CAPTCHA response is missing")

    # Send the CAPTCHA response to hCaptcha
    url = "https://hcaptcha.com/siteverify"
    data = {"secret": HCAPTCHA_SECRET_KEY, "response": h_captcha_response}

    # Perform request to hCaptcha's verification endpoint
    response = requests.post(url, data=data)
    result = response.json()

    # If CAPTCHA validation fails, return an error
    if not result.get("success"):
        raise HTTPException(status_code=400, detail="CAPTCHA verification failed")

    # Set a signed cookie to mark CAPTCHA as verified
    cookie_value = create_secure_cookie(h_captcha_response)
    redirect_response = RedirectResponse(url="/chat", status_code=303)
    redirect_response.set_cookie(
        key="captcha_verified",
        value=cookie_value,
        max_age=3600,  # Cookie expires in 1 hour
        secure=True,   # HTTPS only
        httponly=True  # inaccessible to client side JS
    )

    return redirect_response


mount_chainlit(app=app, target="bin/chat-chainlit.py", path="/chat")
