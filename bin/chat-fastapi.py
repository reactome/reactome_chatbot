import os

import requests
from chainlit.utils import mount_chainlit
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import RedirectResponse

load_dotenv()

app = FastAPI()

HCAPTCHA_SECRET_KEY = os.getenv("HCAPTCHA_SECRET_KEY")
HCAPTCHA_SITE_KEY = os.getenv("HCAPTCHA_SITE_KEY")


@app.middleware("http")
async def verify_captcha_middleware(request: Request, call_next):
    # Allow access to CAPTCHA pages and static files
    if (
        request.url.path in ["/verify_captcha", "/verify_captcha_page", "/static"]
        or request.url.path.startswith("/static")
        or not HCAPTCHA_SECRET_KEY
    ):
        response = await call_next(request)
        return response

    # Check if the user has completed the CAPTCHA verification
    captcha_verified = request.cookies.get("captcha_verified")

    # If CAPTCHA is not verified, block access
    if not captcha_verified:
        return RedirectResponse(url="/verify_captcha_page")

    response = await call_next(request)
    return response


# Serve the CAPTCHA verification page (basic HTML form)
@app.get("/verify_captcha_page")
async def captcha_page():
    html_content = f"""
    <html>
        <head>
            <script src="https://hcaptcha.com/1/api.js" async defer></script>
        </head>
        <body>
            <form action="/verify_captcha" method="post">
                <div class="h-captcha" data-sitekey=\"{HCAPTCHA_SITE_KEY}\"></div>
                <br/>
                <button type="submit">Submit</button>
            </form>
        </body>
    </html>
    """
    return Response(content=html_content, media_type="text/html")


@app.post("/verify_captcha")
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

    # Set cookie to mark CAPTCHA as verified
    redirect_response = RedirectResponse(url="/chainlit", status_code=303)
    redirect_response.set_cookie(
        key="captcha_verified", value="true", max_age=3600
    )  # Cookie expires in 1 hour

    return redirect_response


mount_chainlit(app=app, target="bin/chat-chainlit.py", path="/chainlit")
