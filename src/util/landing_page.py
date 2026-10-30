"""The pages around the chat: the landing page, and the pages a failed human
check ends on.

Both are rendered here rather than in `bin/chat-fastapi.py` so that they can be
rendered in a test. Importing that module mounts Chainlit, whose configuration is
global, and refuses to start without Turnstile keys -- which is why the checks in
`tests/api` shell out to a probe or grep the source instead of calling anything
(see `tests/api/test_chat_requires_human.py`). Nothing here needs the app.

Everything a deployment varies comes from `util.branding`; the two functions take
a `Branding` rather than reading config themselves, so which one is in effect is
decided in one place, at startup, where it can be logged.

`string.Template` rather than `str.format`: these pages are full of CSS braces.
"""

from string import Template

from util.branding import Branding

# The `$` placeholders are filled in by `render_landing_page`. The prose below the
# buttons names no knowledgebase, so it needs no placeholder.
_LANDING_PAGE = Template(
    """
    <html>
    <head>
        <link rel="icon" type="image/x-icon" href="$FAVICON_URL">
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
                <a href="$HOME_URL" target="_blank">
                    <img src="$LOGO_URL" alt="$KNOWLEDGEBASE Logo">
                </a>
            </div>
            <h1>$TITLE</h1>
            <p class="centered-text">$TAGLINE</p>

            <div class="button-container">
                <a class="button" href="$CHAT_URL/chat/guest/" target="_blank">Guest Access</a>
                <a class="button" href="$CHAT_URL/chat/personal/" target="_blank">Log In</a>
                <a class="button feedback-button" href="$FEEDBACK_URL" target="_blank">Feedback</a>
            </div>

            <p class="left-justified">Choose <strong>Guest Access</strong> to try the chatbot out. <strong>Log In</strong> will give an increased query allowance and securely stores your chat history so you can save and continue conversations.</p>
            <p class="left-justified">We encourage you to use the <strong>Feedback</strong> button to tell us about your experience with the chatbot and help us improve it.</p>
        </div>
    </body>
    </html>
    """
)

_ERROR_PAGE = Template(
    """
<html>
    <body>
        <h1>$ERROR_TITLE</h1>
        <p>If you believe this to be in error, please contact the maintainers to report an issue: $SUPPORT_EMAIL</p>
        <form action="$RETRY_URL" method="get">
            <button type="submit">Try again</button>
        </form>
    </body>
</html>
"""
)


def render_landing_page(branding: Branding, chat_url: str) -> str:
    """The page at `/chat/` that sends a reader into the chat.

    `chat_url` is CHAINLIT_URL -- the site the chat is reached at, since the two
    chat containers are mounted at their own paths under it.
    """
    return _LANDING_PAGE.substitute(
        FAVICON_URL=branding.favicon_url,
        HOME_URL=branding.home_url,
        LOGO_URL=branding.logo_url,
        KNOWLEDGEBASE=branding.knowledgebase,
        TITLE=f"Meet the {branding.chatbot_name} AI Chatbot!",
        TAGLINE=(
            f"Your new guide to {branding.knowledgebase}. Whether you're looking "
            "for specific genes and pathways or just browsing, our AI Chatbot is "
            "here to assist you!"
        ),
        CHAT_URL=chat_url,
        FEEDBACK_URL=branding.feedback_url,
    )


def render_error_page(branding: Branding, error_title: str, retry_url: str) -> str:
    """What a failed human check shows.

    `error_title` is one of the caller's own literals, never anything from the
    request, so it goes in as written.
    """
    return _ERROR_PAGE.substitute(
        ERROR_TITLE=error_title,
        SUPPORT_EMAIL=branding.support_email,
        RETRY_URL=retry_url,
    )
