import os
from typing import Dict, List, Tuple

import httpx
from fastapi import HTTPException

from chainlit.oauth_providers import OAuthProvider
from chainlit.user import User


class ORCIDOAuthProvider(OAuthProvider):
    id = "orcid"
    env = ["OAUTH_ORCID_CLIENT_ID", "OAUTH_ORCID_CLIENT_SECRET"]

    def __init__(self):
        self.client_id = os.environ.get("OAUTH_ORCID_CLIENT_ID")
        self.client_secret = os.environ.get("OAUTH_ORCID_CLIENT_SECRET")
        self.authorize_url = "https://orcid.org/oauth/authorize"
        self.token_url = "https://orcid.org/oauth/token"
        self.user_info_url = "https://orcid.org/oauth/userinfo"
        self.authorize_params = {
            "response_type": "code",
            "scope": "/authenticate",
        }

        if prompt := self.get_prompt():
            self.authorize_params["prompt"] = prompt

    async def get_raw_token_response(self, code: str, url: str) -> dict:
        payload = {
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": url,
        }
        async with httpx.AsyncClient() as client:
            response = await client.post(self.token_url, data=payload)
            response.raise_for_status()
            return response.json()

    async def get_token(self, code: str, url: str) -> str:
        json = await self.get_raw_token_response(code, url)
        token = json.get("access_token")
        if not token:
            raise HTTPException(status_code=400, detail="Access token missing in the response")
        return token

    async def get_user_info(self, token: str) -> Tuple[Dict[str, str], User]:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                self.user_info_url,
                headers={"Authorization": f"Bearer {token}"},
            )
            response.raise_for_status()

        orcid_user = response.json()

        # ORCiD /userinfo returns the ORCID iD under "sub" (a stable identifier).
        # Use it so chat history persists per ORCID account.
        orcid_id = orcid_user.get("sub") or orcid_user.get("orcid")
        user = User(
            identifier=orcid_id or orcid_user.get("email", "orcid-user"),
            metadata={"provider": "orcid"},
        )
        return (orcid_user, user)
