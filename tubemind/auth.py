"""Authentication helpers and lightweight user persistence for TubeMind."""

from __future__ import annotations

import json
import secrets
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

from fasthtml.common import RedirectResponse, database

from tubemind.config import APP_ROOT, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, REDIRECT_URI

APP_ROOT.mkdir(parents=True, exist_ok=True)
db = database(str(APP_ROOT / "tubemind.db"))
users_table = db.t.users
if users_table not in db.t:
    users_table.create(dict(id=str, email=str, name=str, picture=str), pk="id")

ERROR_MESSAGES = {
    "no_code": "Google did not return an authorization code.",
    "bad_state": "Security check failed. Please try again.",
    "oauth_failed": "Could not complete sign-in with Google. Please try again.",
}


def google_auth_url(state: str) -> str:
    """Build the Google OAuth authorization URL for the login flow."""

    params = urlencode(
        {
            "client_id": GOOGLE_CLIENT_ID,
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "access_type": "online",
        }
    )
    return f"https://accounts.google.com/o/oauth2/v2/auth?{params}"


def google_exchange_code(code: str) -> dict:
    """Exchange an authorization code for Google OAuth tokens."""

    data = urlencode(
        {
            "code": code,
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "redirect_uri": REDIRECT_URI,
            "grant_type": "authorization_code",
        }
    ).encode()
    request = UrlRequest(
        "https://oauth2.googleapis.com/token",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urlopen(request) as response:
        return json.loads(response.read())


def google_userinfo(access_token: str) -> dict:
    """Fetch the authenticated user's basic profile fields from Google."""

    request = UrlRequest(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urlopen(request) as response:
        return json.loads(response.read())


def current_user(session) -> Any:
    """Resolve the current authenticated user from the session or return `None`."""

    user_id = session.get("user_id")
    if not user_id:
        return None
    try:
        return users_table[user_id]
    except Exception:
        return None


def begin_oauth_session(session) -> str:
    """Create and store the CSRF state token for a new login attempt."""

    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state
    return state


def logout_user(session) -> RedirectResponse:
    """Clear the current session and redirect to the login screen."""

    session.clear()
    return RedirectResponse("/login", status_code=303)
