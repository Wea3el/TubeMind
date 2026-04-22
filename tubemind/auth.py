"""Authentication and durable SQLite helpers for TubeMind.

The board-based product needs durable user-scoped state that survives restarts:
boards, notes, evidence, and the currently selected board. This module owns the
shared SQLite database used for that purpose, alongside the Google OAuth helper
functions that already belong in the persistence/authentication layer.
"""

from __future__ import annotations

import json
import secrets
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

from fasthtml.common import RedirectResponse, database

from tubemind.config import (
    APP_ROOT,
    DEMO_AUTH_ENABLED,
    DEMO_USER_EMAIL,
    DEMO_USER_ID,
    DEMO_USER_NAME,
    DEMO_USER_PICTURE,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    REDIRECT_URI,
)
from tubemind.models import now_ms

APP_ROOT.mkdir(parents=True, exist_ok=True)
db = database(str(APP_ROOT / "tubemind.db"))

users_table = db.t.users
if users_table not in db.t:
    users_table.create(dict(id=str, email=str, name=str, picture=str, active_board_id=int), pk="id")

boards_table = db.t.boards
if boards_table not in db.t:
    boards_table.create(
        dict(
            id=int,
            user_id=str,
            title=str,
            summary=str,
            topic_anchor=str,
            status=str,
            created_at=int,
            updated_at=int,
            last_question_at=int,
        ),
        pk="id",
    )

board_notes_table = db.t.board_notes
if board_notes_table not in db.t:
    board_notes_table.create(
        dict(
            id=int,
            board_id=int,
            question=str,
            answer=str,
            query_mode=str,
            created_at=int,
        ),
        pk="id",
    )

board_queries_table = db.t.board_queries
if board_queries_table not in db.t:
    board_queries_table.create(
        dict(
            id=int,
            board_id=int,
            note_id=int,
            youtube_query=str,
            reason=str,
            created_at=int,
        ),
        pk="id",
    )

board_note_chunks_table = db.t.board_note_chunks
if board_note_chunks_table not in db.t:
    board_note_chunks_table.create(
        dict(
            id=int,
            note_id=int,
            chunk_order=int,
            video_id=str,
            video_title=str,
            source_url=str,
            embed_url=str,
            start_seconds=float,
            start_label=str,
            content=str,
        ),
        pk="id",
    )

board_videos_table = db.t.board_videos
if board_videos_table not in db.t:
    board_videos_table.create(
        dict(
            id=int,
            board_id=int,
            video_id=str,
            title=str,
            url=str,
            thumbnail=str,
            channel_title=str,
            origin_query=str,
            created_at=int,
        ),
        pk="id",
    )

ERROR_MESSAGES = {
    "no_code": "Google did not return an authorization code.",
    "bad_state": "Security check failed. Please try again.",
    "oauth_failed": "Could not complete sign-in with Google. Please try again.",
}


def _row_dict(row: Any) -> dict[str, Any] | None:
    """Normalize database row objects into plain dictionaries.

    FastHTML's database adapter can return row wrappers that behave like
    mappings but are less convenient to pass through the rest of the app.
    Centralizing conversion here keeps callers simple and makes missing-row
    handling consistent.
    """

    if row is None:
        return None
    return dict(row)


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
    """Fetch the authenticated user's basic Google profile fields."""

    request = UrlRequest(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urlopen(request) as response:
        return json.loads(response.read())


def current_user(session) -> Any:
    """Resolve the authenticated user from session storage.

    The rest of the app assumes a plain dictionary-like user record with an
    optional ``active_board_id`` field. Returning ``None`` instead of raising on
    missing data keeps route guards straightforward.
    """

    user_id = session.get("user_id")
    if not user_id:
        return None
    try:
        return _row_dict(users_table[user_id])
    except Exception:
        return None


def demo_user_profile() -> dict[str, Any]:
    """Return the synthetic coursework user profile used in demo mode."""

    return {
        "id": DEMO_USER_ID,
        "email": DEMO_USER_EMAIL,
        "name": DEMO_USER_NAME,
        "picture": DEMO_USER_PICTURE,
    }


def begin_oauth_session(session) -> str:
    """Create and store the CSRF state token for a new login attempt."""

    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state
    return state


def logout_user(session) -> RedirectResponse:
    """Clear the current session and redirect to the login page."""

    session.clear()
    return RedirectResponse("/login", status_code=303)


def upsert_user_profile(profile: dict[str, Any]) -> None:
    """Persist the signed-in user's profile and preserve their board selection.

    TubeMind treats Google as the identity source, but the local database still
    owns user-scoped application preferences such as the currently active
    board. This helper keeps those app-specific fields intact across repeated
    logins while still refreshing the user profile fields provided by Google.
    """

    existing = None
    try:
        existing = _row_dict(users_table[profile["id"]])
    except Exception:
        existing = None

    users_table.upsert(
        {
            "id": profile["id"],
            "email": profile.get("email", ""),
            "name": profile.get("name", ""),
            "picture": profile.get("picture", ""),
            "active_board_id": int(existing.get("active_board_id") or 0) if existing else None,
        },
        pk="id",
    )


def ensure_demo_user_session(session) -> dict[str, Any] | None:
    """Create or reuse the configured demo user for simplified deployments."""

    if not DEMO_AUTH_ENABLED:
        return None
    profile = demo_user_profile()
    upsert_user_profile(profile)
    session["user_id"] = profile["id"]
    session.pop("oauth_state", None)
    return current_user(session) or {**profile, "active_board_id": None}


def set_active_board(user_id: str, board_id: int | None) -> None:
    """Persist the board currently selected in the sidebar for one user.

    The active board controls which note grid, composer target, and retrieval
    context the workspace displays after navigation and refreshes. Storing that
    selection server-side keeps the HTMX flows simple and stable across page
    loads.
    """

    user = _row_dict(users_table[user_id])
    if not user:
        return
    user["active_board_id"] = int(board_id) if board_id else None
    users_table.update(user)


def create_board(user_id: str, title: str, topic_anchor: str = "", summary: str = "", status: str = "idle") -> dict[str, Any]:
    """Create a new durable board row for one user.

    Boards are the top-level topic containers in the new product shape. They
    need their own timestamps so the sidebar can order boards by recent
    activity even before the board has accumulated any notes.
    """

    timestamp = now_ms()
    inserted = boards_table.insert(
        dict(
            user_id=user_id,
            title=title,
            summary=summary,
            topic_anchor=topic_anchor,
            status=status,
            created_at=timestamp,
            updated_at=timestamp,
            last_question_at=0,
        )
    )
    if isinstance(inserted, dict):
        inserted_id = int(inserted.get("id", 0) or 0)
    else:
        inserted_id = int(inserted)
    return get_board_for_user(user_id, inserted_id) or {}


def update_board(board_id: int, **fields: Any) -> dict[str, Any] | None:
    """Update one board row and return the fresh persisted record.

    Board title, summary, status, and timestamps evolve as notes are added and
    the LLM refines the board framing. This helper keeps updates centralized so
    routes and services do not need to reimplement partial-row merge logic.
    """

    try:
        board = _row_dict(boards_table[board_id])
    except Exception:
        return None
    for key, value in fields.items():
        board[key] = value
    if "updated_at" not in fields:
        board["updated_at"] = now_ms()
    boards_table.update(board)
    return _row_dict(boards_table[board_id])


def touch_board(board_id: int, *, status: str | None = None, last_question_at: int | None = None) -> dict[str, Any] | None:
    """Refresh board activity timestamps without rebuilding the whole payload.

    Most question-answering actions should mark the board as recently used even
    if the title or summary did not change. This helper exists for that common
    path and avoids repetitive timestamp bookkeeping throughout the service
    layer.
    """

    fields: dict[str, Any] = {}
    if status is not None:
        fields["status"] = status
    if last_question_at is not None:
        fields["last_question_at"] = last_question_at
    return update_board(board_id, **fields)


def list_boards(user_id: str) -> list[dict[str, Any]]:
    """Return all boards for one user ordered by recent activity.

    Sidebar ordering is a product requirement, not a presentation accident.
    Keeping the ordering policy here ensures every route that needs boards sees
    the same recency-based list.
    """

    rows = boards_table.rows_where(
        "user_id = ?",
        [user_id],
        order_by="updated_at DESC, created_at DESC",
    )
    return [dict(row) for row in rows]


def get_board_for_user(user_id: str, board_id: int | None) -> dict[str, Any] | None:
    """Return one board only if it belongs to the requested user.

    Every board lookup in the app should be ownership-checked because board ids
    are stable integers exposed in URLs. Performing the user scoping here keeps
    the route layer small and avoids accidental cross-user reads.
    """

    if not board_id:
        return None
    rows = list(boards_table.rows_where("id = ? AND user_id = ?", [int(board_id), user_id], limit=1))
    return _row_dict(rows[0]) if rows else None


def list_board_notes(board_id: int) -> list[dict[str, Any]]:
    """Return all saved notes for one board in chronological order.

    The board grid and the board title-generation logic both depend on stable
    note ordering. Rendering oldest-to-newest keeps title derivation and note
    chronology aligned.
    """

    rows = board_notes_table.rows_where("board_id = ?", [board_id], order_by="created_at ASC")
    return [dict(row) for row in rows]


def create_board_note(board_id: int, question: str, answer: str, query_mode: str) -> dict[str, Any]:
    """Persist one board note and return the inserted row.

    A note is the durable unit shown as a Keep-style card. Saving the note
    first gives the service layer a stable ``note_id`` that related tables such
    as note evidence and LLM-generated YouTube queries can reference.
    """

    inserted = board_notes_table.insert(
        dict(
            board_id=board_id,
            question=question,
            answer=answer,
            query_mode=query_mode,
            created_at=now_ms(),
        )
    )
    if isinstance(inserted, dict):
        return dict(inserted)
    return dict(board_notes_table[int(inserted)])


def get_note_for_user(user_id: str, note_id: int) -> dict[str, Any] | None:
    """Return one note if and only if its parent board belongs to the user.

    Note detail pages are URL-addressable, so this helper performs the necessary
    board ownership check and then enriches the note with its parent board data
    for the detail route.
    """

    note_rows = list(board_notes_table.rows_where("id = ?", [note_id], limit=1))
    if not note_rows:
        return None
    note = dict(note_rows[0])
    board = get_board_for_user(user_id, int(note.get("board_id") or 0))
    if not board:
        return None
    note["board"] = board
    return note


def save_note_queries(board_id: int, note_id: int, queries: Iterable[dict[str, str]]) -> None:
    """Persist the YouTube searches the LLM chose for one note.

    The note detail page should explain not only the final answer but also how
    the system expanded its evidence set when the existing board corpus was not
    enough. Storing the generated queries makes that reasoning inspectable.
    """

    timestamp = now_ms()
    for item in queries:
        query_text = str(item.get("query", "") or "").strip()
        if not query_text:
            continue
        board_queries_table.insert(
            dict(
                board_id=board_id,
                note_id=note_id,
                youtube_query=query_text,
                reason=str(item.get("reason", "") or "").strip(),
                created_at=timestamp,
            )
        )


def list_note_queries(note_id: int) -> list[dict[str, Any]]:
    """Return the persisted generated YouTube queries for one note."""

    rows = board_queries_table.rows_where("note_id = ?", [note_id], order_by="id ASC")
    return [dict(row) for row in rows]


def replace_note_chunks(note_id: int, chunks: Iterable[dict[str, Any]]) -> None:
    """Replace the note-scoped evidence rows for one note.

    Evidence rows should match exactly what the user saw for that note at answer
    time. Replacing rather than merging ensures retries or future
    re-synthesizing operations cannot leave stale chunk rows behind.
    """

    board_note_chunks_table.delete_where("note_id = ?", [note_id])
    for index, chunk in enumerate(chunks, start=1):
        board_note_chunks_table.insert(
            dict(
                note_id=note_id,
                chunk_order=index,
                video_id=str(chunk.get("video_id", "") or ""),
                video_title=str(chunk.get("title", "") or ""),
                source_url=str(chunk.get("source_url", "") or chunk.get("url", "") or ""),
                embed_url=str(chunk.get("embed_url", "") or ""),
                start_seconds=float(chunk.get("start_seconds", 0.0) or 0.0),
                start_label=str(chunk.get("start_label", "") or ""),
                content=str(chunk.get("content", "") or "").strip(),
            )
        )


def list_note_chunks(note_id: int) -> list[dict[str, Any]]:
    """Return persisted note evidence rows in the original retrieval order."""

    rows = board_note_chunks_table.rows_where("note_id = ?", [note_id], order_by="chunk_order ASC")
    return [dict(row) for row in rows]


def list_board_videos(board_id: int) -> list[dict[str, Any]]:
    """Return all videos already indexed into a board corpus.

    The retrieval layer uses these rows both to avoid reinserting duplicate
    videos and to describe the board's current evidence set to the LLM when it
    decides whether more YouTube searching is necessary.
    """

    rows = board_videos_table.rows_where("board_id = ?", [board_id], order_by="created_at ASC")
    return [dict(row) for row in rows]


def upsert_board_videos(board_id: int, videos: Iterable[dict[str, Any]], origin_query: str) -> None:
    """Persist newly indexed videos while deduplicating by ``board_id`` and ``video_id``.

    Board corpora are cumulative, so re-asking a similar question should reuse
    the existing indexed videos instead of duplicating their metadata rows.
    """

    existing = {
        str(row.get("video_id") or "").strip(): dict(row)
        for row in board_videos_table.rows_where("board_id = ?", [board_id])
    }
    timestamp = now_ms()
    for item in videos:
        video_id = str(item.get("video_id", "") or item.get("videoId", "") or "").strip()
        if not video_id:
            continue
        payload = {
            "board_id": board_id,
            "video_id": video_id,
            "title": str(item.get("title", "") or "").strip(),
            "url": str(item.get("url", "") or "").strip(),
            "thumbnail": str(item.get("thumbnail", "") or "").strip(),
            "channel_title": str(item.get("channel_title", "") or item.get("channelTitle", "") or "").strip(),
            "origin_query": origin_query,
            "created_at": timestamp,
        }
        if video_id in existing:
            row = existing[video_id]
            row.update(payload)
            board_videos_table.update(row)
        else:
            board_videos_table.insert(payload)
