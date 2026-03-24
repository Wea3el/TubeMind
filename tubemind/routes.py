"""Application factory and route registration for TubeMind."""

from __future__ import annotations

import asyncio

from fasthtml.common import (
    FileResponse,
    JSONResponse,
    Link,
    RedirectResponse,
    Request,
    Script,
    StreamingResponse,
    fast_app,
    to_xml,
)

from tubemind.auth import current_user, google_exchange_code, google_userinfo, logout_user, users_table
from tubemind.config import (
    CSS_FILE,
    HTMX_SSE_EXTENSION_URL,
    MAX_VIDEOS_DEFAULT,
    MIN_SECONDS_DEFAULT,
    SESSION_SECRET,
    SSE_RETRY_MS,
)
from tubemind.models import seconds_to_label
from tubemind.services import get_user_app, shutdown_all_user_apps
from tubemind.ui import home_page, render_answer_panel, render_dashboard_fragment, render_login_page


def sse_message(fragment, *, event: str) -> str:
    """Wrap a rendered HTML fragment in a server-sent event envelope."""

    payload = to_xml(fragment).replace("\n", "\ndata: ")
    return f"event: {event}\ndata: {payload}\n\n"


def form_bool(value: str) -> bool:
    """Normalize checkbox-style form values into booleans."""

    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def create_app():
    """Create the FastHTML app and register TubeMind's routes."""

    app, rt = fast_app(
        title="TubeMind",
        pico=True,
        secret_key=SESSION_SECRET,
        hdrs=(
            Script(src=HTMX_SSE_EXTENSION_URL),
            Link(
                rel="icon",
                href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64'%3E%3Crect width='64' height='64' rx='18' fill='%230f766e'/%3E%3Cpath d='M15 19h34L37 33v12H27V33z' fill='%23fff8ee'/%3E%3C/svg%3E",
            ),
            Link(rel="preconnect", href="https://fonts.googleapis.com"),
            Link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""),
            Link(
                rel="stylesheet",
                href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap",
            ),
            Link(rel="stylesheet", href="/static/tubemind.css"),
        ),
        on_shutdown=[shutdown_all_user_apps],
    )

    @rt("/static/tubemind.css")
    def get_tubemind_css():
        """Serve the standalone TubeMind stylesheet from disk."""

        return FileResponse(CSS_FILE, media_type="text/css")

    @rt("/login")
    def get_login(session, error: str = ""):
        user = current_user(session)
        if user:
            return RedirectResponse("/", status_code=303)
        return render_login_page(session, error=error)

    @rt("/auth/callback")
    def get_auth_callback(request: Request, session, code: str = "", state: str = ""):
        if not code:
            return RedirectResponse("/login?error=no_code", status_code=303)
        if not state or state != session.get("oauth_state"):
            return RedirectResponse("/login?error=bad_state", status_code=303)
        try:
            token = google_exchange_code(code)
            access_token = token.get("access_token")
            if not access_token:
                raise RuntimeError("missing access token")
            profile = google_userinfo(access_token)
            users_table.upsert(
                {
                    "id": profile["id"],
                    "email": profile.get("email", ""),
                    "name": profile.get("name", ""),
                    "picture": profile.get("picture", ""),
                },
                pk="id",
            )
            session["user_id"] = profile["id"]
            session.pop("oauth_state", None)
            return RedirectResponse("/", status_code=303)
        except Exception:
            return RedirectResponse("/login?error=oauth_failed", status_code=303)

    @rt("/logout")
    def get_logout(session):
        return logout_user(session)

    @rt("/")
    async def get_root(request: Request, session):
        user = current_user(session)
        if not user:
            return RedirectResponse("/login", status_code=303)
        app_state = await get_user_app(user["id"])
        return home_page(app_state, user)

    @rt("/api/status")
    async def api_status(request: Request, session):
        user = current_user(session)
        if not user:
            return JSONResponse({"error": "not authenticated"}, status_code=401)
        app_state = await get_user_app(user["id"])
        return app_state.status_payload()

    @rt("/api/dashboard")
    async def api_dashboard(request: Request, session):
        user = current_user(session)
        if not user:
            return RedirectResponse("/login", status_code=303)
        app_state = await get_user_app(user["id"])
        return render_dashboard_fragment(app_state.status_payload())

    @rt("/api/dashboard/stream")
    async def api_dashboard_stream(request: Request, session):
        """Stream dashboard HTML fragments over SSE for the authenticated user."""

        user = current_user(session)
        if not user:
            return JSONResponse({"error": "not authenticated"}, status_code=401)

        app_state = await get_user_app(user["id"])

        async def event_stream():
            with app_state.lock:
                last_revision = app_state._dashboard_revision
                initial_status = app_state.status_payload()

            yield f"retry: {SSE_RETRY_MS}\n\n"
            yield sse_message(render_dashboard_fragment(initial_status), event="dashboard")

            while True:
                if await request.is_disconnected():
                    break

                next_revision, status = await asyncio.to_thread(
                    app_state.wait_for_dashboard_update,
                    last_revision,
                )
                if next_revision <= last_revision:
                    continue

                last_revision = next_revision
                yield sse_message(render_dashboard_fragment(status), event="dashboard")

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @rt("/api/search_youtube")
    async def api_search_youtube(
        request: Request,
        session,
        q: str = "",
        order: str = "relevance",
        minSeconds: str = "240",
        maxResults: str = "12",
        preferredChannels: str = "",
        excludedChannels: str = "",
        preferredOnly: str = "",
    ):
        user = current_user(session)
        if not user:
            return {"error": "not authenticated", "query": q, "results": []}
        app_state = await get_user_app(user["id"])
        try:
            min_seconds = int(minSeconds) if str(minSeconds).isdigit() else 240
            max_results = int(maxResults) if str(maxResults).isdigit() else 12
            max_results = max(1, min(25, max_results))
            videos = await app_state.youtube_search(
                q.strip(),
                max_videos=max_results,
                min_seconds=min_seconds,
                order=order,
                preferred_channels=app_state._parse_channel_filters(preferredChannels),
                excluded_channels=app_state._parse_channel_filters(excludedChannels),
                preferred_only=form_bool(preferredOnly),
            )
            return {
                "query": q,
                "order": order,
                "minSeconds": min_seconds,
                "filters": {
                    "preferredChannels": preferredChannels,
                    "excludedChannels": excludedChannels,
                    "preferredOnly": form_bool(preferredOnly),
                },
                "results": [
                    {
                        "videoId": video.video_id,
                        "title": video.title,
                        "channelTitle": video.channel_title,
                        "publishedAt": video.published_at,
                        "durationSec": video.duration_sec,
                        "durationLabel": seconds_to_label(video.duration_sec),
                        "url": video.url,
                        "thumbnail": video.thumbnail,
                    }
                    for video in videos
                ],
            }
        except Exception as exc:
            return {"error": str(exc), "query": q, "results": []}

    @rt("/api/seed_youtube", methods=["POST"])
    async def api_seed_youtube(
        request: Request,
        session,
        query: str = "",
        order: str = "relevance",
        max_videos: str = "8",
        min_seconds: str = "240",
        preferred_channels: str = "",
        excluded_channels: str = "",
        preferred_only: str = "",
    ):
        user = current_user(session)
        if not user:
            return RedirectResponse("/login", status_code=303)
        app_state = await get_user_app(user["id"])
        max_video_count = int(max_videos) if str(max_videos).isdigit() else MAX_VIDEOS_DEFAULT
        min_video_seconds = int(min_seconds) if str(min_seconds).isdigit() else MIN_SECONDS_DEFAULT
        max_video_count = max(1, min(15, max_video_count))
        min_video_seconds = max(60, min(3600, min_video_seconds))

        is_htmx = request.headers.get("hx-request", "").lower() == "true"

        try:
            job_id = app_state.start_youtube_index_job(
                query,
                max_videos=max_video_count,
                min_seconds=min_video_seconds,
                order=order,
                preferred_channels_raw=preferred_channels,
                excluded_channels_raw=excluded_channels,
                preferred_only=form_bool(preferred_only),
            )
        except ValueError as exc:
            if is_htmx:
                return render_dashboard_fragment(app_state.status_payload(), notice=str(exc))
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

        status = app_state.status_payload()
        if is_htmx:
            return render_dashboard_fragment(status)
        return {"ok": True, "job_id": job_id, **status}

    @rt("/api/query_youtube", methods=["POST"])
    async def api_query_youtube(request: Request, session, question: str = "", mode: str = "mix"):
        user = current_user(session)
        if not user:
            return JSONResponse({"error": "not authenticated"}, status_code=401)
        app_state = await get_user_app(user["id"])
        try:
            retrieval = await app_state.query_youtube(question, mode=mode)
            if request.headers.get("hx-request", "").lower() == "true":
                return render_answer_panel(retrieval=retrieval, indexed=app_state.state.youtube_indexed)
            return {"ok": True, "retrieval": retrieval}
        except Exception as exc:
            if request.headers.get("hx-request", "").lower() == "true":
                return render_answer_panel(error=str(exc), indexed=app_state.state.youtube_indexed)
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    return app


app = create_app()
