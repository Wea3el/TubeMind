"""Application factory and route registration for TubeMind."""

from __future__ import annotations

from fasthtml.common import FileResponse, JSONResponse, Link, RedirectResponse, Request, Script, fast_app

from tubemind.auth import (
    current_user,
    ensure_demo_user_session,
    get_board_for_user,
    get_note_for_user,
    google_exchange_code,
    google_userinfo,
    list_boards,
    logout_user,
    set_active_board,
    upsert_user_profile,
)
from tubemind.config import CSS_FILE, DEMO_AUTH_ENABLED, GOOGLE_AUTH_ENABLED, HTMX_SSE_EXTENSION_URL, SESSION_SECRET
from tubemind.services import get_user_app, shutdown_all_user_apps
from tubemind.ui import render_login_page, render_note_detail_page, render_sidebar, render_workspace

CSS_HREF = f"/static/tubemind.css?v={int(CSS_FILE.stat().st_mtime)}"

THEME_BOOTSTRAP_SCRIPT = """
(() => {
    const storageKey = "tubemind-theme";
    const root = document.documentElement;

    const resolveTheme = () => {
        const stored = window.localStorage.getItem(storageKey);
        if (stored === "light" || stored === "dark") {
            return stored;
        }
        return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
    };

    const updateToggleUi = (theme) => {
        document.querySelectorAll("[data-theme-toggle]").forEach((button) => {
            const isDark = theme === "dark";
            const state = button.querySelector(".theme-toggle-state");
            if (state) {
                state.textContent = isDark ? "Dark" : "Light";
            }
            button.setAttribute("aria-pressed", isDark ? "true" : "false");
            button.setAttribute("aria-label", isDark ? "Switch to light mode" : "Switch to dark mode");
            button.setAttribute("title", isDark ? "Switch to light mode" : "Switch to dark mode");
        });
    };

    const applyTheme = (theme) => {
        root.dataset.theme = theme;
        root.style.colorScheme = theme;
        updateToggleUi(theme);
    };

    const syncTheme = () => applyTheme(resolveTheme());

    applyTheme(resolveTheme());

    document.addEventListener("DOMContentLoaded", syncTheme);
    document.addEventListener("click", (event) => {
        const button = event.target.closest("[data-theme-toggle]");
        if (!button) {
            return;
        }
        const nextTheme = (root.dataset.theme || resolveTheme()) === "dark" ? "light" : "dark";
        window.localStorage.setItem(storageKey, nextTheme);
        applyTheme(nextTheme);
    });
    document.addEventListener("htmx:afterSwap", syncTheme);

    if (window.matchMedia) {
        const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
        const handlePreferenceChange = () => {
            if (!window.localStorage.getItem(storageKey)) {
                applyTheme(resolveTheme());
            }
        };
        if (mediaQuery.addEventListener) {
            mediaQuery.addEventListener("change", handlePreferenceChange);
        } else if (mediaQuery.addListener) {
            mediaQuery.addListener(handlePreferenceChange);
        }
    }
})();
"""


def create_app():
    """Create the FastHTML app and register TubeMind's board-based routes."""

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
            Script(THEME_BOOTSTRAP_SCRIPT),
            Link(rel="preconnect", href="https://fonts.googleapis.com"),
            Link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""),
            Link(
                rel="stylesheet",
                href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap",
            ),
            Link(rel="stylesheet", href=CSS_HREF),
        ),
        on_shutdown=[shutdown_all_user_apps],
    )

    def authenticated_user(session):
        """Resolve the session user, optionally provisioning the demo account."""

        user = current_user(session)
        if user:
            return user
        return ensure_demo_user_session(session)

    @rt("/static/tubemind.css")
    def get_tubemind_css():
        """Serve the standalone TubeMind stylesheet from disk."""

        return FileResponse(CSS_FILE, media_type="text/css")

    @rt("/health")
    def get_health():
        """Expose an unauthenticated health endpoint for deployment checks."""

        return JSONResponse({"ok": True})

    @rt("/login")
    def get_login(session, error: str = ""):
        """Render the sign-in screen or redirect authenticated users home."""

        user = current_user(session)
        if user:
            return RedirectResponse("/", status_code=303)
        return render_login_page(session, error=error)

    @rt("/auth/demo")
    def get_demo_auth(session):
        """Create the demo user session when demo auth is enabled."""

        if not DEMO_AUTH_ENABLED:
            return RedirectResponse("/login", status_code=303)
        ensure_demo_user_session(session)
        return RedirectResponse("/", status_code=303)

    @rt("/auth/callback")
    def get_auth_callback(request: Request, session, code: str = "", state: str = ""):
        """Complete the Google OAuth flow and create/update the local user row."""

        if not GOOGLE_AUTH_ENABLED:
            return RedirectResponse("/login", status_code=303)
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
            upsert_user_profile(profile)
            session["user_id"] = profile["id"]
            session.pop("oauth_state", None)
            return RedirectResponse("/", status_code=303)
        except Exception:
            return RedirectResponse("/login?error=oauth_failed", status_code=303)

    @rt("/logout")
    def get_logout(session):
        """Log the current user out and clear their session."""

        return logout_user(session)

    @rt("/")
    async def get_root(request: Request, session):
        """Render the board workspace using the user's persisted active board."""

        user = authenticated_user(session)
        if not user:
            return RedirectResponse("/login", status_code=303)
        app_state = await get_user_app(user["id"])
        workspace = app_state.build_workspace(int(user.get("active_board_id") or 0) or None)
        return render_workspace(workspace, user)

    @rt("/boards/{board_id}")
    async def get_board(request: Request, session, board_id: int):
        """Switch the active board and render the workspace for that board."""

        user = authenticated_user(session)
        if not user:
            return RedirectResponse("/login", status_code=303)
        board = get_board_for_user(user["id"], board_id)
        if not board:
            return RedirectResponse("/", status_code=303)
        set_active_board(user["id"], int(board["id"]))
        app_state = await get_user_app(user["id"])
        workspace = app_state.build_workspace(int(board["id"]))
        return render_workspace(workspace, {**user, "active_board_id": int(board["id"])})

    @rt("/notes/{note_id}")
    async def get_note(request: Request, session, note_id: int):
        """Render the dedicated note detail page for one persisted note."""

        user = authenticated_user(session)
        if not user:
            return RedirectResponse("/login", status_code=303)
        note = get_note_for_user(user["id"], note_id)
        if not note:
            return RedirectResponse("/", status_code=303)
        return render_note_detail_page(user, list_boards(user["id"]), note)

    @rt("/api/boards", methods=["POST"])
    async def api_create_board(request: Request, session):
        """Create an empty board and refresh the workspace shell."""

        user = authenticated_user(session)
        if not user:
            return RedirectResponse("/login", status_code=303)
        app_state = await get_user_app(user["id"])
        workspace = await app_state.create_empty_board()
        return render_workspace(workspace, {**user, "active_board_id": int(workspace.active_board.get("id", 0) or 0) if workspace.active_board else None})

    @rt("/api/questions", methods=["POST"])
    async def api_add_question(request: Request, session):
        """Answer a question inside the selected board and refresh the workspace."""

        user = authenticated_user(session)
        if not user:
            return RedirectResponse("/login", status_code=303)
        form = await request.form()
        board_id_raw = str(form.get("board_id", "") or "").strip()
        question = str(form.get("question", "") or "")
        mode = str(form.get("mode", "") or "")
        board_id = int(board_id_raw) if board_id_raw.isdigit() else None
        app_state = await get_user_app(user["id"])
        try:
            workspace = await app_state.answer_question(board_id, question, mode)
        except Exception as exc:
            workspace = app_state.build_workspace(board_id or int(user.get("active_board_id") or 0) or None, warning=str(exc))
        active_board_id = int(workspace.active_board.get("id", 0) or 0) if workspace.active_board else None
        return render_workspace(workspace, {**user, "active_board_id": active_board_id})

    @rt("/api/boards/sidebar")
    async def api_sidebar(request: Request, session):
        """Return the current board sidebar fragment for HTMX refreshes."""

        user = authenticated_user(session)
        if not user:
            return JSONResponse({"error": "not authenticated"}, status_code=401)
        return render_sidebar(list_boards(user["id"]), int(user.get("active_board_id") or 0) or None)

    @rt("/api/status")
    async def api_status(request: Request, session):
        """Expose a lightweight JSON view of the current board workspace."""

        user = authenticated_user(session)
        if not user:
            return JSONResponse({"error": "not authenticated"}, status_code=401)
        app_state = await get_user_app(user["id"])
        workspace = app_state.build_workspace(int(user.get("active_board_id") or 0) or None)
        return {
            "boards": workspace.boards,
            "active_board": workspace.active_board,
            "notes": workspace.notes,
        }

    return app


app = create_app()
