"""Server-rendered UI builders for the board-based TubeMind app.

Sprint 5 — UI Revision (Shrutika Yadav)
----------------------------------------
Restored the notes/boards layout requested by Franklin:
- Board shows a masonry grid of note cards (Google Keep style)
- Each note card shows question + truncated answer
- Chat input pinned at bottom for asking new questions
- Sources detail page has tighter padding and cleaner layout
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fasthtml.common import (
    A, Button, Div, Form, H1, H2, H3, Iframe, Img, Input,
    Label, Option, P, Pre, Script, Select, Span, Textarea, Title,
)

from tubemind.auth import (
    ERROR_MESSAGES, begin_oauth_session, google_auth_url,
    list_note_chunks, list_note_queries,
)
from tubemind.config import (
    DEFAULT_QUERY_MODE, DEMO_AUTH_ENABLED,
    GOOGLE_AUTH_ENABLED, QUERY_MODE_LABELS,
)
from tubemind.models import BoardWorkspace


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def truncate_text(text: str, limit: int = 220) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def format_timestamp(ms: int) -> str:
    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000).strftime("%b %d, %Y at %I:%M %p")


# ---------------------------------------------------------------------------
# Shared chrome
# ---------------------------------------------------------------------------

def render_user_badge(user: dict[str, Any]) -> Any:
    avatar = (
        Img(src=user["picture"], cls="user-avatar", alt=user["name"])
        if user.get("picture")
        else Span(
            (user.get("name") or user.get("email") or "U")[:1].upper(),
            cls="user-avatar user-avatar-fallback",
        )
    )
    return Div(
        avatar,
        Span(user.get("name") or user.get("email"), cls="user-name"),
        A("Logout", href="/logout", cls="logout-link"),
        cls="user-badge",
    )


def render_theme_toggle() -> Any:
    return Button(
        Span("Theme", cls="theme-toggle-copy"),
        Span(
            Span("", cls="theme-toggle-knob"),
            cls="theme-toggle-track",
            **{"aria-hidden": "true"},
        ),
        Span("Light", cls="theme-toggle-state"),
        cls="theme-toggle",
        type="button",
        **{
            "data-theme-toggle": "",
            "aria-label": "Switch to dark mode",
            "aria-pressed": "false",
            "title": "Switch to dark mode",
        },
    )


def render_page_topbar(user: Optional[dict[str, Any]] = None) -> Any:
    actions = [render_theme_toggle()]
    if user:
        actions.append(render_user_badge(user))
    return Div(
        Div(*actions, cls="topbar-actions"),
        cls="page-topbar",
    )


# ---------------------------------------------------------------------------
# Login page
# ---------------------------------------------------------------------------

def render_login_page(session, error: str = "") -> Any:
    error_msg = ERROR_MESSAGES.get(error, "")
    actions: list[Any] = []

    if GOOGLE_AUTH_ENABLED:
        state = begin_oauth_session(session)
        actions.append(
            A("Sign in with Google", href=google_auth_url(state), role="button", cls="signin-btn")
        )
    if DEMO_AUTH_ENABLED:
        actions.append(
            A("Enter Demo Workspace", href="/auth/demo", role="button", cls="signin-btn signin-btn-secondary")
        )

    login_copy = "Ask a question, keep the evidence, and let each board stay anchored to one evolving topic."
    if DEMO_AUTH_ENABLED and not GOOGLE_AUTH_ENABLED:
        login_copy = "Demo mode is enabled. Enter the workspace without Google OAuth."
    elif DEMO_AUTH_ENABLED and GOOGLE_AUTH_ENABLED:
        login_copy = "Use Google sign-in or enter the demo workspace."

    return Title("TubeMind - Sign in"), Div(
        render_page_topbar(),
        Div(
            Div(
                Span("Research boards for YouTube knowledge", cls="login-badge"),
                H2("TubeMind", cls="login-title"),
                P(login_copy, cls="login-copy"),
                Div(error_msg, cls="login-error") if error_msg else "",
                Div(*actions, cls="login-actions") if actions else P(
                    "No login method configured. Set DEMO_AUTH_ENABLED=true or configure Google OAuth.",
                    cls="login-copy",
                ),
                P(
                    "Built for slower thinking, clearer summaries, and source-backed notes that still feel easy to scan.",
                    cls="login-footnote",
                ),
                cls="login-card",
            ),
            cls="login-shell",
        ),
        cls="app-shell app-shell-login",
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(boards: list[dict[str, Any]], active_board_id: int | None) -> Any:
    board_links = [
        A(
            Div(
                P(str(board.get("title", "") or "Untitled board"), cls="sidebar-board-title"),
                P(
                    str(
                        board.get("summary", "")
                        or format_timestamp(int(board.get("updated_at", 0) or 0))
                        or "No notes yet."
                    ),
                    cls="sidebar-board-copy",
                ),
                cls=f"sidebar-board {'is-active' if int(board.get('id', 0) or 0) == int(active_board_id or 0) else ''}",
            ),
            href=f"/boards/{int(board.get('id', 0) or 0)}",
            cls="sidebar-board-link",
        )
        for board in boards
    ]

    return Div(
        Div(
            Span("TubeMind", cls="sidebar-brand"),
            P("Topic-bound boards", cls="sidebar-copy"),
            cls="sidebar-head",
        ),
        Form(
            Button("+ New Board", type="submit", cls="sidebar-create-btn"),
            _hx_post="/api/boards",
            _hx_target="#workspace-root",
            _hx_swap="outerHTML",
        ),
        (
            Div(*board_links, cls="sidebar-board-list")
            if board_links
            else Div(
                P("Ask your first question or create a board to get started.", cls="sidebar-empty-copy"),
                cls="sidebar-empty",
            )
        ),
        cls="sidebar-shell",
    )


# ---------------------------------------------------------------------------
# Notes grid — Google Keep style
# ---------------------------------------------------------------------------

def render_note_card(note: dict[str, Any]) -> Any:
    """Render one note as a card in the masonry grid."""
    note_id = int(note.get("id", 0) or 0)
    question = str(note.get("question", "") or "")
    answer = truncate_text(str(note.get("answer", "") or ""), limit=200)
    created = format_timestamp(int(note.get("created_at", 0) or 0))
    chunk_count = len(list_note_chunks(note_id))

    return A(
        Div(
            P(question, cls="note-question"),
            P(answer, cls="note-answer") if answer else "",
            Div(
                Span(created, cls="note-meta") if created else "",
                Span(f"📼 {chunk_count} clip(s)", cls="note-meta") if chunk_count else "",
                Span("Sources →", cls="note-sources-link"),
                cls="note-meta-row",
            ),
            cls="note-card",
        ),
        href=f"/notes/{note_id}",
        cls="note-card-link",
    )


def render_notes_grid(notes: list[dict[str, Any]]) -> Any:
    """Render the masonry grid of note cards, or empty state."""
    if not notes:
        return Div(
            Div(
                P("🎬", cls="board-empty-icon"),
                P("No notes yet", cls="board-empty-title"),
                P(
                    "Ask a question below. TubeMind will search YouTube, read transcripts, "
                    "and save a cited answer as a note on this board.",
                    cls="board-empty-sub",
                ),
                cls="board-empty-state",
            ),
            cls="notes-grid-area",
            id="notes-grid",
        )

    cards = [render_note_card(note) for note in notes]
    return Div(
        Div(*cards, cls="note-grid"),
        cls="notes-grid-area",
        id="notes-grid",
    )


# ---------------------------------------------------------------------------
# Chat input (pinned to bottom)
# ---------------------------------------------------------------------------

def render_chat_bubble_bot_skeleton(status_label: str = "Searching YouTube and drafting answer...") -> Any:
    return Div(
        Div(
            Div(
                Span("🎬", cls="cb-icon"),
                Span("TubeMind", cls="cb-name"),
                Span("Working", cls="cb-chip cb-chip-pending"),
                cls="cb-header",
            ),
            P(status_label, cls="cb-pending-copy"),
            Div(
                Span("", cls="cb-skeleton-line cb-skeleton-line-wide", aria_hidden="true"),
                Span("", cls="cb-skeleton-line cb-skeleton-line-mid", aria_hidden="true"),
                Span("", cls="cb-skeleton-line cb-skeleton-line-wide", aria_hidden="true"),
                Span("", cls="cb-skeleton-line cb-skeleton-line-short", aria_hidden="true"),
                cls="cb-skeleton-copy",
            ),
            cls="cb cb-bot cb-bot-pending",
        ),
        cls="board-pending-row",
        id="board-pending-row",
    )


def render_chat_input(active_board: Optional[dict[str, Any]]) -> Any:
    """Sticky question composer pinned to the bottom of the board view."""
    board_id_val = str(int(active_board.get("id", 0) or 0)) if active_board else ""

    return Div(
        Div(render_chat_bubble_bot_skeleton(), id="tm-pending-template", cls="tm-hidden-template", aria_hidden="true"),
        Form(
            Input(type="hidden", name="board_id", value=board_id_val),
            Input(type="hidden", name="mode", value=DEFAULT_QUERY_MODE),
            Div(
                Textarea(
                    "",
                    name="question",
                    placeholder="Ask anything about YouTube videos…",
                    rows=1,
                    id="tm-input",
                    cls="tm-textarea",
                    **{"onkeydown": "tmKey(event)"},
                ),
                Button("↑", type="submit", cls="tm-send", title="Send", id="tm-send"),
                cls="tm-composer",
            ),
            Span("Searching YouTube...", cls="tm-thinking", id="tm-thinking"),
            _hx_post="/api/questions",
            _hx_target="#workspace-root",
            _hx_swap="outerHTML",
            id="tm-question-form",
        ),
        P("TubeMind searches YouTube videos and cites timestamps.", cls="tm-disclaimer"),
        Script("""
function tmKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        e.target.closest('form').requestSubmit();
    }
}
(function () {
    if (window.__tubeMindComposerBooted) return;
    window.__tubeMindComposerBooted = true;

    function findForm(evt) {
        var elt = evt && evt.detail && evt.detail.elt;
        if (elt && elt.id === 'tm-question-form') return elt;
        if (elt && elt.closest) return elt.closest('#tm-question-form');
        return document.getElementById('tm-question-form');
    }

    function boot() {
        var ta = document.getElementById('tm-input');
        if (!ta) return null;
        ta.style.height = 'auto';
        if (ta.dataset.tmBooted === 'true') return ta;
        ta.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 180) + 'px';
        });
        ta.dataset.tmBooted = 'true';
        ta.focus();
        return ta;
    }

    function setPendingState(isPending) {
        var form = document.getElementById('tm-question-form');
        var ta = document.getElementById('tm-input');
        var send = document.getElementById('tm-send');
        var thinking = document.getElementById('tm-thinking');
        var pending = document.getElementById('board-pending-row');
        if (form) form.classList.toggle('is-pending', !!isPending);
        if (ta) ta.disabled = !!isPending;
        if (send) send.disabled = !!isPending;
        if (thinking) thinking.classList.toggle('is-visible', !!isPending);
        if (pending) pending.style.display = isPending ? 'block' : 'none';
    }

    document.addEventListener('DOMContentLoaded', function () { boot(); });
    document.addEventListener('htmx:beforeRequest', function (evt) {
        var form = findForm(evt);
        if (!form || form.id !== 'tm-question-form') return;
        var ta = document.getElementById('tm-input');
        if (ta) { ta.value = ''; ta.style.height = 'auto'; }
        setPendingState(true);
    });
    document.addEventListener('htmx:afterSwap', function () {
        setPendingState(false);
        boot();
    });
    document.addEventListener('htmx:responseError', function (evt) {
        var form = findForm(evt);
        if (!form || form.id !== 'tm-question-form') return;
        setPendingState(false);
        boot();
    });
    document.addEventListener('htmx:sendError', function (evt) {
        var form = findForm(evt);
        if (!form || form.id !== 'tm-question-form') return;
        setPendingState(false);
        boot();
    });
}());
"""),
        cls="tm-input-area",
    )


# render_question_form is called by routes.py — keep this name
def render_question_form(active_board: Optional[dict[str, Any]]) -> Any:
    return render_chat_input(active_board)


# ---------------------------------------------------------------------------
# Main workspace
# ---------------------------------------------------------------------------

def render_workspace(workspace: BoardWorkspace, user: dict[str, Any]) -> Any:
    board = workspace.active_board
    board_name = str(board.get("title", "") or "New board") if board else "TubeMind"
    board_status = str(board.get("status", "") or "").upper() if board else ""

    notice_block = Div(workspace.notice, cls="notice-banner") if workspace.notice else ""
    warning_block = Div(workspace.warning, cls="warning-banner") if workspace.warning else ""

    return Div(
        render_page_topbar(user),
        Div(
            render_sidebar(
                workspace.boards,
                int(board.get("id", 0) or 0) if board else None,
            ),
            Div(
                Div(
                    Div(
                        Span(board_status, cls=f"cw-status {'is-working' if board_status == 'WORKING' else ''}".strip(), id="tm-board-status") if board_status else "",
                        Span(board_name, cls="cw-title"),
                        cls="cw-title-group",
                        id="tm-board-title-group",
                    ),
                    cls="cw-topbar",
                ),
                notice_block,
                warning_block,
                render_notes_grid(workspace.notes),
                render_chat_input(board),
                cls="board-view",
            ),
            cls="workspace-shell",
        ),
        cls="app-shell",
        id="workspace-root",
    )


# ---------------------------------------------------------------------------
# Note detail / sources page
# ---------------------------------------------------------------------------

def render_note_detail_page(
    user: dict[str, Any],
    boards: list[dict[str, Any]],
    note: dict[str, Any],
) -> Any:
    board = note.get("board") or {}
    chunks = list_note_chunks(int(note.get("id", 0) or 0))
    queries = list_note_queries(int(note.get("id", 0) or 0))

    query_items = [
        Div(
            P(str(item.get("youtube_query", "") or ""), cls="detail-query"),
            P(str(item.get("reason", "") or "Generated to extend the board corpus."), cls="detail-query-reason"),
            cls="detail-query-card",
        )
        for item in queries
    ]

    chunk_items = [
        Div(
            Div(
                Span(chunk.get("start_label", "") or "0:00", cls="chunk-time"),
                A(
                    chunk.get("video_title", "") or "Source video",
                    href=chunk.get("source_url", "#"),
                    target="_blank",
                    rel="noreferrer",
                    cls="chunk-video-link",
                ),
                cls="chunk-head",
            ),
            (
                Iframe(
                    src=chunk.get("embed_url", ""),
                    title=f"Video source for chunk {index}",
                    loading="lazy",
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share",
                    allowfullscreen="true",
                    cls="chunk-embed-frame",
                )
                if chunk.get("embed_url")
                else ""
            ),
            Div(
                A("Watch source", href=chunk.get("source_url", "#"), target="_blank", rel="noreferrer", cls="chunk-open-link"),
                A("Open board", href=f"/boards/{int(board.get('id', 0) or 0)}", cls="chunk-open-link"),
                cls="chunk-actions",
            ),
            Pre(str(chunk.get("content", "") or ""), cls="chunk-copy"),
            cls="chunk-card",
        )
        for index, chunk in enumerate(chunks, start=1)
    ]

    return Div(
        render_page_topbar(user),
        Div(
            render_sidebar(boards, int(board.get("id", 0) or 0)),
            Div(
                Div(
                    A("← Back to board", href=f"/boards/{int(board.get('id', 0) or 0)}", cls="back-link"),
                    H1(str(note.get("question", "") or ""), cls="detail-title"),
                    P(f"Asked {format_timestamp(int(note.get('created_at', 0) or 0))}", cls="detail-meta"),
                    cls="detail-head",
                ),
                Div(
                    H3("Answer", cls="detail-section-title"),
                    Pre(str(note.get("answer", "") or ""), cls="detail-answer"),
                    cls="detail-panel",
                ),
                Div(
                    H3("Generated YouTube queries", cls="detail-section-title"),
                    (
                        Div(*query_items, cls="detail-query-list")
                        if query_items
                        else P("TubeMind answered from the existing board corpus.", cls="detail-muted")
                    ),
                    cls="detail-panel",
                ),
                Div(
                    H3("Supporting video clips", cls="detail-section-title"),
                    (
                        Div(*chunk_items, cls="chunk-list")
                        if chunk_items
                        else P("No chunk previews found for this note.", cls="detail-muted")
                    ),
                    cls="detail-panel",
                ),
                cls="detail-main",
            ),
            cls="workspace-shell workspace-shell-detail",
        ),
        cls="app-shell",
    )