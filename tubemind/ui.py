"""Server-rendered UI builders for the board-based TubeMind app.

Sprint 4 — UI Overhaul (Shrutika Yadav)
----------------------------------------
Replaced the old dual-screen seed/query interface with a single
ChatGPT-style chat experience. The layout mirrors modern AI chat apps:
- Left sidebar for board/topic switching
- Right panel is a full-height chat window
- Messages scroll in the middle
- Input bar is fixed at the bottom
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
# Login page (unchanged)
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
# ✨ NEW Chat UI — Sprint 4 (Shrutika Yadav)
# ---------------------------------------------------------------------------

def render_chat_bubble_user(question: str) -> Any:
    """Render one right-aligned user message bubble inside the chat thread.

    TubeMind's workspace models each note as a user question followed by one
    assistant answer bubble. Keeping the user half of that pair in a dedicated
    renderer makes it possible to reuse the same structure for durable notes
    and for optimistic client-side pending states injected before the HTMX
    request completes.
    """
    return Div(
        Div(
            P(question, cls="cb-text"),
            cls="cb cb-user",
        ),
        cls="cb-row cb-row-user",
    )


def render_chat_bubble_bot(note: dict[str, Any]) -> Any:
    """Render one persisted TubeMind answer bubble with its note metadata.

    This function is responsible for the durable assistant half of a saved
    board note. It reads the persisted note evidence count and creation time so
    the thread stays compact while still linking users to the full note detail
    page when they want the underlying sources.
    """
    note_id = int(note.get("id", 0) or 0)
    chunk_count = len(list_note_chunks(note_id))
    created = format_timestamp(int(note.get("created_at", 0) or 0))

    return Div(
        Div(
            Div(
                Span("🎬", cls="cb-icon"),
                Span("TubeMind", cls="cb-name"),
                cls="cb-header",
            ),
            P(str(note.get("answer", "") or ""), cls="cb-text"),
            Div(
                Span(f"📼  {chunk_count} source clip(s)", cls="cb-chip"),
                Span(created, cls="cb-chip cb-chip-muted") if created else "",
                A("View sources →", href=f"/notes/{note_id}", cls="cb-chip cb-chip-link"),
                cls="cb-footer",
            ),
            cls="cb cb-bot",
        ),
        cls="cb-row cb-row-bot",
    )


def render_chat_bubble_bot_skeleton(status_label: str = "Indexing sources and drafting answer...") -> Any:
    """Render the transient pending assistant bubble used during HTMX submits.

    The loading state should look like a real TubeMind note block rather than a
    separate spinner widget. This renderer mirrors the final assistant bubble
    shape with skeleton text lines and placeholder chips so the pending state
    can be cloned into the chat thread immediately while the server performs
    retrieval, indexing, and synthesis work.
    """

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
            Div(
                Span("", cls="cb-chip cb-chip-skeleton cb-chip-skeleton-wide", aria_hidden="true"),
                Span("", cls="cb-chip cb-chip-skeleton cb-chip-skeleton-mid", aria_hidden="true"),
                Span("", cls="cb-chip cb-chip-skeleton cb-chip-skeleton-short", aria_hidden="true"),
                cls="cb-footer cb-footer-pending",
            ),
            cls="cb cb-bot cb-bot-pending",
        ),
        cls="cb-row cb-row-bot",
    )


def render_chat_thread(notes: list[dict[str, Any]]) -> Any:
    """Render the full scrollable board conversation thread.

    The thread stays intentionally simple: every saved note becomes a user
    bubble followed by a TubeMind answer bubble in chronological order. Pending
    optimistic content is injected client-side so this server renderer can stay
    focused on durable state returned from SQLite.
    """
    if not notes:
        return Div(
            Div(
                P("🎬", cls="chat-empty-icon"),
                P("Ask anything about YouTube videos", cls="chat-empty-title"),
                P(
                    "TubeMind searches relevant YouTube videos, reads their transcripts, "
                    "and gives you a cited answer with timestamps — "
                    "so you don't have to watch hours of content.",
                    cls="chat-empty-sub",
                ),
                cls="chat-empty-state",
            ),
            cls="chat-thread",
            id="chat-thread",
        )

    rows = []
    for note in notes:
        rows.append(render_chat_bubble_user(str(note.get("question", "") or "")))
        rows.append(render_chat_bubble_bot(note))
    return Div(*rows, cls="chat-thread", id="chat-thread")


def render_chat_input(active_board: Optional[dict[str, Any]]) -> Any:
    """Render the sticky chat composer and the optimistic-submit client hooks.

    The composer does more than post the question form. It also owns the small
    client-side script that injects a temporary user bubble plus a pending
    TubeMind skeleton bubble immediately on submit, disables the controls while
    the HTMX request is in flight, and restores the composer if the request
    fails.
    """
    board_id_val = str(int(active_board.get("id", 0) or 0)) if active_board else ""

    return Div(
        Form(
            Input(type="hidden", name="board_id", value=board_id_val),
            Input(type="hidden", name="mode", value=DEFAULT_QUERY_MODE),
            Div(
                Textarea(
                    "",
                    name="question",
                    placeholder="Message TubeMind…",
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
        Div(render_chat_bubble_bot_skeleton(), id="tm-pending-template", cls="tm-hidden-template", aria_hidden="true"),
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

    function escapeHtml(value) {
        return String(value || '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function getThread() {
        return document.getElementById('chat-thread');
    }

    function boot() {
        var ta = document.getElementById('tm-input');
        if (!ta) return null;
        ta.style.height = 'auto';
        if (ta.dataset.tmBooted === 'true') {
            return ta;
        }
        ta.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 180) + 'px';
        });
        ta.dataset.tmBooted = 'true';
        ta.focus();
        return ta;
    }

    function scrollDown() {
        var t = getThread();
        if (t) t.scrollTop = t.scrollHeight;
    }

    function setPendingState(isPending) {
        var form = document.getElementById('tm-question-form');
        var ta = document.getElementById('tm-input');
        var send = document.getElementById('tm-send');
        var thinking = document.getElementById('tm-thinking');
        var titleGroup = document.getElementById('tm-board-title-group');
        var status = document.getElementById('tm-board-status');
        if (form) {
            form.dataset.pending = isPending ? 'true' : 'false';
            form.classList.toggle('is-pending', !!isPending);
        }
        if (ta) ta.disabled = !!isPending;
        if (send) send.disabled = !!isPending;
        if (thinking) thinking.classList.toggle('is-visible', !!isPending);
        if (!titleGroup) return;
        if (isPending) {
            if (!status) {
                status = document.createElement('span');
                status.id = 'tm-board-status';
                status.className = 'cw-status';
                titleGroup.insertBefore(status, titleGroup.firstChild);
            }
            if (!Object.prototype.hasOwnProperty.call(status.dataset, 'originalText')) {
                status.dataset.originalText = status.textContent || '';
            }
            status.dataset.pendingManaged = 'true';
            status.textContent = 'WORKING';
            status.classList.add('is-working');
        } else if (status && status.dataset.pendingManaged === 'true') {
            var originalText = status.dataset.originalText || '';
            status.textContent = originalText;
            status.classList.remove('is-working');
            delete status.dataset.pendingManaged;
            delete status.dataset.originalText;
            if (!originalText) {
                status.remove();
            }
        }
    }

    function removePendingRows() {
        var thread = getThread();
        if (!thread) return;
        thread.querySelectorAll('[data-tm-pending-row="true"]').forEach(function (node) {
            node.remove();
        });
        if (!thread.children.length && thread.dataset.hadEmptyState === 'true') {
            var empty = document.createElement('div');
            empty.className = 'chat-empty-state';
            empty.innerHTML = ''
                + '<p class="chat-empty-icon">🎬</p>'
                + '<p class="chat-empty-title">Ask anything about YouTube videos</p>'
                + '<p class="chat-empty-sub">TubeMind searches relevant YouTube videos, reads their transcripts, and gives you a cited answer with timestamps so you do not have to watch hours of content.</p>';
            thread.appendChild(empty);
        }
    }

    function appendPendingRows(questionText) {
        var form = document.getElementById('tm-question-form');
        var thread = getThread();
        var templateHost = document.getElementById('tm-pending-template');
        if (!form || !thread || !templateHost) return;
        if (form.dataset.pending === 'true') return;

        removePendingRows();
        var trimmed = String(questionText || '').trim();
        if (!trimmed) return;

        var empty = thread.querySelector('.chat-empty-state');
        thread.dataset.hadEmptyState = empty ? 'true' : 'false';
        if (empty) {
            empty.remove();
        }

        var userRow = document.createElement('div');
        userRow.className = 'cb-row cb-row-user';
        userRow.setAttribute('data-tm-pending-row', 'true');
        userRow.innerHTML = '<div class="cb cb-user"><p class="cb-text">' + escapeHtml(trimmed) + '</p></div>';
        thread.appendChild(userRow);

        var skeletonRow = templateHost.firstElementChild.cloneNode(true);
        skeletonRow.setAttribute('data-tm-pending-row', 'true');
        thread.appendChild(skeletonRow);
        scrollDown();
    }

    document.addEventListener('DOMContentLoaded', function () { boot(); scrollDown(); });
    document.addEventListener('htmx:beforeRequest', function (evt) {
        var form = findForm(evt);
        if (!form || form.id !== 'tm-question-form') return;
        var ta = document.getElementById('tm-input');
        var questionText = ta ? ta.value : '';
        if (!String(questionText || '').trim()) return;
        form.dataset.lastQuestion = questionText;
        appendPendingRows(questionText);
        if (ta) {
            ta.value = '';
            ta.style.height = 'auto';
        }
        setPendingState(true);
    });
    document.addEventListener('htmx:afterSwap', function () {
        setPendingState(false);
        boot();
        scrollDown();
    });
    document.addEventListener('htmx:responseError', function (evt) {
        var form = findForm(evt);
        if (!form || form.id !== 'tm-question-form') return;
        removePendingRows();
        setPendingState(false);
        var ta = boot();
        if (ta && form.dataset.lastQuestion) {
            ta.value = form.dataset.lastQuestion;
            ta.style.height = 'auto';
            ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
        }
    });
    document.addEventListener('htmx:sendError', function (evt) {
        var form = findForm(evt);
        if (!form || form.id !== 'tm-question-form') return;
        removePendingRows();
        setPendingState(false);
        var ta = boot();
        if (ta && form.dataset.lastQuestion) {
            ta.value = form.dataset.lastQuestion;
            ta.style.height = 'auto';
            ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
        }
    });
}());
"""),
        cls="tm-input-area",
    )


# render_question_form is called by routes.py — keep the name
def render_question_form(active_board: Optional[dict[str, Any]]) -> Any:
    return render_chat_input(active_board)


# ---------------------------------------------------------------------------
# Main workspace — full-height chat layout
# ---------------------------------------------------------------------------

def render_workspace(workspace: BoardWorkspace, user: dict[str, Any]) -> Any:
    """Render the full authenticated TubeMind workspace shell.

    This view combines the sidebar, active board header, chat thread, and
    composer into the single fragment that HTMX replaces after most actions.
    The board status badge intentionally exposes the persisted board state so
    long-running retrieval/indexing work can be reflected consistently in the
    top bar during optimistic submits and after the final server response.
    """
    board = workspace.active_board
    board_name = str(board.get("title", "") or "New board") if board else "TubeMind"
    board_status = str(board.get("status", "") or "").upper() if board else ""

    notice_block = Div(workspace.notice, cls="notice-banner") if workspace.notice else ""
    warning_block = Div(workspace.warning, cls="warning-banner") if workspace.warning else ""

    return Div(
        render_page_topbar(user),
        Div(
            # ── Sidebar ────────────────────────────────────────────────────
            render_sidebar(
                workspace.boards,
                int(board.get("id", 0) or 0) if board else None,
            ),
            # ── Chat window ────────────────────────────────────────────────
            Div(
                # top bar inside chat window
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
                # scrollable messages
                render_chat_thread(workspace.notes),
                # sticky input
                render_chat_input(board),
                cls="chat-window",
            ),
            cls="workspace-shell",
        ),
        cls="app-shell",
        id="workspace-root",
    )


# ---------------------------------------------------------------------------
# Note detail page (unchanged)
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
            cls="workspace-shell",
        ),
        cls="app-shell",
    )
