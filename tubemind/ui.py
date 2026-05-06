"""Server-rendered UI builders for the board-based TubeMind app."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fasthtml.common import A, Button, Details, Div, Form, H1, H2, H3, Iframe, Img, Input, Label, Option, P, Pre, Script, Select, Span, Summary, Textarea, Title

from tubemind.auth import ERROR_MESSAGES, begin_oauth_session, google_auth_url, list_note_chunks, list_note_queries
from tubemind.config import DEFAULT_QUERY_MODE, MAX_VIDEOS_DEFAULT, MIN_SECONDS_DEFAULT, MIN_VIDEOS_DEFAULT, QUERY_MODE_LABELS
from tubemind.models import BoardWorkspace


def truncate_text(text: str, limit: int = 220) -> str:
    """Clamp long text so note cards stay readable inside the masonry board."""

    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "..."


def format_timestamp(ms: int) -> str:
    """Convert a stored millisecond timestamp into a short local label."""

    if not ms:
        return ""
    return datetime.fromtimestamp(ms / 1000).strftime("%b %d, %Y at %I:%M %p")


def render_user_badge(user: dict[str, Any]) -> Any:
    """Render the authenticated user chip shown in the workspace header."""

    avatar = Img(src=user["picture"], cls="user-avatar", alt=user["name"]) if user.get("picture") else Span((user.get("name") or user.get("email") or "U")[:1].upper(), cls="user-avatar user-avatar-fallback")
    return Div(
        avatar,
        Span(user.get("name") or user.get("email"), cls="user-name"),
        A("Logout", href="/logout", cls="logout-link"),
        cls="user-badge",
    )


def render_login_page(session, error: str = "") -> Any:
    """Render the Google sign-in page for unauthenticated visitors."""

    state = begin_oauth_session(session)
    error_msg = ERROR_MESSAGES.get(error, "")
    return Title("TubeMind - Sign in"), Div(
        Div(
            H2("TubeMind", cls="login-title"),
            P("Sign in with Google to create topic-bound research boards from YouTube videos.", cls="login-copy"),
            Div(error_msg, cls="login-error") if error_msg else "",
            A("Sign in with Google", href=google_auth_url(state), role="button", cls="signin-btn"),
            cls="login-card",
        ),
        cls="login-shell",
    )


def render_sidebar(boards: list[dict[str, Any]], active_board_id: int | None) -> Any:
    """Render the persistent board list used for topic switching."""

    board_links = [
        Div(
            A(
                Div(
                    P(str(board.get("title", "") or "Untitled board"), cls="sidebar-board-title"),
                    P(
                        str(board.get("summary", "") or format_timestamp(int(board.get("updated_at", 0) or 0)) or "No notes yet."),
                        cls="sidebar-board-copy",
                    ),
                    cls=f"sidebar-board {'is-active' if int(board.get('id', 0) or 0) == int(active_board_id or 0) else ''}",
                ),
                href=f"/boards/{int(board.get('id', 0) or 0)}",
                cls="sidebar-board-link",
            ),
            Button(
                "×",
                cls="sidebar-board-delete",
                title="Delete board",
                **{
                    "hx-delete": f"/api/boards/{int(board.get('id', 0) or 0)}",
                    "hx-target": "#workspace-root",
                    "hx-swap": "outerHTML",
                    "hx-confirm": f"Delete \"{str(board.get('title', '') or 'Untitled board')}\"? This cannot be undone.",
                },
            ),
            cls="sidebar-board-row",
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
        Div(*board_links, cls="sidebar-board-list") if board_links else Div(P("Ask your first question or create a board to get started.", cls="sidebar-empty-copy"), cls="sidebar-empty"),
        cls="sidebar-shell",
    )


def render_board_header(board: Optional[dict[str, Any]]) -> Any:
    """Render the active board title and summary region above the note grid."""

    if not board:
        return Div(
            Span("No board selected", cls="board-kicker"),
            H1("Start a board with a question.", cls="board-title"),
            P("TubeMind will create a new board automatically from your first question and keep future notes in the same topic region.", cls="board-summary"),
            cls="board-header",
        )
    return Div(
        Span(str(board.get("status", "idle") or "idle").upper(), cls="board-kicker"),
        H1(str(board.get("title", "") or "Untitled board"), cls="board-title"),
        P(
            str(board.get("summary", "") or "Add notes to this board. After the third note, TubeMind will generate a board summary automatically."),
            cls="board-summary",
        ),
        cls="board-header",
    )


def render_question_form(active_board: Optional[dict[str, Any]]) -> Any:
    """Render the note composer that submits questions into the active board."""

    return Form(
        Input(type="hidden", name="board_id", value=str(int(active_board.get("id", 0) or 0)) if active_board else ""),
        Div(
            Div(
                Label("Question", cls="field-label"),
                Textarea("", name="question", placeholder="Example: What are the main tradeoffs these videos mention?", rows=4),
                P("TubeMind will first try the existing board corpus and then expand the board with new YouTube queries only if needed.", cls="field-help"),
                cls="field",
            ),
            Div(
                Label("Answer style", cls="field-label"),
                Select(
                    *[Option(label, value=value, selected=(value == DEFAULT_QUERY_MODE)) for value, label in QUERY_MODE_LABELS.items()],
                    name="mode",
                ),
                P("Balanced is the best default. Focused Detail is better for narrow follow-up questions.", cls="field-help"),
                cls="field",
            ),
            cls="composer-grid",
        ),
        Details(
            Summary("Search settings", cls="search-settings-toggle"),
            Div(
                Div(
                    Label("Min. video length (s)", cls="field-label"),
                    Input(type="number", name="min_seconds", value=str(MIN_SECONDS_DEFAULT), min="30", max="3600", step="30"),
                    cls="field",
                ),
                Div(
                    Label("Min videos to index", cls="field-label"),
                    Input(type="number", name="min_videos", value=str(MIN_VIDEOS_DEFAULT), min="1", max="20", step="1"),
                    cls="field",
                ),
                Div(
                    Label("Max videos to index", cls="field-label"),
                    Input(type="number", name="max_videos", value=str(MAX_VIDEOS_DEFAULT), min="1", max="20", step="1"),
                    cls="field",
                ),
                cls="search-settings-grid",
            ),
            cls="search-settings",
        ),
        Div(
            Button("Add Note", type="submit", cls="primary-btn"),
            Span("", cls="htmx-indicator tm-progress-msg", id="tm-progress-msg"),
            cls="composer-actions",
        ),
        _hx_post="/api/questions",
        _hx_target="#workspace-root",
        _hx_swap="outerHTML",
        id="tm-question-form",
        cls="composer-shell",
    )


def _make_kicker_script(board_id: int) -> Any:
    return Script(f"""
(function () {{
    var BOARD_ID = {board_id};
    var _pollTimer = null;

    function kicker() {{ return document.querySelector('.board-kicker'); }}
    function progressMsg() {{ return document.getElementById('tm-progress-msg'); }}

    function stopPolling() {{
        if (_pollTimer) {{ clearInterval(_pollTimer); _pollTimer = null; }}
        var msg = progressMsg();
        if (msg) msg.textContent = '';
    }}

    function startPolling() {{
        if (_pollTimer || !window.__tmBoardId) return;
        _pollTimer = setInterval(function () {{
            fetch('/api/boards/' + window.__tmBoardId + '/progress')
                .then(function (r) {{ return r.json(); }})
                .then(function (data) {{
                    var msg = progressMsg();
                    if (msg) msg.textContent = data.message || '';
                }})
                .catch(function () {{}});
        }}, 1200);
    }}

    function onBefore(e) {{
        var form = e.detail && e.detail.elt;
        if (!form || form.id !== 'tm-question-form') return;
        var k = kicker();
        if (k) {{
            k.dataset.orig = k.textContent;
            var q = (form.querySelector('[name="question"]') || {{}}).value || '';
            var label = q.trim() ? 'WORKING — ' + q.trim() : 'WORKING';
            k.textContent = label;
        }}
        startPolling();
    }}

    function onRestore(e) {{
        var form = e.detail && e.detail.elt;
        if (!form || form.id !== 'tm-question-form') return;
        stopPolling();
        var k = kicker();
        if (k && k.dataset.orig != null) {{ k.textContent = k.dataset.orig; delete k.dataset.orig; }}
    }}

    if (!window.__tmKickerBooted) {{
        window.__tmKickerBooted = true;
        document.addEventListener('htmx:beforeRequest', onBefore);
        document.addEventListener('htmx:afterSwap',     stopPolling);
        document.addEventListener('htmx:responseError', onRestore);
        document.addEventListener('htmx:sendError',     onRestore);
    }}
    // Always refresh board id on each render so the poller targets the right board.
    window.__tmBoardId = BOARD_ID;
}}());
""")


def render_note_card(note: dict[str, Any]) -> Any:
    """Render one Keep-like note card for the active board."""

    note_id = int(note.get("id", 0) or 0)
    chunk_rows = list_note_chunks(note_id)

    # Build deduped thumbnail strip from unique source videos (max 3)
    seen_video_ids: set[str] = set()
    thumb_els = []
    for c in chunk_rows:
        vid = str(c.get("video_id", "") or "").strip()
        if vid and vid not in seen_video_ids:
            seen_video_ids.add(vid)
            thumb_els.append(
                Img(
                    src=f"https://img.youtube.com/vi/{vid}/mqdefault.jpg",
                    alt=str(c.get("video_title", "") or ""),
                    cls="note-thumb",
                    loading="lazy",
                )
            )
            if len(thumb_els) >= 3:
                break

    source_chips = [
        Span(
            f"{str(c.get('video_title', 'Source') or 'Source')[:24].rstrip()} · {c.get('start_label', '0:00')}",
            cls="note-source-chip",
        )
        for c in chunk_rows[:3]
    ]
    extra_count = len(chunk_rows) - 3
    if extra_count > 0:
        source_chips.append(Span(f"+{extra_count} more", cls="note-source-chip note-source-chip--more"))
    return A(
        Div(
            P(str(note.get("question", "") or ""), cls="note-question"),
            Div(
                Pre(truncate_text(str(note.get("answer", "") or ""), limit=280), cls="note-answer-text"),
                Div(*thumb_els, cls="note-answer-thumbs") if thumb_els else "",
                cls="note-answer",
            ),
            Div(
                Span(format_timestamp(int(note.get("created_at", 0) or 0)), cls="note-meta"),
                cls="note-meta-row",
            ),
            Div(*source_chips, cls="note-source-row") if source_chips else "",
            cls="note-card",
        ),
        href=f"/notes/{note_id}",
        cls="note-card-link",
    )


def render_note_grid(notes: list[dict[str, Any]]) -> Any:
    """Render the active board's note board."""

    if not notes:
        return Div(
            P("This board does not have any notes yet. Ask a question above and TubeMind will turn it into the first card.", cls="empty-copy"),
            cls="empty-board",
        )
    return Div(*[render_note_card(note) for note in notes], cls="note-grid")


def render_workspace(workspace: BoardWorkspace, user: dict[str, Any]) -> Any:
    """Render the main board workspace with sidebar and active note grid."""

    board = workspace.active_board
    board_id_int = int(board.get("id", 0) or 0) if board else 0
    notice_block = Div(workspace.notice, cls="notice-banner") if workspace.notice else ""
    warning_block = Div(workspace.warning, cls="warning-banner") if workspace.warning else ""
    return Div(
        _make_kicker_script(board_id_int),
        Div(render_user_badge(user), cls="page-topbar"),
        Div(
            render_sidebar(workspace.boards, int(board.get("id", 0) or 0) if board else None),
            Div(
                notice_block,
                warning_block,
                render_board_header(board),
                render_question_form(board),
                render_note_grid(workspace.notes),
                cls="board-main",
            ),
            cls="workspace-shell",
        ),
        cls="app-shell",
        id="workspace-root",
    )


def render_note_detail_page(user: dict[str, Any], boards: list[dict[str, Any]], note: dict[str, Any]) -> Any:
    """Render the dedicated note detail page with note-scoped evidence only."""

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
                A(chunk.get("video_title", "") or "Source video", href=chunk.get("source_url", "#"), target="_blank", rel="noreferrer", cls="chunk-video-link"),
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

    sources_md = ""
    if chunks:
        refs = []
        for i, chunk in enumerate(chunks, 1):
            title = str(chunk.get("video_title", "") or "Source video").strip()
            timestamp = str(chunk.get("start_label", "") or "0:00").strip()
            src_url = str(chunk.get("source_url", "") or "#").strip()
            snippet = str(chunk.get("content", "") or "").strip()[:180]
            if snippet:
                snippet = snippet + ("…" if len(chunk.get("content", "") or "") > 180 else "")
            refs.append(f"**[{i}]** *{title}* — [{timestamp}]({src_url})\n\n> {snippet}")
        if refs:
            sources_md = "\n\n---\n\n**Sources**\n\n" + "\n\n".join(refs)

    answer_display = str(note.get("answer", "") or "") + sources_md

    return Div(
        Div(render_user_badge(user), cls="page-topbar"),
        Div(
            render_sidebar(boards, int(board.get("id", 0) or 0)),
            Div(
                Div(
                    A("Back to board", href=f"/boards/{int(board.get('id', 0) or 0)}", cls="back-link"),
                    H1(str(note.get("question", "") or ""), cls="detail-title"),
                    P(f"Asked {format_timestamp(int(note.get('created_at', 0) or 0))}", cls="detail-meta"),
                    cls="detail-head",
                ),
                Div(
                    H3("Answer", cls="detail-section-title"),
                    Div(answer_display, cls="detail-answer", **{"data-md": "1"}),
                    cls="detail-panel",
                ),
                Div(
                    H3("Generated YouTube queries", cls="detail-section-title"),
                    Div(*query_items, cls="detail-query-list") if query_items else P("TubeMind answered this note from the existing board corpus.", cls="detail-muted"),
                    cls="detail-panel",
                ),
                Div(
                    H3("Supporting video chunks", cls="detail-section-title"),
                    Div(*chunk_items, cls="chunk-list") if chunk_items else P("No persisted chunk previews were found for this note.", cls="detail-muted"),
                    cls="detail-panel",
                ),
                cls="detail-main",
            ),
            cls="workspace-shell",
        ),
        cls="app-shell",
    )
