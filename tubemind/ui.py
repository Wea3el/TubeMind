"""Server-rendered UI builders for TubeMind."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from fasthtml.common import A, Button, Div, Form, H1, H2, H3, Iframe, Img, Input, Label, Option, P, Pre, Select, Span, Textarea, Title

from tubemind.auth import ERROR_MESSAGES, begin_oauth_session, google_auth_url
from tubemind.config import DEFAULT_QUERY_MODE, MAX_RECOMMENDATIONS, PROMPT_SUGGESTIONS, QUERY_MODE_LABELS, SEARCH_ORDER_LABELS
from tubemind.services import TubeMindApp


def truncate_text(text: str, limit: int = 180) -> str:
    """Clamp long UI copy so cards remain readable and visually stable."""

    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def format_filter_text(values: list[str]) -> str:
    """Render stored channel filters back into textarea-friendly text."""

    return "\n".join(values)


def render_channel_settings_panel(saved_blacklist_text: str = "", notice: str = "") -> Any:
    """Render the user's always-on channel blacklist settings."""

    return Div(
        H3("Channel Settings", cls="section-title"),
        P("Save a general blacklist for channels you never want TubeMind to search or index.", cls="section-copy"),
        Div(notice, cls="login-error") if notice else "",
        Form(
            Div(
                Label("Always Exclude Channels", cls="field-label"),
                Textarea(saved_blacklist_text, name="global_excluded_channels", placeholder="One channel per line. TubeMind will apply this list to every future run automatically.", rows=4),
                P("Use this for creators you always want hidden. Run-specific exclusions in Step 1 still work on top of this saved blacklist.", cls="field-help"),
                cls="field",
            ),
            Div(
                Button("Save Channel Blacklist", type="submit", cls="primary-btn"),
                Span("Saving your settings...", cls="htmx-indicator small"),
                cls="action-row",
            ),
            _hx_post="/api/channel_settings",
            _hx_target="#channel-settings-panel",
            _hx_swap="outerHTML",
        ),
        id="channel-settings-panel",
        cls="panel",
    )


def render_search_preview_panel(
    *,
    results: Optional[list[Dict[str, Any]]] = None,
    query: str = "",
    error: str = "",
) -> Any:
    """Render candidate videos with per-run include checkboxes."""

    if error:
        return Div(
            H3("Choose Videos For This Run", cls="section-title"),
            P(error, cls="section-copy"),
            id="search-preview-panel",
            cls="preview-panel preview-panel-error",
        )

    if results:
        select_all_js = "document.querySelectorAll('#search-preview-panel input[name=\"selected_video_ids\"]').forEach((cb) => { cb.checked = true; });"
        clear_all_js = "document.querySelectorAll('#search-preview-panel input[name=\"selected_video_ids\"]').forEach((cb) => { cb.checked = false; });"
        preview_cards = [
            Label(
                Input(type="checkbox", name="selected_video_ids", value=item["videoId"], checked=True),
                Img(src=item.get("thumbnail", ""), alt=item.get("title", "Video thumbnail"), cls="preview-thumb")
                if item.get("thumbnail")
                else Div("No thumbnail", cls="preview-thumb preview-thumb-empty"),
                Div(
                    Div(
                        P(item.get("title", "Untitled video"), cls="item-title"),
                        P(item.get("channelTitle", "Unknown channel"), cls="item-copy"),
                        cls="preview-copy",
                    ),
                    Div(
                        Span(item.get("durationLabel", ""), cls="micro-pill") if item.get("durationLabel") else Span("Video", cls="micro-pill"),
                        A("Open on YouTube", href=item.get("url", "#"), target="_blank", rel="noreferrer", cls="tiny"),
                        cls="preview-line",
                    ),
                    cls="preview-meta",
                ),
                cls="preview-card",
            )
            for item in results
        ]
        return Div(
            H3("Choose Videos For This Run", cls="section-title"),
            P(f"Previewed {len(results)} candidate videos for '{query}'. Every video starts selected, so uncheck any results you want to leave out before indexing.", cls="section-copy"),
            Input(type="hidden", name="preview_loaded", value="true"),
            Div(
                Button("Select All", type="button", cls="prompt-chip", onclick=select_all_js),
                Button("Clear All", type="button", cls="prompt-chip", onclick=clear_all_js),
                P("These checkboxes only affect the current indexing run.", cls="small"),
                cls="action-row preview-actions",
            ),
            Div(*preview_cards, cls="preview-grid"),
            id="search-preview-panel",
            cls="preview-panel",
        )

    return Div(
        H3("Choose Videos For This Run", cls="section-title"),
        P("Preview candidate videos if you want per-video control before TubeMind starts transcript fetching.", cls="section-copy"),
        Div(
            P("Click `Preview Candidate Videos` to load selectable results. If you skip preview, TubeMind will use the top-ranked transcript-enabled matches automatically.", cls="item-copy"),
            cls="empty-state",
        ),
        id="search-preview-panel",
        cls="preview-panel",
    )


def friendly_job_stage(stage: str) -> str:
    """Translate internal job stage ids into dashboard labels."""

    return {
        "search": "Finding matching videos",
        "transcripts": "Collecting transcripts",
        "index": "Building the knowledge base",
        "done": "Ready for questions",
        "error": "Something needs attention",
    }.get(stage or "", "Waiting to start")


def progress_percent(progress: int, total: int) -> int:
    """Convert raw counters into a safe percentage for the progress bar."""

    if total <= 0:
        return 0
    return max(0, min(100, round((progress / total) * 100)))


def status_badge(status: Dict[str, Any]) -> tuple[str, str]:
    """Return the badge label and tone class for the current dashboard state."""

    if status["job"]["active"]:
        return "Indexing in progress", "working"
    if status["youtube"]["indexed"]:
        return "Ready to answer", "ready"
    if status["skipped"]:
        return "Needs attention", "warn"
    return "Waiting for a corpus", "idle"


def summarize_skip_reason(reason: str) -> str:
    """Normalize verbose transcript failures into user-facing copy."""

    raw = (reason or "").strip()
    lower = raw.lower()
    if "429" in lower or "too many requests" in lower:
        return "YouTube temporarily rate-limited transcript access for this video. TubeMind will try external transcript fallbacks when available."
    if "transcripts are disabled" in lower:
        return "This creator has disabled transcripts for the video."
    if "no transcripts are available" in lower or "no transcript found" in lower:
        return "No usable transcript was available for this video."
    if "video is no longer available" in lower:
        return "The video is no longer available."
    if "empty transcript" in lower:
        return "A transcript existed, but it did not contain usable text."
    if not raw:
        return "The transcript could not be fetched."
    return truncate_text(raw.splitlines()[0], limit=120)


def render_metric_card(label: str, value: str, hint: str) -> Any:
    """Render one summary metric card."""

    return Div(
        P(label, cls="metric-label"),
        P(value, cls="metric-value"),
        P(hint, cls="metric-hint"),
        cls="metric-card",
    )


def render_dashboard_fragment(status: Dict[str, Any], notice: str = "") -> Any:
    """Render the swappable dashboard body used by HTMX and SSE updates."""

    badge_text, badge_tone = status_badge(status)
    youtube = status["youtube"]
    filters = youtube.get("filters", {}) or {}
    job = status["job"]
    pct = progress_percent(job["progress"], job["total"])
    indexed_titles = youtube["titles"]
    recommendations = youtube.get("recommendations", []) or []
    preferred_channels = list(filters.get("preferred_channels", []) or [])
    excluded_channels = list(filters.get("excluded_channels", []) or [])
    global_excluded_channels = list(filters.get("global_excluded_channels", []) or [])
    preferred_only = bool(filters.get("preferred_only", False))
    rate_limit_count = sum(
        1
        for item in status["skipped"]
        if "429" in str(item.get("reason", "")).lower() or "too many requests" in str(item.get("reason", "")).lower()
    )

    indexed_items = [
        Div(
            Div(f"{idx:02d}", cls="source-num"),
            Div(
                P(A(title, href=youtube["urls"].get(title, "#"), target="_blank", rel="noreferrer"), cls="item-title"),
                P(youtube["urls"].get(title, ""), cls="mono-copy"),
            ),
            cls="source-item",
        )
        for idx, title in enumerate(indexed_titles, start=1)
    ]

    skipped_items = [
        Div(
            Div(Span("Skipped", cls="micro-pill"), P(item.get("title", "Untitled video"), cls="item-title"), cls="inline-meta"),
            P(summarize_skip_reason(str(item.get("reason", ""))), cls="item-copy"),
            P(truncate_text(str(item.get("reason", "")), limit=220), cls="tiny"),
            cls="skip-item",
        )
        for item in reversed(status["skipped"][-6:])
    ]

    recommendation_items = [
        A(
            Img(src=item.get("thumbnail", ""), alt=item.get("title", "Recommended video"), cls="recommend-thumb")
            if item.get("thumbnail")
            else Div("No image", cls="empty-state"),
            Div(
                P(item.get("title", "Untitled video"), cls="item-title"),
                P(item.get("channelTitle", "Unknown channel"), cls="item-copy"),
                Div(
                    Span(item.get("durationLabel", ""), cls="micro-pill") if item.get("durationLabel") else Span("Video", cls="micro-pill"),
                    Span("Open on YouTube", cls="tiny"),
                    cls="recommend-line",
                ),
                cls="recommend-meta",
            ),
            href=item.get("url", "#"),
            target="_blank",
            rel="noreferrer",
            cls="recommend-card",
        )
        for item in recommendations[:MAX_RECOMMENDATIONS]
    ]

    summary_children: list[Any] = [
        Span(badge_text, cls=f"status-pill {badge_tone}"),
        P(youtube["seed_query"] and f'Current topic: "{youtube["seed_query"]}"' or "No topic indexed yet", cls="section-copy"),
        H2(friendly_job_stage(job["stage"]), cls="section-title"),
        P(job["message"] or "Search a topic, index a few transcript-enabled videos, then ask questions in plain English.", cls="section-copy"),
    ]

    if preferred_channels or excluded_channels or global_excluded_channels:
        filter_chips = []
        if preferred_channels:
            label = "Only: " if preferred_only else "Prefer: "
            filter_chips.append(Span(f"{label}{', '.join(preferred_channels[:3])}", cls="micro-pill"))
            if len(preferred_channels) > 3:
                filter_chips.append(Span(f"+{len(preferred_channels) - 3} more preferred", cls="micro-pill"))
        if excluded_channels:
            filter_chips.append(Span(f"Hide: {', '.join(excluded_channels[:3])}", cls="micro-pill"))
            if len(excluded_channels) > 3:
                filter_chips.append(Span(f"+{len(excluded_channels) - 3} more excluded", cls="micro-pill"))
        if global_excluded_channels:
            filter_chips.append(Span(f"Always block: {', '.join(global_excluded_channels[:3])}", cls="micro-pill"))
            if len(global_excluded_channels) > 3:
                filter_chips.append(Span(f"+{len(global_excluded_channels) - 3} more saved blocks", cls="micro-pill"))
        summary_children.append(Div(*filter_chips, cls="status-row"))

    if notice:
        summary_children.append(Div(P(notice, cls="item-copy"), cls="skip-item"))

    if rate_limit_count:
        summary_children.append(
            Div(
                P(
                    "Transcript access is being rate-limited by YouTube. TubeMind now tries TranscriptAPI and other fallbacks, and the recommended videos panel still gives users direct access while indexing catches up.",
                    cls="item-copy",
                ),
                cls="skip-item",
            )
        )

    summary_children.extend(
        [
            Div(
                render_metric_card("Indexed Videos", str(youtube["count"]), "Videos ready for question answering."),
                render_metric_card("Skipped Videos", str(len(status["skipped"])), "Usually caused by missing captions or rate limits."),
                render_metric_card(
                    "Progress",
                    f"{pct}%",
                    f"{job['progress']} of {job['total']} current job steps completed." if job["total"] else "Waiting for the next indexing run.",
                ),
                cls="metrics-grid",
            ),
            Div(
                Div(Div(cls="progress-fill", style=f"width:{pct}%;"), cls="progress-track"),
                Div(Span(friendly_job_stage(job["stage"]), cls="small"), Span(job["message"] or "No active job", cls="small"), cls="status-row"),
                cls="progress-wrap",
            ),
        ]
    )

    return Div(
        Div(*summary_children, cls="panel"),
        Div(
            Div(
                Div(
                    H3("Indexed Video Library", cls="section-title"),
                    P("Each successful transcript becomes part of the corpus that answers are drawn from.", cls="section-copy"),
                    Div(*indexed_items, cls="list-stack") if indexed_items else Div(P("Your indexed videos will appear here once TubeMind finishes building the corpus.", cls="item-copy"), cls="empty-state"),
                    cls="panel",
                ),
                Div(
                    H3("Skipped or Unavailable Videos", cls="section-title"),
                    P("TubeMind only works with videos that expose transcripts. This panel helps explain what was left out.", cls="section-copy"),
                    Div(*skipped_items, cls="list-stack") if skipped_items else Div(P("No skipped videos so far. That usually means your current indexing run is healthy.", cls="item-copy"), cls="empty-state"),
                    cls="panel",
                ),
                cls="dashboard-main",
            ),
            Div(
                H3("Recommended Videos", cls="section-title"),
                P("Top matches for the current corpus topic. People can open these directly even before indexing finishes.", cls="section-copy"),
                Div(*recommendation_items, cls="recommend-stack") if recommendation_items else Div(P("Search a topic and the top 5 recommended videos will appear here with thumbnails and links.", cls="item-copy"), cls="empty-state"),
                cls="panel",
            ),
            cls="dashboard-body",
        ),
        id="dashboard-panels",
        **{"sse-swap": "dashboard"},
    )


def render_dashboard(status: Dict[str, Any], notice: str = "") -> Any:
    """Render the stable dashboard shell that owns the SSE connection."""

    return Div(
        render_dashboard_fragment(status, notice=notice),
        id="dashboard-stream",
        **{"hx-ext": "sse", "sse-connect": "/api/dashboard/stream"},
    )


def render_answer_panel(*, retrieval: Optional[Dict[str, Any]] = None, error: str = "", indexed: bool = False) -> Any:
    """Render a synthesized answer plus supporting transcript chunks.

    The transcript text shown here is the clean text indexed into LightRAG, not the
    timestamp-decorated source transcript. Timing is reconstructed separately from a
    sidecar transcript artifact so the answer UI can still jump back into the source
    video at the right moment without polluting the RAG corpus with timestamp tokens.
    """

    if error:
        return Div(
            H3("Question Could Not Be Answered", cls="section-title"),
            P(error, cls="answer-pre"),
            id="answer-panel",
            cls="answer-shell error",
        )
    if retrieval and (retrieval.get("answer") or retrieval.get("chunks")):
        answer_text = str(retrieval.get("answer") or "").strip()
        chunk_cards = [
            Div(
                Div(
                    Span(f"Chunk {idx}", cls="micro-pill"),
                    A(chunk.get("title", "Source video"), href=chunk.get("source_url") or chunk.get("url", "#"), target="_blank", rel="noreferrer", cls="item-title")
                    if chunk.get("source_url") or chunk.get("url")
                    else P(chunk.get("title", "Indexed transcript"), cls="item-title"),
                    cls="inline-meta",
                ),
                (
                    Div(
                        Iframe(src=chunk.get("embed_url", ""), title=f"Video source for chunk {idx}", loading="lazy", allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share", allowfullscreen="true", cls="chunk-embed-frame"),
                        P(f"Starts at {chunk.get('start_label', '0:00')}", cls="tiny"),
                        cls="chunk-embed",
                    )
                    if chunk.get("embed_url")
                    else ""
                ),
                Pre(chunk.get("content", ""), cls="answer-pre chunk-copy"),
                cls="panel tight",
            )
            for idx, chunk in enumerate(retrieval["chunks"], start=1)
        ]
        answer_block = (
            Div(
                Div(
                    H3("Answer", cls="section-title"),
                    P("Synthesized from the currently indexed YouTube transcript corpus.", cls="section-copy"),
                    cls="answer-summary-head",
                ),
                Pre(answer_text, cls="answer-pre answer-summary-copy"),
                cls="answer-summary",
            )
            if answer_text and answer_text.lower() not in {"none", "null"}
            else Div(
                H3("Answer", cls="section-title"),
                P("TubeMind found supporting transcript evidence, but it could not synthesize a final answer for this question yet.", cls="section-copy"),
                cls="answer-summary answer-summary-empty",
            )
        )
        return Div(
            answer_block,
            Div(
                H3("Supporting Transcript Chunks", cls="section-title"),
                P("These passages were retrieved from the indexed YouTube transcript corpus and can be used to verify the answer.", cls="section-copy"),
                P(f"Question: {retrieval.get('question', '')} | Mode: {str(retrieval.get('mode', DEFAULT_QUERY_MODE)).upper()}", cls="small"),
                Div(*chunk_cards, cls="list-stack") if chunk_cards else Div(
                    P("No transcript chunk previews were returned for this answer.", cls="item-copy"),
                    cls="empty-state",
                ),
                cls="answer-evidence",
            ),
            id="answer-panel",
            cls="answer-shell",
        )
    placeholder = (
        "Ask about themes, disagreements, examples, takeaways, or summaries once your indexing run is ready."
        if indexed
        else "Index a topic first, then ask natural-language questions about the videos here."
    )
    return Div(
        Div(H3("Answers Appear Here", cls="section-title"), P(placeholder, cls="section-copy"), cls="empty-state"),
        id="answer-panel",
        cls="answer-shell empty",
    )


def render_user_badge(user: Any) -> Any:
    """Render the authenticated user chip shown at the top of the page."""

    avatar = (
        Img(src=user["picture"], cls="user-avatar", alt=user["name"])
        if user["picture"]
        else Span(user["name"][:1].upper(), cls="user-avatar user-avatar-fallback")
    )
    return Div(
        avatar,
        Span(user["name"] or user["email"], cls="user-name"),
        A("Logout", href="/logout", cls="logout-link"),
        cls="user-badge",
    )


def render_login_page(session, error: str = "") -> Any:
    """Render the Google sign-in page for unauthenticated visitors."""

    state = begin_oauth_session(session)
    error_msg = ERROR_MESSAGES.get(error, "")
    return Title("TubeMind - Sign in"), Div(
        Div(
            H2("Welcome to TubeMind", cls="login-title"),
            P("Sign in with Google to get your own private YouTube research workspace.", cls="login-copy"),
            Div(error_msg, cls="login-error") if error_msg else "",
            A("Sign in with Google", href=google_auth_url(state), role="button", cls="signin-btn"),
            cls="login-card",
        ),
        cls="login-shell",
    )


def home_page(app_state: TubeMindApp, user: Any, notice: str = "") -> Any:
    """Render the primary authenticated TubeMind workspace."""

    status = app_state.status_payload()
    filters = status["youtube"].get("filters", {}) or {}
    preferred_channels_text = format_filter_text(list(filters.get("preferred_channels", []) or []))
    excluded_channels_text = format_filter_text(list(filters.get("excluded_channels", []) or []))
    saved_blacklist_text = format_filter_text(list(filters.get("global_excluded_channels", []) or []))
    preferred_only = bool(filters.get("preferred_only", False))
    return Div(
        Div(render_user_badge(user), cls="page-header"),
        Div(
            Div(
                Span("YouTube Corpus Q&A", cls="eyebrow"),
                H1("Turn a handful of YouTube videos into a searchable research brief.", cls="display"),
                P(
                    "TubeMind finds transcript-enabled videos, builds a lightweight knowledge base from them, and lets people ask grounded questions in everyday language.",
                    cls="lead",
                ),
                Div(Span("1. Search a topic", cls="step-chip"), Span("2. Index the transcripts", cls="step-chip"), Span("3. Ask grounded questions", cls="step-chip"), cls="step-row"),
            ),
            Div(
                Div(Span("Fastest path to a useful corpus", cls="status-pill ready"), P("Start with a topic that naturally maps to 5 to 8 long-form videos, not a single exact video title.", cls="section-copy"), cls="panel tight"),
                Div(
                    P("Use the live dashboard below to watch indexing progress, see what made it into the corpus, and understand skipped videos.", cls="section-copy"),
                    P("If some videos are skipped, TubeMind will tell you whether transcripts were missing or YouTube throttled access.", cls="section-copy"),
                    cls="panel tight",
                ),
            ),
            cls="hero hero-grid",
        ),
        render_dashboard(status, notice=notice),
        Div(
            Div(
                Div(
                    H3("Step 1: Build the Corpus", cls="section-title"),
                    P("Pick a topic, choose how YouTube should sort results, preview the candidate videos, and uncheck any results you do not want in this run.", cls="section-copy"),
                    Form(
                        Div(
                            Div(Label("Search Topic", cls="field-label"), Input(id="query-input", type="text", name="query", value=status["youtube"].get("seed_query", ""), placeholder="Example: machine learning for beginners"), P("Use a phrase close to what a real person would search on YouTube.", cls="field-help"), cls="field"),
                            Div(Label("Sort Results By", cls="field-label"), Select(*[Option(label, value=value, selected=(value == "relevance")) for value, label in SEARCH_ORDER_LABELS.items()], name="order"), P("Best Match is safest. Newest First is useful for recent topics.", cls="field-help"), cls="field"),
                            Div(Label("How Many Videos", cls="field-label"), Input(type="number", name="max_videos", value="8", min="1", max="15"), P("TubeMind will stop once it has enough successful transcript matches.", cls="field-help"), cls="field"),
                            Div(Label("Minimum Video Length (seconds)", cls="field-label"), Input(type="number", name="min_seconds", value="240", min="60", max="3600"), P("Higher values usually reduce short, low-signal clips.", cls="field-help"), cls="field"),
                            cls="field-grid",
                        ),
                        Div(
                            Div(
                                Label("Preferred Channels", cls="field-label"),
                                Textarea(preferred_channels_text, name="preferred_channels", placeholder="One channel per line, for example:\nIBM Technology\nFireship", rows=4),
                                P("TubeMind will prioritize these channels first. Use exact or close channel names.", cls="field-help"),
                                cls="field",
                            ),
                            Div(
                                Label("Excluded Channels", cls="field-label"),
                                Textarea(excluded_channels_text, name="excluded_channels", placeholder="Hide creators you do not want in just this run.", rows=4),
                                P("These exclusions apply only to the current corpus build. Your saved channel blacklist is applied automatically too.", cls="field-help"),
                                cls="field",
                            ),
                            cls="field-grid",
                        ),
                        Div(
                            Label(
                                Input(type="checkbox", name="preferred_only", value="true", checked=preferred_only),
                                Span("Only include preferred channels", cls="field-label"),
                                cls="toggle-row",
                            ),
                            P("Turn this on when you trust a fixed creator list. Leave it off to use those creators as a ranking preference instead.", cls="field-help"),
                            cls="field",
                        ),
                        Div(
                            Button("Preview Candidate Videos", type="button", cls="prompt-chip", _hx_post="/api/search_preview", _hx_target="#search-preview-panel", _hx_swap="outerHTML", _hx_include="closest form"),
                            Button(
                                "Start Fresh",
                                type="button",
                                cls="prompt-chip",
                                _hx_post="/api/reset_youtube",
                                _hx_target="#workspace-root",
                                _hx_swap="outerHTML",
                                _hx_confirm="Clear the current corpus, skipped items, and answers? Saved channel settings will be kept.",
                            ),
                            Button("Start Indexing", type="submit", cls="primary-btn"),
                            Span("TubeMind is building your corpus...", cls="htmx-indicator small"),
                            P("Preview first if you want per-video control. If you skip preview, TubeMind will use the highest-ranked transcript-enabled matches automatically.", cls="small"),
                            cls="action-row",
                        ),
                        render_search_preview_panel(),
                        _hx_post="/api/seed_youtube",
                        _hx_target="#dashboard-panels",
                        _hx_swap="outerHTML",
                    ),
                    cls="panel",
                ),
                render_channel_settings_panel(saved_blacklist_text),
                cls="panel-stack",
            ),
            Div(
                H3("Step 2: Ask Questions", cls="section-title"),
                P("Ask for a summary, compare viewpoints, pull out practical advice, or explain the topic in simpler language.", cls="section-copy"),
                Form(
                    Div(Label("Your Question", cls="field-label"), Textarea("", id="question-input", name="question", placeholder="Example: What are the most important machine learning concepts these videos agree on?", rows=6), P("Questions work best after at least a few videos have been indexed successfully.", cls="field-help"), cls="field"),
                    Div(Label("Answer Style", cls="field-label"), Select(*[Option(label, value=value, selected=(value == DEFAULT_QUERY_MODE)) for value, label in QUERY_MODE_LABELS.items()], name="mode"), P("Balanced is the best default. Focused Detail is useful for precise follow-ups.", cls="field-help"), cls="field"),
                    Div(Button("Ask TubeMind", type="submit", cls="primary-btn"), Span("Thinking through the indexed videos...", cls="htmx-indicator small"), cls="action-row"),
                    _hx_post="/api/query_youtube",
                    _hx_target="#answer-panel",
                    _hx_swap="outerHTML",
                ),
                P("Prompt ideas:", cls="field-label prompt-ideas-label"),
                Div(
                    *[
                        Button(
                            prompt,
                            type="button",
                            cls="prompt-chip",
                            onclick=f"document.getElementById('question-input').value = {json.dumps(prompt)}; document.getElementById('question-input').focus();",
                        )
                        for prompt in PROMPT_SUGGESTIONS
                    ],
                    cls="prompt-row",
                ),
                render_answer_panel(indexed=status["youtube"]["indexed"]),
                cls="panel",
            ),
            cls="workflow-grid",
        ),
        Div(
            H3("Developer Tools", cls="section-title"),
            P("The main screen is designed for people, but the JSON endpoints are still available for debugging and scripting.", cls="section-copy"),
            Pre("Status:\ncurl -s http://localhost:5001/api/status | python3 -m json.tool\n\nSearch preview:\ncurl -s 'http://localhost:5001/api/search_youtube?q=machine%20learning&order=relevance&minSeconds=240' | python3 -m json.tool", cls="mono-copy"),
            cls="panel dev-panel",
        ),
        cls="wrap",
        id="workspace-root",
    )
