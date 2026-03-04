from __future__ import annotations

import asyncio
import json
import os
import secrets
import threading
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

from dotenv import load_dotenv
from fasthtml.common import *
from monsterui.all import *


ROOT = Path(__file__).resolve().parent
APP_ROOT = ROOT / ".local"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TubeMind/0.1 (https://github.com/mailf/TubeMind)"
INDEX_SEARCH_LIMIT = 5
DEFAULT_QUERY_MODE = "mix"
QUERY_MODES = ("mix", "hybrid", "local", "global", "naive")


# ── environment ───────────────────────────────────────────────────────────────

def load_environment() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY was not found in .env")


load_environment()

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)
BASE_URL = os.environ.get("BASE_URL", "http://localhost:5001")
REDIRECT_URI = f"{BASE_URL}/auth/callback"


# ── database ──────────────────────────────────────────────────────────────────

APP_ROOT.mkdir(parents=True, exist_ok=True)
db = database(str(APP_ROOT / "tubemind.db"))
users_table = db.t.users
if users_table not in db.t:
    users_table.create(dict(id=str, email=str, name=str, picture=str), pk="id")


# ── corpus state ──────────────────────────────────────────────────────────────

@dataclass
class CorpusState:
    indexed: bool = False
    seed_query: str = ""
    indexed_page_ids: list[int] = field(default_factory=list)
    indexed_titles: list[str] = field(default_factory=list)
    page_urls: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls, state_file: Path) -> "CorpusState":
        if not state_file.exists():
            return cls()
        data = json.loads(state_file.read_text(encoding="utf-8"))
        return cls(
            indexed=bool(data.get("indexed", False)),
            seed_query=str(data.get("seed_query", "")),
            indexed_page_ids=[int(p) for p in data.get("indexed_page_ids", [])],
            indexed_titles=[str(t) for t in data.get("indexed_titles", [])],
            page_urls={str(k): str(v) for k, v in data.get("page_urls", {}).items()},
        )

    def save(self, state_file: Path) -> None:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "indexed": self.indexed,
            "seed_query": self.seed_query,
            "indexed_page_ids": self.indexed_page_ids,
            "indexed_titles": self.indexed_titles,
            "page_urls": self.page_urls,
        }
        state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ── per-user RAG app ──────────────────────────────────────────────────────────

class WikiGraphApp:
    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.user_dir = APP_ROOT / "users" / user_id
        self.rag_dir = self.user_dir / "rag_storage"
        self.state_file = self.user_dir / "state.json"
        self.user_dir.mkdir(parents=True, exist_ok=True)
        self.state = CorpusState.load(self.state_file)
        self.lock = threading.RLock()
        self.rag = self._create_rag()

    def _create_rag(self):
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed

        llm_model = partial(
            openai_complete_if_cache,
            "gpt-4o-mini",
            reasoning_effort="low",
        )
        return LightRAG(
            working_dir=str(self.rag_dir),
            llm_model_func=llm_model,
            embedding_func=openai_embed,
        )

    async def startup(self) -> None:
        await self.rag.initialize_storages()

    async def shutdown(self) -> None:
        await self.rag.finalize_storages()

    def seed_corpus(self, topic: str) -> dict[str, Any]:
        normalized_topic = topic.strip()
        if not normalized_topic:
            raise ValueError("Enter a topic to search Wikipedia.")

        with self.lock:
            if self.state.indexed:
                return {
                    "inserted_titles": [],
                    "skipped_titles": list(self.state.indexed_titles),
                    "message": "The corpus is already indexed. Use the follow-up query form below.",
                }

            search_results = wikipedia_search(normalized_topic, limit=INDEX_SEARCH_LIMIT)
            if not search_results:
                raise ValueError("Wikipedia returned no matching articles for that topic.")

            page_ids = [result["pageid"] for result in search_results]
            articles = wikipedia_fetch_articles(page_ids)
            if not articles:
                raise ValueError("Wikipedia search worked, but article content could not be fetched.")

            documents: list[str] = []
            ids: list[str] = []
            file_paths: list[str] = []
            inserted_titles: list[str] = []

            for article in articles:
                page_id = int(article["pageid"])
                title = article["title"]
                text = article["extract"].strip()
                canonical_url = article["fullurl"]
                if not text or page_id in self.state.indexed_page_ids:
                    continue

                documents.append(
                    "\n\n".join([f"Title: {title}", f"Source: {canonical_url}", text])
                )
                ids.append(f"wikipedia:{page_id}")
                file_paths.append(canonical_url)
                inserted_titles.append(title)
                self.state.indexed_page_ids.append(page_id)
                self.state.indexed_titles.append(title)
                self.state.page_urls[title] = canonical_url

            if not documents:
                raise ValueError("No new Wikipedia article text was available to index.")

            self.rag.insert(documents, ids=ids, file_paths=file_paths)
            self.state.indexed = True
            self.state.seed_query = normalized_topic
            self.state.save(self.state_file)

            return {
                "inserted_titles": inserted_titles,
                "skipped_titles": [],
                "message": f"Indexed {len(inserted_titles)} Wikipedia articles for '{normalized_topic}'.",
            }

    def query_corpus(self, query: str, mode: str = DEFAULT_QUERY_MODE) -> str:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Enter a question for the indexed corpus.")
        if not self.state.indexed:
            raise ValueError("Index the Wikipedia corpus first.")
        if mode not in QUERY_MODES:
            mode = DEFAULT_QUERY_MODE

        with self.lock:
            from lightrag import QueryParam

            return str(
                self.rag.query(
                    normalized_query,
                    param=QueryParam(mode=mode, response_type="Multiple Paragraphs"),
                )
            )


# ── user app registry ─────────────────────────────────────────────────────────

_user_apps: dict[str, WikiGraphApp] = {}
_user_locks: dict[str, asyncio.Lock] = {}


async def get_user_app(user_id: str) -> WikiGraphApp:
    if user_id in _user_apps:
        return _user_apps[user_id]
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    async with _user_locks[user_id]:
        if user_id not in _user_apps:
            instance = WikiGraphApp(user_id)
            await instance.startup()
            _user_apps[user_id] = instance
    return _user_apps[user_id]


# ── Google OAuth helpers ──────────────────────────────────────────────────────

def google_auth_url(state: str) -> str:
    params = urlencode({
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "access_type": "online",
    })
    return f"https://accounts.google.com/o/oauth2/v2/auth?{params}"


def google_exchange_code(code: str) -> dict:
    data = urlencode({
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": REDIRECT_URI,
        "grant_type": "authorization_code",
    }).encode()
    req = UrlRequest(
        "https://oauth2.googleapis.com/token",
        data=data,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urlopen(req) as resp:
        return json.loads(resp.read())


def google_userinfo(access_token: str) -> dict:
    req = UrlRequest(
        "https://www.googleapis.com/oauth2/v2/userinfo",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    with urlopen(req) as resp:
        return json.loads(resp.read())


# ── app setup ─────────────────────────────────────────────────────────────────

app, rt = fast_app(
    title="TubeMind Wikipedia GraphRAG",
    pico=False,
    secret_key=SESSION_SECRET,
    hdrs=(
        *Theme.orange.headers(
            mode="light",
            radii=ThemeRadii.lg,
            shadows=ThemeShadows.md,
            font=ThemeFont.default,
        ),
    ),
)


# ── session helper ────────────────────────────────────────────────────────────

def current_user(session) -> Any | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    try:
        return users_table[user_id]
    except Exception:
        return None


# ── auth routes ───────────────────────────────────────────────────────────────

_ERROR_MESSAGES = {
    "no_code": "Google did not return an authorization code.",
    "bad_state": "Security check failed. Please try again.",
    "oauth_failed": "Could not complete sign-in with Google. Please try again.",
}


@rt("/login")
def get(session, error: str = ""):
    user = current_user(session)
    if user:
        return RedirectResponse("/", status_code=303)
    state = secrets.token_urlsafe(16)
    session["oauth_state"] = state
    error_msg = _ERROR_MESSAGES.get(error, "")
    return (
        Title("TubeMind – Sign in"),
        Main(
            Section(
                Container(
                    Card(
                        DivVStacked(
                            H2("Welcome to TubeMind", cls="text-center"),
                            P(
                                "Sign in with Google to get your own private Wikipedia GraphRAG workspace.",
                                cls=(TextPresets.muted_sm, "text-center"),
                            ),
                            (
                                Div(
                                    UkIcon("alert-circle", cls="size-4 shrink-0"),
                                    Span(error_msg),
                                    cls="flex gap-2 items-center text-sm text-red-600 bg-red-50 border border-red-200 rounded-lg px-3 py-2",
                                )
                                if error_msg
                                else ""
                            ),
                            A(
                                DivHStacked(
                                    UkIcon("log-in", cls="size-5"),
                                    Span("Sign in with Google"),
                                    cls="gap-2 justify-center",
                                ),
                                href=google_auth_url(state),
                                cls=(ButtonT.primary, "w-full"),
                            ),
                            cls="space-y-5 items-stretch",
                        ),
                        cls="max-w-sm mx-auto mt-24 shadow-lg",
                    ),
                ),
                cls=(SectionT.muted, SectionT.lg),
            ),
        ),
    )


@rt("/auth/callback")
def get(request: Request, session, code: str = "", state: str = ""):
    if not code:
        return RedirectResponse("/login?error=no_code", status_code=303)
    if state != session.get("oauth_state"):
        return RedirectResponse("/login?error=bad_state", status_code=303)
    session.pop("oauth_state", None)

    try:
        token_data = google_exchange_code(code)
        info = google_userinfo(token_data["access_token"])
    except Exception:
        return RedirectResponse("/login?error=oauth_failed", status_code=303)

    user_id = str(info["id"])
    try:
        users_table[user_id]
    except Exception:
        users_table.insert(
            dict(
                id=user_id,
                email=info.get("email", ""),
                name=info.get("name", ""),
                picture=info.get("picture", ""),
            )
        )

    session["user_id"] = user_id
    return RedirectResponse("/", status_code=303)


@rt("/logout")
def get(session):
    session.clear()
    return RedirectResponse("/login", status_code=303)


# ── UI helpers ────────────────────────────────────────────────────────────────

def user_badge(user) -> Any:
    avatar = (
        Img(src=user["picture"], cls="size-8 rounded-full", alt=user["name"])
        if user["picture"]
        else Div(UkIcon("user", cls="size-4"), cls="size-8 rounded-full bg-base-200 flex items-center justify-center")
    )
    return DivHStacked(
        avatar,
        Small(user["name"] or user["email"], cls=TextT.muted),
        A("Logout", href="/logout", cls=(AT.muted, "text-sm")),
        cls="gap-2 items-center",
    )


def page_main(user: Any, app_instance: WikiGraphApp, *content: Any):
    state = app_instance.state
    seed_button_label = "Index corpus and answer" if not state.indexed else "Corpus already indexed"
    return Main(
        Section(
            Container(
                DivVStacked(
                    DivFullySpaced(
                        DivVStacked(
                            H1("TubeMind", cls="mb-0"),
                            Small("Wikipedia GraphRAG workspace", cls=TextT.muted),
                            cls="items-start space-y-1",
                        ),
                        DivHStacked(
                            Label(
                                "Corpus ready" if state.indexed else "Awaiting seed",
                                cls=LabelT.primary if state.indexed else LabelT.secondary,
                            ),
                            Label(f"{len(state.indexed_titles)} articles", cls=LabelT.secondary),
                            user_badge(user),
                            cls="gap-3 flex-wrap items-center",
                        ),
                        cls="gap-4 flex-wrap",
                    ),
                    Grid(
                        DivVStacked(
                            Grid(
                                Card(
                                    Form(
                                        LabelTextArea(
                                            "Wikipedia topic or search phrase",
                                            value=state.seed_query or "",
                                            id="seed_query",
                                            placeholder="Example: Ada Lovelace and the analytical engine",
                                            disabled=state.indexed,
                                            input_cls="min-h-40",
                                        ),
                                        LoaderButton(
                                            seed_button_label,
                                            type="submit",
                                            disabled=state.indexed,
                                            cls=(ButtonT.primary, ButtonT.lg, "w-full justify-center"),
                                            hx_post="/seed",
                                            hx_target="main",
                                            hx_swap="outerHTML",
                                        ),
                                    ),
                                    DivVStacked(
                                        (
                                            Accordion(
                                                AccordionItem(
                                                    f"Indexed titles ({len(state.indexed_titles)})",
                                                    Ul(
                                                        *[
                                                            Li(
                                                                A(
                                                                    title,
                                                                    href=state.page_urls.get(title, "#"),
                                                                    target="_blank",
                                                                    cls=AT.muted,
                                                                )
                                                            )
                                                            for title in state.indexed_titles
                                                        ],
                                                        cls=ListT.divider,
                                                    ),
                                                    open=False,
                                                ),
                                            )
                                            if state.indexed_titles
                                            else P(
                                                "Titles will appear here after the corpus is indexed.",
                                                cls=TextPresets.muted_sm,
                                            )
                                        ),
                                        cls="items-start space-y-3",
                                    ),
                                    header=DivVStacked(
                                        DivHStacked(
                                            UkIcon("database", cls="size-5"),
                                            CardTitle("Seed corpus"),
                                            cls="gap-2",
                                        ),
                                        Small(
                                            "Search Wikipedia, index the top articles, and answer the seed topic from the new graph.",
                                            cls=TextT.muted,
                                        ),
                                        cls="items-start space-y-1",
                                    ),
                                    footer=DivFullySpaced(
                                        Small("One-time ingestion for the active corpus.", cls=TextT.muted),
                                        Label(
                                            "Locked after index" if state.indexed else "Ready to index",
                                            cls=LabelT.secondary if state.indexed else LabelT.primary,
                                        ),
                                    ),
                                    cls=CardT.primary,
                                ),
                                Card(
                                    Form(
                                        LabelTextArea(
                                            "Question",
                                            id="query",
                                            placeholder="Ask about the indexed Wikipedia corpus",
                                            input_cls="min-h-40",
                                        ),
                                        LabelSelect(
                                            *[
                                                Option(
                                                    mode.upper(),
                                                    value=mode,
                                                    selected=mode == DEFAULT_QUERY_MODE,
                                                )
                                                for mode in QUERY_MODES
                                            ],
                                            label="Retrieval mode",
                                            id="mode",
                                            disabled=not state.indexed,
                                        ),
                                        LoaderButton(
                                            "Query graph",
                                            type="submit",
                                            disabled=not state.indexed,
                                            cls=(ButtonT.secondary, ButtonT.lg, "w-full justify-center"),
                                            hx_post="/query",
                                            hx_target="main",
                                            hx_swap="outerHTML",
                                        ),
                                    ),
                                    header=DivVStacked(
                                        DivHStacked(
                                            UkIcon("sparkles", cls="size-5"),
                                            CardTitle("Ask the graph"),
                                            cls="gap-2",
                                        ),
                                        Small(
                                            "Run repeated analysis against the stored graph without triggering a new ingestion pass.",
                                            cls=TextT.muted,
                                        ),
                                        cls="items-start space-y-1",
                                    ),
                                    footer=DivFullySpaced(
                                        Small("Default mode: ", CodeSpan("mix"), cls=TextT.muted),
                                        Label(
                                            "Corpus required",
                                            cls=LabelT.primary if state.indexed else LabelT.secondary,
                                        ),
                                    ),
                                    cls=CardT.secondary,
                                ),
                                cols_lg=2,
                                cols_min=1,
                                cls="gap-6",
                            ),
                            *content if content else (response_panel(app_instance),),
                            cls="space-y-6 lg:col-span-2 w-full items-stretch",
                        ),
                        cols_min=1,
                        cols_lg=3,
                        cls="gap-6 items-start",
                    ),
                    cls="space-y-6 w-full items-stretch",
                ),
                cls="max-w-7xl",
            ),
            cls=(SectionT.muted, SectionT.lg),
        ),
    )


def layout(user: Any, app_instance: WikiGraphApp, *content: Any):
    return Title("TubeMind Wikipedia GraphRAG"), page_main(user, app_instance, *content)


def details_block(app_instance: WikiGraphApp, titles: list[str]) -> Any:
    return Accordion(
        AccordionItem(
            f"Indexed articles ({len(titles)})",
            Ul(
                *[
                    Li(
                        A(
                            title,
                            href=app_instance.state.page_urls.get(title, "#"),
                            target="_blank",
                            cls=AT.primary,
                        )
                    )
                    for title in titles
                ],
                cls=ListT.divider,
            ),
            open=False,
        ),
        cls="mt-2",
    )


def response_panel(
    app_instance: WikiGraphApp,
    heading: str = "Awaiting Input",
    message: str = "Seed the corpus to begin. Once indexing is complete, this page will clearly show that the index is ready.",
    answer: str = "",
    titles: list[str] | None = None,
) -> Any:
    state = app_instance.state
    body = (
        Div(answer, cls="whitespace-pre-wrap leading-7 text-sm")
        if answer
        else Div(
            P(
                "Indexing is complete. Every follow-up question now reuses the existing LightRAG corpus."
                if state.indexed
                else "Responses will appear here after you index a corpus or run a follow-up query.",
                cls=TextPresets.muted_sm,
            ),
            cls="rounded-box border border-dashed p-6",
        )
    )
    return Card(
        body,
        details_block(app_instance, titles or []) if titles else "",
        header=DivFullySpaced(
            DivVStacked(
                CardTitle(heading),
                P(message, cls=TextPresets.muted_sm),
                cls="items-start space-y-1",
            ),
            Label(
                "Ready" if state.indexed else "Idle",
                cls=LabelT.primary if state.indexed else LabelT.secondary,
            ),
            cls="gap-4 flex-wrap",
        ),
        id="response-area",
        cls="w-full",
        body_cls="space-y-4",
    )


# ── main routes ───────────────────────────────────────────────────────────────

@rt("/")
async def get(request: Request, session):
    user = current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    app_instance = await get_user_app(user["id"])
    if request.headers.get("HX-Request") == "true":
        return page_main(user, app_instance)
    return layout(user, app_instance)


@rt("/seed", methods=["POST"])
async def post(request: Request, session, seed_query: str = ""):
    user = current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    app_instance = await get_user_app(user["id"])
    try:
        seed_result = app_instance.seed_corpus(seed_query)
        answer = app_instance.query_corpus(seed_query, mode=DEFAULT_QUERY_MODE)
        content = response_panel(
            app_instance,
            "Index Complete",
            seed_result["message"] + " The corpus is now ready for follow-up questions without reindexing.",
            answer=answer,
            titles=seed_result["inserted_titles"] or app_instance.state.indexed_titles,
        )
    except Exception as exc:
        content = response_panel(app_instance, "Seed Corpus", str(exc))
    if request.headers.get("HX-Request") == "true":
        return page_main(user, app_instance, content)
    return layout(user, app_instance, content)


@rt("/query", methods=["POST"])
async def post_query(request: Request, session, query: str = "", mode: str = DEFAULT_QUERY_MODE):
    user = current_user(session)
    if not user:
        return RedirectResponse("/login", status_code=303)
    app_instance = await get_user_app(user["id"])
    try:
        answer = app_instance.query_corpus(query, mode=mode)
        content = response_panel(
            app_instance,
            "Follow-up Answer",
            f"Answered from the existing corpus with mode '{mode}'. No reindex was performed.",
            answer=answer,
            titles=app_instance.state.indexed_titles,
        )
    except Exception as exc:
        content = response_panel(app_instance, "Query Existing Corpus", str(exc))
    if request.headers.get("HX-Request") == "true":
        return page_main(user, app_instance, content)
    return layout(user, app_instance, content)


# ── Wikipedia helpers ─────────────────────────────────────────────────────────

def wikipedia_request(params: dict[str, Any]) -> dict[str, Any]:
    query = urlencode({**params, "format": "json", "formatversion": "2"})
    request = UrlRequest(
        f"{WIKIPEDIA_API_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wikipedia_search(topic: str, limit: int = INDEX_SEARCH_LIMIT) -> list[dict[str, Any]]:
    data = wikipedia_request({
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": limit,
        "srprop": "",
    })
    return data.get("query", {}).get("search", [])


def wikipedia_fetch_articles(page_ids: list[int]) -> list[dict[str, Any]]:
    data = wikipedia_request({
        "action": "query",
        "prop": "extracts|info",
        "pageids": "|".join(str(p) for p in page_ids),
        "inprop": "url",
        "explaintext": 1,
        "redirects": 1,
    })
    pages = data.get("query", {}).get("pages", [])
    return [
        p for p in pages
        if p.get("missing") is None and p.get("extract") and p.get("fullurl")
    ]


if __name__ == "__main__":
    serve()
