from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

from dotenv import load_dotenv
from fasthtml.common import *


ROOT = Path(__file__).resolve().parent
APP_ROOT = ROOT / ".local" / "wiki_graph_app"
RAG_STORAGE_DIR = APP_ROOT / "rag_storage"
STATE_FILE = APP_ROOT / "state.json"
WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
USER_AGENT = "TubeMind/0.1 (https://github.com/mailf/TubeMind)"
INDEX_SEARCH_LIMIT = 5
DEFAULT_QUERY_MODE = "mix"
QUERY_MODES = ("mix", "hybrid", "local", "global", "naive")


def load_environment() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY was not found in .env")


@dataclass
class CorpusState:
    indexed: bool = False
    seed_query: str = ""
    indexed_page_ids: list[int] = field(default_factory=list)
    indexed_titles: list[str] = field(default_factory=list)
    page_urls: dict[str, str] = field(default_factory=dict)

    @classmethod
    def load(cls) -> "CorpusState":
        if not STATE_FILE.exists():
            return cls()
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return cls(
            indexed=bool(data.get("indexed", False)),
            seed_query=str(data.get("seed_query", "")),
            indexed_page_ids=[int(page_id) for page_id in data.get("indexed_page_ids", [])],
            indexed_titles=[str(title) for title in data.get("indexed_titles", [])],
            page_urls={str(k): str(v) for k, v in data.get("page_urls", {}).items()},
        )

    def save(self) -> None:
        APP_ROOT.mkdir(parents=True, exist_ok=True)
        payload = {
            "indexed": self.indexed,
            "seed_query": self.seed_query,
            "indexed_page_ids": self.indexed_page_ids,
            "indexed_titles": self.indexed_titles,
            "page_urls": self.page_urls,
        }
        STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class WikiGraphApp:
    def __init__(self) -> None:
        load_environment()
        APP_ROOT.mkdir(parents=True, exist_ok=True)
        self.state = CorpusState.load()
        self.lock = threading.RLock()
        self.rag = self._create_rag()

    def _create_rag(self):
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed

        llm_model = partial(
            openai_complete_if_cache,
            "gpt-5.1",
            reasoning_effort="none",
        )

        return LightRAG(
            working_dir=str(RAG_STORAGE_DIR),
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
                    "\n\n".join(
                        [
                            f"Title: {title}",
                            f"Source: {canonical_url}",
                            text,
                        ]
                    )
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
            self.state.save()

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


def wikipedia_request(params: dict[str, Any]) -> dict[str, Any]:
    query = urlencode({**params, "format": "json", "formatversion": "2"})
    request = UrlRequest(
        f"{WIKIPEDIA_API_URL}?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    with urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def wikipedia_search(topic: str, limit: int = INDEX_SEARCH_LIMIT) -> list[dict[str, Any]]:
    data = wikipedia_request(
        {
            "action": "query",
            "list": "search",
            "srsearch": topic,
            "srlimit": limit,
            "srprop": "",
        }
    )
    return data.get("query", {}).get("search", [])


def wikipedia_fetch_articles(page_ids: list[int]) -> list[dict[str, Any]]:
    data = wikipedia_request(
        {
            "action": "query",
            "prop": "extracts|info",
            "pageids": "|".join(str(page_id) for page_id in page_ids),
            "inprop": "url",
            "explaintext": 1,
            "redirects": 1,
        }
    )
    pages = data.get("query", {}).get("pages", [])
    return [
        page
        for page in pages
        if page.get("missing") is None and page.get("extract") and page.get("fullurl")
    ]


app_state = WikiGraphApp()

app, rt = fast_app(
    title="TubeMind Wikipedia GraphRAG",
    pico=True,
    hdrs=(
        Style(
            """
            .htmx-indicator {
                display: none;
            }
            .htmx-request.htmx-indicator,
            .htmx-request .htmx-indicator {
                display: block;
            }
            #response-area pre {
                white-space: pre-wrap;
            }
            """
        ),
    ),
    on_startup=[app_state.startup],
    on_shutdown=[app_state.shutdown],
)


def page_main(*content: Any):
    state = app_state.state
    return Main(
        Header(
            H1("TubeMind Wikipedia GraphRAG"),
            P(
                "Index Wikipedia once into LightRAG, then keep querying the existing graph and corpus without reindexing."
            ),
        ),
        status_panel(),
        Article(
            H2("Seed Corpus"),
            P(
                "This searches Wikipedia, indexes the returned articles into LightRAG, and answers the first question."
            ),
            Form(
                Label("Wikipedia topic or search phrase", _for="seed_query"),
                Textarea(
                    state.seed_query or "",
                    id="seed_query",
                    name="seed_query",
                    placeholder="Example: Ada Lovelace and the analytical engine",
                    disabled=state.indexed,
                ),
                Button("Index corpus and answer", type="submit", disabled=state.indexed),
                Small("Indexing is one-time. After completion, this form stays disabled."),
                hx_post="/seed",
                hx_target="main",
                hx_swap="outerHTML",
                hx_indicator="#seed-indicator",
            ),
            Article(
                P("Indexing in progress. The page will update when the corpus is ready.", aria_busy="true"),
                id="seed-indicator",
                cls="htmx-indicator",
            ),
        ),
        Article(
            H2("Query Existing Corpus"),
            P("These follow-up questions only query the saved LightRAG index."),
            Form(
                Label("Question", _for="query"),
                Textarea(
                    "",
                    id="query",
                    name="query",
                    placeholder="Ask about the indexed Wikipedia corpus",
                ),
                Label("Retrieval mode", _for="mode"),
                Select(
                    *[
                        Option(mode.upper(), value=mode, selected=mode == DEFAULT_QUERY_MODE)
                        for mode in QUERY_MODES
                    ],
                    id="mode",
                    name="mode",
                    disabled=not state.indexed,
                ),
                Button("Query graph", type="submit", disabled=not state.indexed),
                hx_post="/query",
                hx_target="main",
                hx_swap="outerHTML",
                hx_indicator="#query-indicator",
            ),
            Article(
                P("Query in progress. Results will replace the page content below.", aria_busy="true"),
                id="query-indicator",
                cls="htmx-indicator",
            ),
        ),
        Section(*content, id="response-area") if content else response_panel(),
        cls="container",
    )


def layout(*content: Any):
    return Title("TubeMind Wikipedia GraphRAG"), page_main(*content)


def status_panel() -> Any:
    state = app_state.state
    if state.indexed:
        return Article(
            Header(H2("Index Ready")),
            P("Indexing is complete. Future questions use the existing LightRAG corpus only."),
            Ul(
                Li(f"Seed query: {state.seed_query}"),
                Li(f"Stored articles: {len(state.indexed_titles)}"),
            ),
        )
    return Article(
        Header(H2("Index Not Built")),
        P("No Wikipedia corpus has been indexed yet. Use the seed form once to create it."),
    )


def details_block(titles: list[str]) -> Any:
    return Details(
        Summary("Indexed articles"),
        Ul(
            *[
                Li(A(title, href=app_state.state.page_urls.get(title, "#"), target="_blank"))
                for title in titles
            ]
        ),
        open=True,
    )


def response_panel(
    heading: str = "Awaiting Input",
    message: str = "Seed the corpus to begin. Once indexing is complete, this page will clearly show that the index is ready.",
    answer: str = "",
    titles: list[str] | None = None,
) -> Any:
    return Article(
        Header(H2(heading)),
        P(message),
        Pre(answer) if answer else "",
        details_block(titles or []) if titles else "",
    )


@rt("/")
def get(request: Request):
    if request.headers.get("HX-Request") == "true":
        return page_main()
    return layout()


@rt("/seed", methods=["POST"])
def post(request: Request, seed_query: str = ""):
    try:
        seed_result = app_state.seed_corpus(seed_query)
        answer = app_state.query_corpus(seed_query, mode=DEFAULT_QUERY_MODE)
        content = response_panel(
            "Index Complete",
            seed_result["message"] + " The corpus is now ready for follow-up questions without reindexing.",
            answer=answer,
            titles=seed_result["inserted_titles"] or app_state.state.indexed_titles,
        )
        if request.headers.get("HX-Request") == "true":
            return page_main(content)
        return layout(content)
    except Exception as exc:
        content = response_panel("Seed Corpus", str(exc))
        if request.headers.get("HX-Request") == "true":
            return page_main(content)
        return layout(content)


@rt("/query", methods=["POST"])
def post_query(request: Request, query: str = "", mode: str = DEFAULT_QUERY_MODE):
    try:
        answer = app_state.query_corpus(query, mode=mode)
        content = response_panel(
            "Follow-up Answer",
            f"Answered from the existing corpus with mode '{mode}'. No reindex was performed.",
            answer=answer,
            titles=app_state.state.indexed_titles,
        )
        if request.headers.get("HX-Request") == "true":
            return page_main(content)
        return layout(content)
    except Exception as exc:
        content = response_panel("Query Existing Corpus", str(exc))
        if request.headers.get("HX-Request") == "true":
            return page_main(content)
        return layout(content)


if __name__ == "__main__":
    serve()
