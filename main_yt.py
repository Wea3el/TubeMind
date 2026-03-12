from __future__ import annotations

import asyncio
import json
import os
import re
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode
from urllib.request import Request as UrlRequest, urlopen

import httpx
from dotenv import load_dotenv
from youtube_transcript_api import NoTranscriptFound, TooManyRequests, YouTubeRequestFailed, YouTubeTranscriptApi

from fasthtml.common import *

ROOT = Path(__file__).resolve().parent
APP_ROOT = ROOT / ".local" / "tubemind_app"
RAG_STORAGE_DIR = APP_ROOT / "rag_storage"
STATE_FILE = APP_ROOT / "state.json"

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

DEFAULT_QUERY_MODE = "mix"
QUERY_MODES = ("mix", "hybrid", "local", "global", "naive")
QUERY_MODE_LABELS = {
    "mix": "Balanced",
    "hybrid": "Deep Retrieval",
    "local": "Focused Detail",
    "global": "Big Picture",
    "naive": "Fast Draft",
}
SEARCH_ORDER_LABELS = {
    "relevance": "Best Match",
    "viewCount": "Most Viewed",
    "date": "Newest First",
}
PROMPT_SUGGESTIONS = (
    "Give me the main ideas across these videos.",
    "What are the most practical takeaways for a beginner?",
    "Compare the speakers' different opinions on this topic.",
    "Summarize the videos in plain English with examples.",
)

MIN_SECONDS_DEFAULT = 240
MAX_VIDEOS_DEFAULT = 8
JOB_POLL_SECONDS = 1.0
TRANSCRIPT_RETRY_ATTEMPTS = 3
TRANSCRIPT_RETRY_BASE_DELAY = 1.0


def load_environment() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY was not found in backend/.env")
    if not os.environ.get("OPENAI_MODEL"):
        os.environ["OPENAI_MODEL"] = "gpt-4o-mini"
    if not os.environ.get("YOUTUBE_API_KEY"):
        raise RuntimeError("YOUTUBE_API_KEY was not found in backend/.env")


def now_ms() -> int:
    return int(time.time() * 1000)


def iso8601_duration_to_seconds(d: str) -> int:
    # PT#H#M#S
    if not d or not isinstance(d, str) or not d.startswith("PT"):
        return 0
    m = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", d)
    if not m:
        return 0
    h = int(m.group(1) or 0)
    mm = int(m.group(2) or 0)
    s = int(m.group(3) or 0)
    return h * 3600 + mm * 60 + s


def seconds_to_label(sec: int) -> str:
    if sec <= 0:
        return ""
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def yt_watch_url(video_id: str, t: Optional[float] = None) -> str:
    if t is None:
        return f"https://www.youtube.com/watch?v={video_id}"
    return f"https://www.youtube.com/watch?v={video_id}&t={int(t)}s"


@dataclass
class YouTubeVideo:
    video_id: str
    title: str
    channel_title: str
    published_at: str
    thumbnail: str
    duration_sec: int
    url: str


@dataclass
class CorpusState:
    youtube_indexed: bool = False
    youtube_seed_query: str = ""
    youtube_video_ids: List[str] = field(default_factory=list)
    youtube_titles: List[str] = field(default_factory=list)
    youtube_urls: Dict[str, str] = field(default_factory=dict)

    # debug info
    youtube_skipped: List[Dict[str, Any]] = field(default_factory=list)

    # job status
    job_active: bool = False
    job_id: str = ""
    job_stage: str = ""
    job_progress: int = 0
    job_total: int = 0
    job_message: str = ""

    @classmethod
    def load(cls) -> "CorpusState":
        if not STATE_FILE.exists():
            return cls()
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return cls(
            youtube_indexed=bool(data.get("youtube_indexed", False)),
            youtube_seed_query=str(data.get("youtube_seed_query", "")),
            youtube_video_ids=[str(x) for x in data.get("youtube_video_ids", [])],
            youtube_titles=[str(x) for x in data.get("youtube_titles", [])],
            youtube_urls={str(k): str(v) for k, v in (data.get("youtube_urls", {}) or {}).items()},
            youtube_skipped=list(data.get("youtube_skipped", []) or []),
            job_active=bool(data.get("job_active", False)),
            job_id=str(data.get("job_id", "")),
            job_stage=str(data.get("job_stage", "")),
            job_progress=int(data.get("job_progress", 0)),
            job_total=int(data.get("job_total", 0)),
            job_message=str(data.get("job_message", "")),
        )

    def save(self) -> None:
        APP_ROOT.mkdir(parents=True, exist_ok=True)
        payload = {
            "youtube_indexed": self.youtube_indexed,
            "youtube_seed_query": self.youtube_seed_query,
            "youtube_video_ids": self.youtube_video_ids,
            "youtube_titles": self.youtube_titles,
            "youtube_urls": self.youtube_urls,
            "youtube_skipped": self.youtube_skipped,
            "job_active": self.job_active,
            "job_id": self.job_id,
            "job_stage": self.job_stage,
            "job_progress": self.job_progress,
            "job_total": self.job_total,
            "job_message": self.job_message,
        }
        STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class TubeMindApp:
    def __init__(self) -> None:
        load_environment()
        APP_ROOT.mkdir(parents=True, exist_ok=True)
        self.state = CorpusState.load()
        self.lock = threading.RLock()
        self.rag = self._create_rag()
        self._bg_thread: Optional[threading.Thread] = None

    def _create_rag(self):
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed

        model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        llm_model = partial(openai_complete_if_cache, model, reasoning_effort="none")

        return LightRAG(
            working_dir=str(RAG_STORAGE_DIR),
            llm_model_func=llm_model,
            embedding_func=openai_embed,
        )

    async def startup(self) -> None:
        await self.rag.initialize_storages()

    async def shutdown(self) -> None:
        await self.rag.finalize_storages()

    def reset_youtube_index(self) -> None:
        with self.lock:
            self.state.youtube_indexed = False
            self.state.youtube_seed_query = ""
            self.state.youtube_video_ids = []
            self.state.youtube_titles = []
            self.state.youtube_urls = {}
            self.state.youtube_skipped = []
            self.state.job_active = False
            self.state.job_id = ""
            self.state.job_stage = ""
            self.state.job_progress = 0
            self.state.job_total = 0
            self.state.job_message = ""
            self.state.save()

    def _set_job(self, *, active: bool, stage: str = "", progress: int = 0, total: int = 0, msg: str = "") -> None:
        self.state.job_active = active
        self.state.job_stage = stage
        self.state.job_progress = progress
        self.state.job_total = total
        self.state.job_message = msg
        self.state.save()

    async def youtube_search(self, query: str, *, max_videos: int, min_seconds: int, order: str) -> List[YouTubeVideo]:
        key = os.environ["YOUTUBE_API_KEY"]

        params = {
            "part": "snippet",
            "type": "video",
            "maxResults": str(min(max_videos, 25)),
            "q": query,
            "order": order,
            "key": key,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(YOUTUBE_SEARCH_URL, params=params)
            data = r.json()
            if r.status_code != 200:
                raise RuntimeError(f"YouTube search failed: {data}")

        items = data.get("items", [])
        video_ids = [it.get("id", {}).get("videoId") for it in items]
        video_ids = [vid for vid in video_ids if vid]
        if not video_ids:
            return []

        params2 = {
            "part": "snippet,contentDetails",
            "id": ",".join(video_ids),
            "key": key,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            r2 = await client.get(YOUTUBE_VIDEOS_URL, params=params2)
            data2 = r2.json()
            if r2.status_code != 200:
                raise RuntimeError(f"YouTube videos.list failed: {data2}")

        vids: List[YouTubeVideo] = []
        for v in data2.get("items", []):
            vid = str(v.get("id", ""))
            sn = v.get("snippet", {}) or {}
            cd = v.get("contentDetails", {}) or {}
            dur_iso = str(cd.get("duration", "") or "")
            dur = iso8601_duration_to_seconds(dur_iso)

            thumbs = sn.get("thumbnails", {}) or {}
            thumb = (
                (thumbs.get("medium") or {}).get("url")
                or (thumbs.get("high") or {}).get("url")
                or (thumbs.get("default") or {}).get("url")
                or ""
            )

            if dur < min_seconds:
                continue

            vids.append(
                YouTubeVideo(
                    video_id=vid,
                    title=str(sn.get("title", "")),
                    channel_title=str(sn.get("channelTitle", "")),
                    published_at=str(sn.get("publishedAt", "")),
                    thumbnail=thumb,
                    duration_sec=dur,
                    url=yt_watch_url(vid),
                )
            )

        return vids[:max_videos]

    def _transcript_request_kwargs(self) -> Dict[str, Any]:
        cookies_file = str(os.environ.get("YOUTUBE_TRANSCRIPT_COOKIES_FILE", "")).strip()
        if not cookies_file:
            return {}
        return {"cookies": cookies_file}

    def _is_transcript_rate_limited(self, exc: Exception) -> bool:
        if isinstance(exc, TooManyRequests):
            return True
        text = str(exc).lower()
        return "429" in text or "too many requests" in text

    def _should_retry_transcript_error(self, exc: Exception) -> bool:
        if self._is_transcript_rate_limited(exc):
            return True
        if isinstance(exc, YouTubeRequestFailed):
            text = str(exc).lower()
            return "timed out" in text or "temporarily unavailable" in text
        return False

    def _describe_transcript_error(self, exc: Exception, *, using_cookies: bool) -> str:
        msg = f"{type(exc).__name__}: {str(exc)}"
        if self._is_transcript_rate_limited(exc) and not using_cookies:
            return (
                f"{msg}\nHint: export a YouTube browser cookie file and set "
                "YOUTUBE_TRANSCRIPT_COOKIES_FILE to reduce transcript 429s."
            )
        return msg

    def _fetch_transcript(self, video_id: str) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        # Return (segments, error_string) with retries for flaky YouTube responses.
        last_err: Optional[str] = None
        request_kwargs = self._transcript_request_kwargs()

        for attempt in range(1, TRANSCRIPT_RETRY_ATTEMPTS + 1):
            try:
                try:
                    segs = YouTubeTranscriptApi.get_transcript(video_id, languages=("en",), **request_kwargs)
                except NoTranscriptFound:
                    segs = YouTubeTranscriptApi.get_transcript(video_id, **request_kwargs)

                if not segs:
                    last_err = "empty transcript payload"
                else:
                    return segs, None
            except Exception as exc:
                last_err = self._describe_transcript_error(exc, using_cookies=bool(request_kwargs.get("cookies")))
                if not self._should_retry_transcript_error(exc):
                    break

            if attempt < TRANSCRIPT_RETRY_ATTEMPTS:
                time.sleep(TRANSCRIPT_RETRY_BASE_DELAY * (2 ** (attempt - 1)))

        return None, last_err or "unknown transcript error"

    def _format_transcript(self, segs: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for s in segs:
            t = float(s.get("start", 0.0) or 0.0)
            txt = str(s.get("text", "")).replace("\n", " ").strip()
            if not txt:
                continue
            lines.append(f"[t={t:.1f}] {txt}")
        return "\n".join(lines)

    def start_youtube_index_job(self, query: str, *, max_videos: int, min_seconds: int, order: str) -> str:
        normalized = query.strip()
        if not normalized:
            raise ValueError("Enter a YouTube search phrase to index.")

        with self.lock:
            self.reset_youtube_index()

            job_id = f"job_{now_ms()}"
            self.state.job_id = job_id
            self.state.youtube_seed_query = normalized
            self._set_job(active=True, stage="search", progress=0, total=max_videos, msg="Searching YouTube...")
            self.state.save()

            def runner():
                try:
                    asyncio.run(self._run_youtube_index_job(job_id, normalized, max_videos, min_seconds, order))
                except Exception as e:
                    with self.lock:
                        self._set_job(active=False, stage="error", progress=0, total=0, msg=str(e))

            self._bg_thread = threading.Thread(target=runner, daemon=True)
            self._bg_thread.start()
            return job_id

    async def _run_youtube_index_job(self, job_id: str, query: str, max_videos: int, min_seconds: int, order: str) -> None:
        candidate_pool = max(max_videos * 3, max_videos + 6)
        videos = await self.youtube_search(query, max_videos=candidate_pool, min_seconds=min_seconds, order=order)

        with self.lock:
            if self.state.job_id != job_id:
                return
            self._set_job(
                active=True,
                stage="transcripts",
                progress=0,
                total=len(videos),
                msg=f"Fetching transcripts from {len(videos)} candidate videos...",
            )

        documents: List[str] = []
        ids: List[str] = []
        file_paths: List[str] = []

        for i, v in enumerate(videos, start=1):
            if len(documents) >= max_videos:
                break

            with self.lock:
                if self.state.job_id != job_id:
                    return
                self._set_job(
                    active=True,
                    stage="transcripts",
                    progress=i - 1,
                    total=len(videos),
                    msg=f"Transcript {i} of {len(videos)}",
                )

            segs, err = self._fetch_transcript(v.video_id)
            if not segs:
                with self.lock:
                    self.state.youtube_skipped.append(
                        {
                            "videoId": v.video_id,
                            "title": v.title,
                            "url": v.url,
                            "reason": err or "unknown",
                        }
                    )
                    self.state.save()
                continue

            transcript = self._format_transcript(segs)
            if not transcript.strip():
                with self.lock:
                    self.state.youtube_skipped.append(
                        {
                            "videoId": v.video_id,
                            "title": v.title,
                            "url": v.url,
                            "reason": "empty transcript",
                        }
                    )
                    self.state.save()
                continue

            doc = "\n\n".join(
                [
                    "Source: YouTube",
                    f"Title: {v.title}",
                    f"Channel: {v.channel_title}",
                    f"PublishedAt: {v.published_at}",
                    f"DurationSec: {v.duration_sec}",
                    f"VideoURL: {v.url}",
                    "",
                    "Transcript (timestamped, seconds):",
                    transcript,
                    "",
                    "Instruction: cite timestamps by writing VideoURL&t=SECONDS.",
                ]
            )

            documents.append(doc)
            ids.append(f"youtube:{v.video_id}")
            file_paths.append(v.url)

            with self.lock:
                self.state.youtube_video_ids.append(v.video_id)
                self.state.youtube_titles.append(v.title)
                self.state.youtube_urls[v.title] = v.url
                self.state.save()

        with self.lock:
            self._set_job(active=True, stage="index", progress=0, total=len(documents), msg="Indexing into LightRAG...")

        if documents:
            self.rag.insert(documents, ids=ids, file_paths=file_paths)

        with self.lock:
            indexed_count = len(documents)
            self.state.youtube_indexed = indexed_count > 0
            done_msg = f"Indexed {indexed_count} videos."
            if indexed_count == 0:
                done_msg += " (Most likely transcripts were disabled. Check skipped list.)"
            self._set_job(active=False, stage="done", progress=indexed_count, total=indexed_count, msg=done_msg)
            self.state.save()

    def query_youtube(self, question: str, mode: str = DEFAULT_QUERY_MODE) -> str:
        q = question.strip()
        if not q:
            raise ValueError("Enter a question.")
        if not self.state.youtube_indexed:
            raise ValueError("Index YouTube first.")
        if mode not in QUERY_MODES:
            mode = DEFAULT_QUERY_MODE

        from lightrag import QueryParam

        answer = str(
            self.rag.query(
                q,
                param=QueryParam(mode=mode, response_type="Multiple Paragraphs"),
            )
        ).strip()

        if (not answer) or (answer.lower() in ("none", "null")):
            raise RuntimeError("Empty answer. Try a simpler question or re-index different videos.")
        return answer

    def status_payload(self) -> Dict[str, Any]:
        s = self.state
        return {
            "youtube": {
                "indexed": s.youtube_indexed,
                "seed_query": s.youtube_seed_query,
                "count": len(s.youtube_titles),
                "titles": s.youtube_titles,
                "urls": s.youtube_urls,
            },
            "job": {
                "active": s.job_active,
                "id": s.job_id,
                "stage": s.job_stage,
                "progress": s.job_progress,
                "total": s.job_total,
                "message": s.job_message,
            },
            "skipped": s.youtube_skipped[-30:],  # last 30
        }


app_state = TubeMindApp()

app, rt = fast_app(
    title="TubeMind",
    pico=True,
    hdrs=(
        Link(rel="preconnect", href="https://fonts.googleapis.com"),
        Link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin=""),
        Link(
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Source+Sans+3:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap",
        ),
        Style(
            """
            :root {
                --pico-font-family: "Source Sans 3", sans-serif;
                --pico-font-size: 106%;
                --canvas: #f4eee4;
                --canvas-deep: #eadfce;
                --surface: rgba(255, 251, 245, 0.92);
                --surface-strong: #fffdf9;
                --surface-soft: #f8f1e7;
                --ink: #162033;
                --muted-ink: #667085;
                --line: rgba(22, 32, 51, 0.12);
                --line-strong: rgba(22, 32, 51, 0.18);
                --accent: #0f766e;
                --accent-strong: #115e59;
                --accent-soft: rgba(15, 118, 110, 0.12);
                --warm: #ea580c;
                --warm-soft: rgba(234, 88, 12, 0.12);
                --shadow: 0 18px 44px rgba(65, 42, 19, 0.08);
                --radius-lg: 24px;
                --radius-md: 18px;
                --radius-sm: 14px;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                color: var(--ink);
                background:
                    radial-gradient(circle at top left, rgba(15, 118, 110, 0.12), transparent 34%),
                    radial-gradient(circle at top right, rgba(234, 88, 12, 0.12), transparent 28%),
                    linear-gradient(180deg, #fbf7f0 0%, var(--canvas) 48%, var(--canvas-deep) 100%);
            }
            h1, h2, h3, h4, .display, .section-title {
                font-family: "Space Grotesk", sans-serif;
                letter-spacing: -0.03em;
            }
            p, label, li, a, button, input, select, textarea { color: var(--ink); }
            a { color: var(--accent-strong); }
            .wrap { max-width: 1180px; margin: 0 auto; padding: 34px 18px 56px; }
            .hero {
                background: linear-gradient(145deg, rgba(255, 255, 255, 0.7), rgba(255, 248, 239, 0.92));
                border: 1px solid rgba(255, 255, 255, 0.7);
                border-radius: 30px;
                box-shadow: var(--shadow);
                padding: 28px;
            }
            .hero-grid, .workflow-grid, .dashboard-grid, .metrics-grid, .field-grid {
                display: grid;
                gap: 18px;
            }
            .hero-grid { grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.7fr); align-items: end; }
            .workflow-grid, .dashboard-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 18px; }
            .metrics-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
            .field-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            .eyebrow {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                padding: 8px 12px;
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid rgba(22, 32, 51, 0.08);
                font-size: 0.84rem;
                font-weight: 700;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: var(--accent-strong);
            }
            .display {
                margin: 14px 0 10px;
                font-size: clamp(2.4rem, 5vw, 4.1rem);
                line-height: 0.98;
            }
            .lead {
                margin: 0;
                max-width: 58ch;
                font-size: 1.12rem;
                color: var(--muted-ink);
            }
            .step-row, .status-row, .prompt-row, .inline-meta {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                align-items: center;
            }
            .step-chip, .status-pill, .micro-pill {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                border-radius: 999px;
                padding: 9px 12px;
                font-weight: 700;
                font-size: 0.94rem;
            }
            .step-chip {
                background: rgba(255, 255, 255, 0.75);
                border: 1px solid rgba(22, 32, 51, 0.08);
            }
            .micro-pill {
                background: rgba(22, 32, 51, 0.06);
                border: 1px solid rgba(22, 32, 51, 0.08);
            }
            .status-pill {
                background: rgba(255, 255, 255, 0.86);
                border: 1px solid rgba(22, 32, 51, 0.08);
            }
            .status-pill.ready { color: var(--accent-strong); background: rgba(15, 118, 110, 0.12); }
            .status-pill.working { color: var(--accent-strong); background: rgba(15, 118, 110, 0.14); }
            .status-pill.warn { color: #9a3412; background: rgba(234, 88, 12, 0.12); }
            .status-pill.idle { color: var(--ink); background: rgba(22, 32, 51, 0.06); }
            .panel, .metric-card, .answer-shell {
                background: var(--surface);
                border: 1px solid rgba(255, 255, 255, 0.72);
                box-shadow: var(--shadow);
                border-radius: var(--radius-lg);
            }
            .panel { padding: 22px; }
            .panel.tight { padding: 18px; }
            .metric-card {
                padding: 16px;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.9), rgba(248, 241, 231, 0.88));
            }
            .metric-label {
                margin: 0 0 4px;
                color: var(--muted-ink);
                font-size: 0.92rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
            .metric-value {
                margin: 0;
                font-family: "Space Grotesk", sans-serif;
                font-size: 2rem;
                line-height: 1;
            }
            .metric-hint {
                margin: 8px 0 0;
                color: var(--muted-ink);
                font-size: 0.95rem;
            }
            .section-title { margin: 0; font-size: 1.45rem; }
            .section-copy, .muted { color: var(--muted-ink); }
            .section-copy { margin-top: 8px; }
            .field { display: grid; gap: 8px; }
            .field-label {
                font-weight: 700;
                margin: 0;
            }
            .field-help {
                margin: 0;
                font-size: 0.95rem;
                color: var(--muted-ink);
            }
            input, select, textarea {
                margin: 0;
                border-radius: 16px;
                border: 1px solid var(--line);
                background: rgba(255, 255, 255, 0.9);
                box-shadow: none;
            }
            input:focus, select:focus, textarea:focus {
                border-color: rgba(15, 118, 110, 0.55);
                box-shadow: 0 0 0 0.18rem rgba(15, 118, 110, 0.12);
            }
            textarea { min-height: 180px; }
            .action-row {
                display: flex;
                justify-content: space-between;
                align-items: center;
                gap: 12px;
                flex-wrap: wrap;
                margin-top: 18px;
            }
            .primary-btn, .prompt-chip {
                border: 0;
                box-shadow: none;
            }
            .primary-btn {
                background: linear-gradient(135deg, var(--accent), var(--accent-strong));
                color: white;
                font-weight: 700;
                padding-inline: 18px;
            }
            .secondary-btn {
                background: rgba(22, 32, 51, 0.08);
                color: var(--ink);
            }
            .prompt-chip {
                background: rgba(255, 255, 255, 0.94);
                color: var(--ink);
                border-radius: 999px;
                padding: 10px 14px;
                font-size: 0.94rem;
            }
            .prompt-chip:hover { background: #ffffff; }
            .progress-wrap { display: grid; gap: 10px; margin-top: 16px; }
            .progress-track {
                height: 12px;
                width: 100%;
                border-radius: 999px;
                background: rgba(22, 32, 51, 0.08);
                overflow: hidden;
            }
            .progress-fill {
                height: 100%;
                border-radius: 999px;
                background: linear-gradient(90deg, var(--warm), #fb923c);
            }
            .list-stack { display: grid; gap: 12px; margin-top: 16px; }
            .source-item, .skip-item, .empty-state {
                border-radius: var(--radius-sm);
                border: 1px solid var(--line);
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.72), rgba(248, 241, 231, 0.92));
                padding: 14px 16px;
            }
            .source-item { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 14px; align-items: start; }
            .source-num {
                display: inline-grid;
                place-items: center;
                width: 38px;
                height: 38px;
                border-radius: 12px;
                font-family: "Space Grotesk", sans-serif;
                background: rgba(15, 118, 110, 0.12);
                color: var(--accent-strong);
            }
            .item-title {
                margin: 0;
                font-size: 1.04rem;
                font-weight: 700;
            }
            .item-copy, .mono-copy, .tiny {
                margin: 4px 0 0;
                color: var(--muted-ink);
            }
            .mono-copy, pre {
                font-family: "IBM Plex Mono", monospace;
                font-size: 0.88rem;
            }
            .answer-shell {
                padding: 22px;
                margin-top: 16px;
                min-height: 240px;
            }
            .answer-shell.empty {
                display: grid;
                place-items: center;
                text-align: center;
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.78), rgba(248, 241, 231, 0.9));
            }
            .answer-shell.error {
                border-color: rgba(234, 88, 12, 0.28);
                background: linear-gradient(180deg, rgba(255, 244, 236, 0.92), rgba(255, 250, 246, 0.92));
            }
            .answer-pre {
                margin: 12px 0 0;
                white-space: pre-wrap;
                font-family: "Source Sans 3", sans-serif;
                font-size: 1.02rem;
                line-height: 1.7;
            }
            .small { font-size: 0.92rem; color: var(--muted-ink); }
            .tiny { font-size: 0.84rem; }
            .htmx-indicator { display:none; }
            .htmx-request .htmx-indicator { display:inline-flex; }
            .dev-panel {
                margin-top: 18px;
                border-style: dashed;
                background: rgba(255, 255, 255, 0.62);
            }
            @media (max-width: 960px) {
                .hero-grid, .workflow-grid, .dashboard-grid, .metrics-grid, .field-grid {
                    grid-template-columns: 1fr;
                }
                .wrap { padding: 22px 14px 40px; }
                .hero, .panel, .answer-shell { padding: 18px; }
                .display { font-size: clamp(2rem, 11vw, 3rem); }
                .source-item { grid-template-columns: 1fr; }
            }
            """
        ),
    ),
    on_startup=[app_state.startup],
    on_shutdown=[app_state.shutdown],
)


def truncate_text(text: str, limit: int = 180) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def friendly_job_stage(stage: str) -> str:
    return {
        "search": "Finding matching videos",
        "transcripts": "Collecting transcripts",
        "index": "Building the knowledge base",
        "done": "Ready for questions",
        "error": "Something needs attention",
    }.get(stage or "", "Waiting to start")


def progress_percent(progress: int, total: int) -> int:
    if total <= 0:
        return 0
    return max(0, min(100, round((progress / total) * 100)))


def status_badge(status: Dict[str, Any]) -> tuple[str, str]:
    if status["job"]["active"]:
        return "Indexing in progress", "working"
    if status["youtube"]["indexed"]:
        return "Ready to answer", "ready"
    if status["skipped"]:
        return "Needs attention", "warn"
    return "Waiting for a corpus", "idle"


def summarize_skip_reason(reason: str) -> str:
    raw = (reason or "").strip()
    lower = raw.lower()
    if "429" in lower or "too many requests" in lower:
        return "YouTube temporarily rate-limited transcript access for this video."
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
    return Div(
        P(label, cls="metric-label"),
        P(value, cls="metric-value"),
        P(hint, cls="metric-hint"),
        cls="metric-card",
    )


def render_dashboard(status: Dict[str, Any], notice: str = "") -> Any:
    badge_text, badge_tone = status_badge(status)
    youtube = status["youtube"]
    job = status["job"]
    pct = progress_percent(job["progress"], job["total"])
    indexed_titles = youtube["titles"]

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
            Div(
                Span("Skipped", cls="micro-pill"),
                P(item.get("title", "Untitled video"), cls="item-title"),
                cls="inline-meta",
            ),
            P(summarize_skip_reason(str(item.get("reason", ""))), cls="item-copy"),
            P(truncate_text(str(item.get("reason", "")), limit=220), cls="tiny"),
            cls="skip-item",
        )
        for item in reversed(status["skipped"][-6:])
    ]

    summary_children: List[Any] = [
        Span(badge_text, cls=f"status-pill {badge_tone}"),
        P(
            youtube["seed_query"] and f'Current topic: "{youtube["seed_query"]}"' or "No topic indexed yet",
            cls="section-copy",
        ),
        H2(friendly_job_stage(job["stage"]), cls="section-title"),
        P(
            job["message"] or "Search a topic, index a few transcript-enabled videos, then ask questions in plain English.",
            cls="section-copy",
        ),
    ]

    if notice:
        summary_children.append(
            Div(
                P(notice, cls="item-copy"),
                cls="skip-item",
            )
        )

    summary_children.extend(
        [
            Div(
                render_metric_card("Indexed Videos", str(youtube["count"]), "Videos ready for question answering."),
                render_metric_card("Skipped Videos", str(len(status["skipped"])), "Usually caused by missing captions or rate limits."),
                render_metric_card("Progress", f"{pct}%", f"{job['progress']} of {job['total']} current job steps completed." if job["total"] else "Waiting for the next indexing run."),
                cls="metrics-grid",
            ),
            Div(
                Div(Div(style=f"width:{pct}%;", cls="progress-fill"), cls="progress-track"),
                Div(
                    Span(friendly_job_stage(job["stage"]), cls="small"),
                    Span(job["message"] or "No active job", cls="small"),
                    cls="status-row",
                ),
                cls="progress-wrap",
            ),
        ]
    )

    return Div(
        Div(*summary_children, cls="panel"),
        Div(
            Div(
                H3("Indexed Video Library", cls="section-title"),
                P("Each successful transcript becomes part of the corpus that answers are drawn from.", cls="section-copy"),
                Div(*indexed_items, cls="list-stack") if indexed_items else Div(
                    P("Your indexed videos will appear here once TubeMind finishes building the corpus.", cls="item-copy"),
                    cls="empty-state",
                ),
                cls="panel",
            ),
            Div(
                H3("Skipped or Unavailable Videos", cls="section-title"),
                P("TubeMind only works with videos that expose transcripts. This panel helps explain what was left out.", cls="section-copy"),
                Div(*skipped_items, cls="list-stack") if skipped_items else Div(
                    P("No skipped videos so far. That usually means your current indexing run is healthy.", cls="item-copy"),
                    cls="empty-state",
                ),
                cls="panel",
            ),
            cls="dashboard-grid",
        ),
        id="dashboard-panels",
        _hx_get="/api/dashboard",
        _hx_trigger="load, every 2s",
        _hx_swap="outerHTML",
    )


def render_answer_panel(*, answer: str = "", error: str = "", indexed: bool = False) -> Any:
    if error:
        return Div(
            H3("Question Could Not Be Answered", cls="section-title"),
            P(error, cls="answer-pre"),
            id="answer-panel",
            cls="answer-shell error",
        )
    if answer:
        return Div(
            H3("Answer", cls="section-title"),
            P("Generated from the currently indexed YouTube transcript corpus.", cls="section-copy"),
            Pre(answer, cls="answer-pre"),
            id="answer-panel",
            cls="answer-shell",
        )
    placeholder = (
        "Ask about themes, disagreements, examples, takeaways, or summaries once your indexing run is ready."
        if indexed
        else "Index a topic first, then ask natural-language questions about the videos here."
    )
    return Div(
        Div(
            H3("Answers Appear Here", cls="section-title"),
            P(placeholder, cls="section-copy"),
            cls="empty-state",
        ),
        id="answer-panel",
        cls="answer-shell empty",
    )


def home_page(msg: str = "", answer: str = "") -> Any:
    s = app_state.status_payload()
    return Div(
        Div(
            Div(
                Span("YouTube Corpus Q&A", cls="eyebrow"),
                H1("Turn a handful of YouTube videos into a searchable research brief.", cls="display"),
                P(
                    "TubeMind finds transcript-enabled videos, builds a lightweight knowledge base from them, and lets people ask grounded questions in everyday language.",
                    cls="lead",
                ),
                Div(
                    Span("1. Search a topic", cls="step-chip"),
                    Span("2. Index the transcripts", cls="step-chip"),
                    Span("3. Ask grounded questions", cls="step-chip"),
                    cls="step-row",
                ),
            ),
            Div(
                Div(
                    Span(status_badge(s)[0], cls=f"status-pill {status_badge(s)[1]}"),
                    P(
                        s["youtube"]["seed_query"] and f'Current topic: "{s["youtube"]["seed_query"]}"' or "Choose a topic to get started.",
                        cls="section-copy",
                    ),
                    cls="panel tight",
                ),
                Div(
                    P("A good first run uses 5 to 8 videos and a minimum length of 4 to 8 minutes.", cls="section-copy"),
                    P("If some videos are skipped, TubeMind will explain whether captions were missing or YouTube throttled transcript access.", cls="section-copy"),
                    cls="panel tight",
                ),
            ),
            cls="hero hero-grid",
        ),
        render_dashboard(s),
        Div(
            Div(
                H3("Step 1: Build the Corpus", cls="section-title"),
                P("Pick a topic, choose how YouTube should sort results, and set how many videos TubeMind should try to ingest.", cls="section-copy"),
                Form(
                    Div(
                        Div(
                            Label("Search Topic", cls="field-label"),
                            Input(id="query-input", type="text", name="query", placeholder="Example: machine learning for beginners"),
                            P("Use a phrase close to what a real person would search on YouTube.", cls="field-help"),
                            cls="field",
                        ),
                        Div(
                            Label("Sort Results By", cls="field-label"),
                            Select(
                                *[
                                    Option(label, value=value, selected=(value == "relevance"))
                                    for value, label in SEARCH_ORDER_LABELS.items()
                                ],
                                name="order",
                            ),
                            P("Best Match is safest. Newest First is useful for recent topics.", cls="field-help"),
                            cls="field",
                        ),
                        Div(
                            Label("How Many Videos", cls="field-label"),
                            Input(type="number", name="max_videos", value="8", min="1", max="15"),
                            P("TubeMind will stop once it has enough successful transcript matches.", cls="field-help"),
                            cls="field",
                        ),
                        Div(
                            Label("Minimum Video Length (seconds)", cls="field-label"),
                            Input(type="number", name="min_seconds", value="240", min="60", max="3600"),
                            P("Higher values usually reduce short, low-signal clips.", cls="field-help"),
                            cls="field",
                        ),
                        cls="field-grid",
                    ),
                    Div(
                        Button("Start Indexing", type="submit", cls="primary-btn"),
                        Span("TubeMind is building your corpus...", cls="htmx-indicator small"),
                        P("You can keep watching the live dashboard while indexing runs.", cls="small"),
                        cls="action-row",
                    ),
                    _hx_post="/api/seed_youtube",
                    _hx_target="#dashboard-panels",
                    _hx_swap="outerHTML",
                ),
                cls="panel",
            ),
            Div(
                H3("Step 2: Ask Questions", cls="section-title"),
                P("Ask for a summary, compare viewpoints, pull out practical advice, or explain the topic in simpler language.", cls="section-copy"),
                Form(
                    Div(
                        Label("Your Question", cls="field-label"),
                        Textarea(
                            "",
                            id="question-input",
                            name="question",
                            placeholder="Example: What are the most important machine learning concepts these videos agree on?",
                            rows=6,
                        ),
                        P("Questions work best after at least a few videos have been indexed successfully.", cls="field-help"),
                        cls="field",
                    ),
                    Div(
                        Label("Answer Style", cls="field-label"),
                        Select(
                            *[
                                Option(label, value=value, selected=(value == DEFAULT_QUERY_MODE))
                                for value, label in QUERY_MODE_LABELS.items()
                            ],
                            name="mode",
                        ),
                        P("Balanced is the best default. Focused Detail is useful for precise follow-ups.", cls="field-help"),
                        cls="field",
                    ),
                    Div(
                        Button("Ask TubeMind", type="submit", cls="primary-btn"),
                        Span("Thinking through the indexed videos...", cls="htmx-indicator small"),
                        cls="action-row",
                    ),
                    _hx_post="/api/query_youtube",
                    _hx_target="#answer-panel",
                    _hx_swap="outerHTML",
                ),
                P("Prompt ideas:", cls="field-label", style="margin-top:18px;"),
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
                render_answer_panel(indexed=s["youtube"]["indexed"]),
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
    )


@rt("/")
def get_root(request: Request):
    return home_page()


@rt("/api/status")
def api_status(request: Request):
    return app_state.status_payload()


@rt("/api/dashboard")
def api_dashboard(request: Request):
    return render_dashboard(app_state.status_payload())


@rt("/api/search_youtube")
async def api_search_youtube(request: Request, q: str = "", order: str = "relevance", minSeconds: str = "240", maxResults: str = "12"):
    try:
        ms = int(minSeconds) if str(minSeconds).isdigit() else 240
        mr = int(maxResults) if str(maxResults).isdigit() else 12
        mr = max(1, min(25, mr))
        vids = await app_state.youtube_search(q.strip(), max_videos=mr, min_seconds=ms, order=order)
        return {
            "query": q,
            "order": order,
            "minSeconds": ms,
            "results": [
                {
                    "videoId": v.video_id,
                    "title": v.title,
                    "channelTitle": v.channel_title,
                    "publishedAt": v.published_at,
                    "durationSec": v.duration_sec,
                    "durationLabel": seconds_to_label(v.duration_sec),
                    "url": v.url,
                    "thumbnail": v.thumbnail,
                }
                for v in vids
            ],
        }
    except Exception as exc:
        return {"error": str(exc), "query": q, "results": []}


@rt("/api/seed_youtube", methods=["POST"])
def api_seed_youtube(request: Request, query: str = "", order: str = "relevance", max_videos: str = "8", min_seconds: str = "240"):
    mv = int(max_videos) if str(max_videos).isdigit() else MAX_VIDEOS_DEFAULT
    ms = int(min_seconds) if str(min_seconds).isdigit() else MIN_SECONDS_DEFAULT
    mv = max(1, min(15, mv))
    ms = max(60, min(3600, ms))

    is_htmx = request.headers.get("hx-request", "").lower() == "true"

    try:
        job_id = app_state.start_youtube_index_job(query, max_videos=mv, min_seconds=ms, order=order)
    except ValueError as exc:
        if is_htmx:
            return render_dashboard(app_state.status_payload(), notice=str(exc))
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)

    s = app_state.status_payload()
    if is_htmx:
        return render_dashboard(s)
    return {
        "ok": True,
        "job_id": job_id,
        **s,
    }


@rt("/api/query_youtube", methods=["POST"])
def api_query_youtube(request: Request, question: str = "", mode: str = DEFAULT_QUERY_MODE):
    try:
        ans = app_state.query_youtube(question, mode=mode)
        if request.headers.get("hx-request", "").lower() == "true":
            return render_answer_panel(answer=ans, indexed=app_state.state.youtube_indexed)
        return {"ok": True, "answer": ans}
    except Exception as exc:
        if request.headers.get("hx-request", "").lower() == "true":
            return render_answer_panel(error=str(exc), indexed=app_state.state.youtube_indexed)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


if __name__ == "__main__":
    serve(host="0.0.0.0", port=5001)
