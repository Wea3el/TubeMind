from __future__ import annotations

import asyncio
import concurrent.futures
import html
import json
import os
import re
import tempfile
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
TRANSCRIPTAPI_BASE_URL = "https://transcriptapi.com/api/v2"

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
COOKIE_BROWSER_LABELS = {
    "chrome": "Google Chrome",
    "brave": "Brave",
    "safari": "Safari",
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
MAX_RECOMMENDATIONS = 5
TRANSCRIPT_RETRY_ATTEMPTS = 3
TRANSCRIPT_RETRY_BASE_DELAY = 1.0
TRANSCRIPT_REQUEST_DELAY_SECONDS = 1.5
TRANSCRIPT_CANDIDATE_PADDING = 2


def load_environment() -> None:
    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY was not found in backend/.env")
    if not os.environ.get("OPENAI_MODEL"):
        os.environ["OPENAI_MODEL"] = "gpt-4.1-nano"
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
    youtube_recommendations: List[Dict[str, Any]] = field(default_factory=list)

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
            youtube_recommendations=list(data.get("youtube_recommendations", []) or []),
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
            "youtube_recommendations": self.youtube_recommendations,
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
        self._rag_loop = asyncio.new_event_loop()
        self._rag_loop_ready = threading.Event()
        self._rag_thread: Optional[threading.Thread] = None
        self.rag = None
        self._start_rag_runtime()
        self.rag = self._run_coro_on_rag_loop_sync(self._create_rag())
        self._bg_thread: Optional[threading.Thread] = None
        self._repair_state_after_restart()

    def _start_rag_runtime(self) -> None:
        self._rag_thread = threading.Thread(target=self._run_rag_loop, name="tubemind-rag-loop", daemon=True)
        self._rag_thread.start()
        if not self._rag_loop_ready.wait(timeout=5):
            raise RuntimeError("TubeMind could not start the LightRAG worker loop.")

    def _run_rag_loop(self) -> None:
        asyncio.set_event_loop(self._rag_loop)
        self._rag_loop_ready.set()
        try:
            self._rag_loop.run_forever()
        finally:
            pending = asyncio.all_tasks(self._rag_loop)
            for task in pending:
                task.cancel()
            if pending:
                self._rag_loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            self._rag_loop.run_until_complete(self._rag_loop.shutdown_asyncgens())
            self._rag_loop.run_until_complete(self._rag_loop.shutdown_default_executor())
            self._rag_loop.close()

    def _submit_coro_to_rag_loop(self, coro) -> concurrent.futures.Future[Any]:
        if not self._rag_thread or not self._rag_thread.is_alive():
            raise RuntimeError("TubeMind knowledge base worker is not running.")
        return asyncio.run_coroutine_threadsafe(coro, self._rag_loop)

    async def _run_coro_on_rag_loop(self, coro):
        try:
            future = self._submit_coro_to_rag_loop(coro)
        except Exception:
            coro.close()
            raise
        return await asyncio.wrap_future(future)

    def _run_coro_on_rag_loop_sync(self, coro):
        try:
            future = self._submit_coro_to_rag_loop(coro)
        except Exception:
            coro.close()
            raise
        return future.result()

    async def _create_rag(self):
        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed

        model = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")
        llm_model = partial(openai_complete_if_cache, model)

        return LightRAG(
            working_dir=str(RAG_STORAGE_DIR),
            llm_model_func=llm_model,
            embedding_func=openai_embed,
        )

    async def startup(self) -> None:
        await self._run_coro_on_rag_loop(self.rag.initialize_storages())
        await self._repair_rag_backed_state()

    async def shutdown(self) -> None:
        try:
            await self._run_coro_on_rag_loop(self.rag.finalize_storages())
        finally:
            if self._rag_thread and self._rag_thread.is_alive():
                self._rag_loop.call_soon_threadsafe(self._rag_loop.stop)
                self._rag_thread.join(timeout=5)

    def _repair_state_after_restart(self) -> None:
        changed = False

        if self.state.job_active:
            self.state.job_active = False
            self.state.job_stage = "interrupted"
            self.state.job_message = "Previous indexing run was interrupted. Start indexing again."
            changed = True

        if not self.state.youtube_indexed and (
            self.state.youtube_video_ids or self.state.youtube_titles or self.state.youtube_urls
        ):
            self.state.youtube_video_ids = []
            self.state.youtube_titles = []
            self.state.youtube_urls = {}
            changed = True

        if changed:
            self.state.save()

    def _youtube_video_id_from_doc_id(self, doc_id: str) -> str:
        if not doc_id.startswith("youtube:"):
            return ""
        return doc_id.split(":", 1)[1].strip()

    def _extract_title_from_summary(self, summary: str) -> str:
        match = re.search(r"(?m)^Title:\s*(.+)$", summary or "")
        return match.group(1).strip() if match else ""

    def _merge_skipped_items(self, existing: List[Dict[str, Any]], incoming: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for item in [*existing, *incoming]:
            key = (
                str(item.get("videoId") or "").strip()
                or str(item.get("url") or "").strip()
                or str(item.get("title") or "").strip()
            )
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)

        return merged[-30:]

    def _classify_doc_status_docs(
        self,
        docs: Dict[str, Any],
        video_lookup: Optional[Dict[str, YouTubeVideo]] = None,
    ) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        from lightrag.base import DocStatus

        successful: List[Dict[str, str]] = []
        failed: List[Dict[str, str]] = []
        video_lookup = video_lookup or {}
        video_lookup_by_url = {video.url: video for video in video_lookup.values()}
        order = {video_id: idx for idx, video_id in enumerate(video_lookup.keys())}

        for doc_id, status_doc in docs.items():
            video_id = self._youtube_video_id_from_doc_id(doc_id)
            file_path = str(getattr(status_doc, "file_path", "") or "")
            video = video_lookup.get(video_id)
            if not video and file_path:
                video = video_lookup_by_url.get(file_path)
                if video and not video_id:
                    video_id = video.video_id

            title = (
                video.title
                if video
                else self._extract_title_from_summary(str(getattr(status_doc, "content_summary", "") or ""))
            )
            title = title or file_path or doc_id
            url = file_path or (video.url if video else yt_watch_url(video_id) if video_id else "")
            thumbnail = video.thumbnail if video else ""

            item = {
                "videoId": video_id,
                "title": title,
                "url": url,
                "thumbnail": thumbnail,
            }

            status = getattr(status_doc, "status", None)
            if status == DocStatus.PROCESSED:
                successful.append(item)
            elif status == DocStatus.FAILED:
                failed.append(
                    {
                        **item,
                        "reason": f"Indexing failed: {str(getattr(status_doc, 'error_msg', '') or 'unknown error')}",
                    }
                )

        if order:
            successful.sort(key=lambda item: order.get(item.get("videoId", ""), len(order)))
            failed.sort(key=lambda item: order.get(item.get("videoId", ""), len(order)))

        return successful, failed

    async def _get_docs_by_status(self, status) -> Dict[str, Any]:
        return await self._run_coro_on_rag_loop(self.rag.doc_status.get_docs_by_status(status))

    async def _get_docs_by_track_id(self, track_id: str) -> Dict[str, Any]:
        return await self._run_coro_on_rag_loop(self.rag.doc_status.get_docs_by_track_id(track_id))

    async def _repair_rag_backed_state(self) -> None:
        from lightrag.base import DocStatus

        processed_docs = await self._get_docs_by_status(DocStatus.PROCESSED)
        failed_docs = await self._get_docs_by_status(DocStatus.FAILED)
        processing_docs = await self._get_docs_by_status(DocStatus.PROCESSING)
        pending_docs = await self._get_docs_by_status(DocStatus.PENDING)

        should_resume_queue = bool(processing_docs or pending_docs)
        should_retry_failed_docs = (
            bool(failed_docs)
            and any(
                "reasoning_effort" in str(getattr(doc, "error_msg", "") or "").lower()
                for doc in failed_docs.values()
            )
        )

        if should_resume_queue or should_retry_failed_docs:
            with self.lock:
                self._set_job(
                    active=True,
                    stage="index",
                    progress=0,
                    total=len(processing_docs) + len(pending_docs) + len(failed_docs),
                    msg="Repairing stored transcripts into a usable corpus...",
                )
            await self._run_coro_on_rag_loop(self.rag.apipeline_process_enqueue_documents())
            processed_docs = await self._get_docs_by_status(DocStatus.PROCESSED)
            failed_docs = await self._get_docs_by_status(DocStatus.FAILED)

        successful, failed = self._classify_doc_status_docs({**processed_docs, **failed_docs})
        rag_video_ids = {item["videoId"] for item in [*successful, *failed] if item.get("videoId")}

        with self.lock:
            prior_state_video_ids = set(self.state.youtube_video_ids)
            self.state.youtube_indexed = bool(successful)
            self.state.youtube_video_ids = [item["videoId"] for item in successful if item.get("videoId")]
            self.state.youtube_titles = [item["title"] for item in successful]
            self.state.youtube_urls = {
                item["title"]: item["url"]
                for item in successful
                if item.get("title") and item.get("url")
            }

            if prior_state_video_ids and not prior_state_video_ids.issubset(rag_video_ids):
                self.state.youtube_recommendations = []

            non_index_failures = [
                item
                for item in self.state.youtube_skipped
                if not str(item.get("reason", "")).startswith("Indexing failed:")
            ]
            self.state.youtube_skipped = self._merge_skipped_items(non_index_failures, failed)

            self.state.job_active = False
            self.state.job_progress = len(successful)
            self.state.job_total = len(successful)

            if successful:
                self.state.job_stage = "done"
                if should_resume_queue or should_retry_failed_docs:
                    self.state.job_message = f"Recovered {len(successful)} indexed video(s) from stored transcripts."
                else:
                    self.state.job_message = f"Indexed {len(successful)} videos."
            elif failed:
                self.state.job_stage = "error"
                if should_resume_queue or should_retry_failed_docs:
                    self.state.job_message = "Stored transcripts were found, but rebuilding the corpus still failed. Check skipped videos for the latest error."
                else:
                    self.state.job_message = "Stored transcripts exist, but the corpus is not ready yet. Start indexing again to rebuild it."

            if self.state.job_id == "job_test":
                self.state.job_id = ""

            self.state.save()

    def reset_youtube_index(self) -> None:
        with self.lock:
            self.state.youtube_indexed = False
            self.state.youtube_seed_query = ""
            self.state.youtube_video_ids = []
            self.state.youtube_titles = []
            self.state.youtube_urls = {}
            self.state.youtube_recommendations = []
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

    def _transcript_candidate_pool(self, max_videos: int) -> int:
        raw_pad = str(os.environ.get("YOUTUBE_TRANSCRIPT_CANDIDATE_PADDING", TRANSCRIPT_CANDIDATE_PADDING))
        try:
            pad = int(raw_pad)
        except ValueError:
            pad = TRANSCRIPT_CANDIDATE_PADDING
        pad = max(0, min(8, pad))
        return min(25, max_videos + pad)

    def _transcript_request_delay(self) -> float:
        raw = str(os.environ.get("YOUTUBE_TRANSCRIPT_REQUEST_DELAY_SECONDS", TRANSCRIPT_REQUEST_DELAY_SECONDS))
        try:
            delay = float(raw)
        except ValueError:
            delay = TRANSCRIPT_REQUEST_DELAY_SECONDS
        return max(0.0, min(10.0, delay))

    def _serialize_recommendation(self, video: YouTubeVideo) -> Dict[str, Any]:
        return {
            "videoId": video.video_id,
            "title": video.title,
            "channelTitle": video.channel_title,
            "durationSec": video.duration_sec,
            "durationLabel": seconds_to_label(video.duration_sec),
            "thumbnail": video.thumbnail,
            "url": video.url,
        }

    def _save_recommendations(self, videos: List[YouTubeVideo]) -> None:
        self.state.youtube_recommendations = [
            self._serialize_recommendation(video)
            for video in videos[:MAX_RECOMMENDATIONS]
        ]
        self.state.save()

    def _transcript_request_kwargs(self) -> Dict[str, Any]:
        cookies_file = str(os.environ.get("YOUTUBE_TRANSCRIPT_COOKIES_FILE", "")).strip()
        if not cookies_file:
            return {}
        return {"cookies": cookies_file}

    def _transcript_api_key(self) -> str:
        return str(os.environ.get("TRANSCRIPTAPI_API_KEY", "")).strip()

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

    def _looks_rate_limited(self, reason: str) -> bool:
        lower = (reason or "").lower()
        return "429" in lower or "too many requests" in lower or "rate-limit" in lower

    def _yt_dlp_cookie_sources(self) -> List[tuple[str, Dict[str, Any]]]:
        sources: List[tuple[str, Dict[str, Any]]] = [("yt-dlp", {})]

        cookie_file = str(os.environ.get("YOUTUBE_TRANSCRIPT_COOKIES_FILE", "")).strip()
        if cookie_file:
            sources.append(("yt-dlp + cookie file", {"cookiefile": cookie_file}))

        browsers_raw = str(os.environ.get("YOUTUBE_COOKIES_BROWSER", "")).strip()
        for browser in [b.strip().lower() for b in browsers_raw.split(",") if b.strip()]:
            sources.append(
                (
                    f"yt-dlp + {COOKIE_BROWSER_LABELS.get(browser, browser.title())} cookies",
                    {"cookiesfrombrowser": (browser, None, None, None)},
                )
            )

        return sources

    def _extract_transcriptapi_error(self, payload: Any) -> str:
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, dict):
            return str(detail.get("message") or detail.get("reason") or json.dumps(detail))
        if detail:
            return str(detail)
        if isinstance(payload, dict) and payload.get("code"):
            return str(payload["code"])
        return "unknown TranscriptAPI error"

    def _fetch_transcript_with_transcriptapi(self, video: YouTubeVideo) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        api_key = self._transcript_api_key()
        if not api_key:
            return None, None

        params = {
            "video_url": video.url,
            "format": "json",
            "include_timestamp": "true",
            "send_metadata": "false",
        }
        headers = {"Authorization": f"Bearer {api_key}"}
        retry_delay = 1.0
        last_err = "unknown TranscriptAPI error"

        with httpx.Client(timeout=45) as client:
            for _attempt in range(3):
                resp = client.get(f"{TRANSCRIPTAPI_BASE_URL}/youtube/transcript", params=params, headers=headers)
                if resp.status_code == 200:
                    payload = resp.json()
                    transcript = payload.get("transcript") or []
                    segs = [
                        {
                            "start": float(item.get("start", 0.0) or 0.0),
                            "text": str(item.get("text", "")).strip(),
                        }
                        for item in transcript
                        if str(item.get("text", "")).strip()
                    ]
                    if segs:
                        return segs, None
                    last_err = "TranscriptAPI returned an empty transcript"
                    break

                try:
                    payload = resp.json()
                except Exception:
                    payload = {"detail": resp.text}
                last_err = self._extract_transcriptapi_error(payload)

                if resp.status_code in (408, 429, 503):
                    retry_after = resp.headers.get("Retry-After")
                    try:
                        retry_delay = float(retry_after) if retry_after else retry_delay
                    except ValueError:
                        retry_delay = retry_delay
                    time.sleep(max(1.0, retry_delay))
                    retry_delay = min(retry_delay * 2, 10.0)
                    continue
                break

        return None, f"TranscriptAPI: {last_err}"

    def _parse_seconds_label(self, value: str) -> float:
        clean = value.strip().replace(",", ".")
        parts = clean.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(clean)

    def _parse_vtt_segments(self, text: str) -> List[Dict[str, Any]]:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segs: List[Dict[str, Any]] = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if "-->" not in line:
                i += 1
                continue

            start_raw = line.split("-->", 1)[0].strip().split(" ")[0]
            i += 1
            cue_lines: List[str] = []
            while i < len(lines) and lines[i].strip():
                cue_lines.append(lines[i].strip())
                i += 1

            cue_text = re.sub(r"<[^>]+>", "", " ".join(cue_lines)).strip()
            cue_text = html.unescape(cue_text)
            if cue_text:
                segs.append({"start": self._parse_seconds_label(start_raw), "text": cue_text})
        return segs

    def _parse_json3_segments(self, text: str) -> List[Dict[str, Any]]:
        payload = json.loads(text)
        segs: List[Dict[str, Any]] = []
        for event in payload.get("events", []) or []:
            parts = event.get("segs") or []
            if not parts:
                continue
            cue_text = html.unescape("".join(str(part.get("utf8", "")) for part in parts)).replace("\n", " ").strip()
            if not cue_text:
                continue
            segs.append({"start": float(event.get("tStartMs", 0) or 0) / 1000.0, "text": cue_text})
        return segs

    def _read_subtitle_segments(self, path: Path) -> List[Dict[str, Any]]:
        text = path.read_text(encoding="utf-8")
        suffix = path.suffix.lower()
        if suffix == ".json3":
            return self._parse_json3_segments(text)
        if suffix == ".vtt":
            return self._parse_vtt_segments(text)
        raise RuntimeError(f"Unsupported subtitle format: {path.name}")

    def _fetch_transcript_with_ytdlp(self, video: YouTubeVideo) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        try:
            from yt_dlp import DownloadError, YoutubeDL
        except Exception as exc:
            return None, f"yt-dlp fallback unavailable: {type(exc).__name__}: {exc}"

        last_err: Optional[str] = None
        for source_label, extra_opts in self._yt_dlp_cookie_sources():
            try:
                with tempfile.TemporaryDirectory(prefix="tubemind_subs_") as tmpdir:
                    outtmpl = str(Path(tmpdir) / "%(id)s.%(ext)s")
                    opts = {
                        "skip_download": True,
                        "quiet": True,
                        "no_warnings": True,
                        "writesubtitles": True,
                        "writeautomaticsub": True,
                        "subtitleslangs": ["en", "en.*"],
                        "subtitlesformat": "json3/vtt/best",
                        "outtmpl": {"default": outtmpl, "subtitle": outtmpl},
                    } | extra_opts

                    with YoutubeDL(opts) as ydl:
                        ydl.download([video.url])

                    subtitle_files = sorted(Path(tmpdir).glob(f"{video.video_id}*.json3"))
                    subtitle_files.extend(sorted(Path(tmpdir).glob(f"{video.video_id}*.vtt")))
                    if not subtitle_files:
                        last_err = f"{source_label} did not produce a subtitle file"
                        continue

                    segs = self._read_subtitle_segments(subtitle_files[0])
                    if segs:
                        return segs, None
                    last_err = f"{source_label} produced an empty subtitle file"
            except Exception as exc:
                label = source_label
                if extra_opts.get("cookiesfrombrowser"):
                    browser = extra_opts["cookiesfrombrowser"][0]
                    label = f"{source_label} ({COOKIE_BROWSER_LABELS.get(browser, browser.title())})"
                if isinstance(exc, DownloadError):
                    last_err = f"{label}: {exc}"
                else:
                    last_err = f"{label}: {type(exc).__name__}: {exc}"

        return None, last_err

    def _fetch_transcript(self, video: YouTubeVideo) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        # Return (segments, error_string) with retries for flaky YouTube responses.
        last_err: Optional[str] = None
        request_kwargs = self._transcript_request_kwargs()

        for attempt in range(1, TRANSCRIPT_RETRY_ATTEMPTS + 1):
            try:
                try:
                    segs = YouTubeTranscriptApi.get_transcript(video.video_id, languages=("en",), **request_kwargs)
                except NoTranscriptFound:
                    segs = YouTubeTranscriptApi.get_transcript(video.video_id, **request_kwargs)

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

        transcript_api_segs, transcript_api_err = self._fetch_transcript_with_transcriptapi(video)
        if transcript_api_segs:
            return transcript_api_segs, None

        yt_dlp_segs, yt_dlp_err = self._fetch_transcript_with_ytdlp(video)
        if yt_dlp_segs:
            return yt_dlp_segs, None

        fallback_errors = [err for err in (transcript_api_err, yt_dlp_err) if err]
        if fallback_errors:
            last_err = "\n".join([last_err or "transcript fetch failed", *fallback_errors])

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
                        self.state.youtube_indexed = False
                        self.state.youtube_video_ids = []
                        self.state.youtube_titles = []
                        self.state.youtube_urls = {}
                        self._set_job(active=False, stage="error", progress=0, total=0, msg=str(e))

            self._bg_thread = threading.Thread(target=runner, daemon=True)
            self._bg_thread.start()
            return job_id

    async def _run_youtube_index_job(self, job_id: str, query: str, max_videos: int, min_seconds: int, order: str) -> None:
        candidate_pool = self._transcript_candidate_pool(max_videos)
        videos = await self.youtube_search(query, max_videos=candidate_pool, min_seconds=min_seconds, order=order)

        with self.lock:
            if self.state.job_id != job_id:
                return
            self._save_recommendations(videos)
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
        indexed_videos: List[YouTubeVideo] = []
        consecutive_rate_limits = 0

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

            segs, err = self._fetch_transcript(v)
            if not segs:
                if err and self._looks_rate_limited(err):
                    consecutive_rate_limits += 1
                else:
                    consecutive_rate_limits = 0
                with self.lock:
                    self.state.youtube_skipped.append(
                        {
                            "videoId": v.video_id,
                            "title": v.title,
                            "thumbnail": v.thumbnail,
                            "url": v.url,
                            "reason": err or "unknown",
                        }
                    )
                    self.state.save()
                if consecutive_rate_limits >= 2 and not documents:
                    with self.lock:
                        self._set_job(
                            active=False,
                            stage="done",
                            progress=i,
                            total=len(videos),
                            msg="YouTube rate-limited transcript access. Recommendations are still available on the right.",
                        )
                    break
                time.sleep(self._transcript_request_delay())
                continue

            consecutive_rate_limits = 0
            transcript = self._format_transcript(segs)
            if not transcript.strip():
                with self.lock:
                    self.state.youtube_skipped.append(
                        {
                            "videoId": v.video_id,
                            "title": v.title,
                            "thumbnail": v.thumbnail,
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
            indexed_videos.append(v)

            time.sleep(self._transcript_request_delay())

        with self.lock:
            self._set_job(active=True, stage="index", progress=0, total=len(documents), msg="Indexing into LightRAG...")

        successful_videos: List[Dict[str, str]] = [
            {
                "videoId": video.video_id,
                "title": video.title,
                "url": video.url,
                "thumbnail": video.thumbnail,
            }
            for video in indexed_videos
        ]
        processing_failures: List[Dict[str, str]] = []

        if documents:
            track_id = await self._run_coro_on_rag_loop(self.rag.ainsert(documents, ids=ids, file_paths=file_paths))
            track_docs = await self._get_docs_by_track_id(track_id)
            successful_videos, processing_failures = self._classify_doc_status_docs(
                track_docs,
                {video.video_id: video for video in indexed_videos},
            )

        with self.lock:
            self.state.youtube_skipped = self._merge_skipped_items(self.state.youtube_skipped, processing_failures)
            indexed_count = len(successful_videos)
            self.state.youtube_indexed = indexed_count > 0
            self.state.youtube_video_ids = [video["videoId"] for video in successful_videos if video.get("videoId")]
            self.state.youtube_titles = [video["title"] for video in successful_videos]
            self.state.youtube_urls = {
                video["title"]: video["url"]
                for video in successful_videos
                if video.get("title") and video.get("url")
            }
            done_msg = f"Indexed {indexed_count} videos."
            if indexed_count == 0:
                if processing_failures:
                    done_msg = "Transcripts were fetched, but building the corpus failed. Check skipped videos for the exact OpenAI or indexing error."
                elif any(self._looks_rate_limited(str(item.get("reason", ""))) for item in self.state.youtube_skipped):
                    done_msg += " YouTube rate-limited transcript access for this IP. Try fewer videos or configure cookies in .env."
                else:
                    done_msg += " (Most likely transcripts were disabled. Check skipped list.)"
            final_stage = "done" if indexed_count > 0 else "error"
            self._set_job(active=False, stage=final_stage, progress=indexed_count, total=indexed_count, msg=done_msg)
            self.state.save()

    async def query_youtube(self, question: str, mode: str = DEFAULT_QUERY_MODE) -> str:
        q = question.strip()
        if not q:
            raise ValueError("Enter a question.")
        if not self.state.youtube_indexed:
            raise ValueError("Index YouTube first. If the previous run failed, start indexing again to finalize the corpus.")
        if mode not in QUERY_MODES:
            mode = DEFAULT_QUERY_MODE

        from lightrag import QueryParam

        answer = str(
            await self._run_coro_on_rag_loop(
                self.rag.aquery(
                    q,
                    param=QueryParam(mode=mode, response_type="Multiple Paragraphs"),
                )
            )
        ).strip()

        if (not answer) or (answer.lower() in ("none", "null")):
            raise RuntimeError("Empty answer. Try a simpler question or re-index different videos.")
        return answer

    def status_payload(self) -> Dict[str, Any]:
        s = self.state
        indexed_titles = s.youtube_titles if s.youtube_indexed else []
        indexed_urls = s.youtube_urls if s.youtube_indexed else {}
        return {
            "youtube": {
                "indexed": s.youtube_indexed,
                "seed_query": s.youtube_seed_query,
                "count": len(indexed_titles),
                "titles": indexed_titles,
                "urls": indexed_urls,
                "recommendations": s.youtube_recommendations,
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
            .hero-grid, .workflow-grid, .dashboard-grid, .dashboard-body, .dashboard-main, .metrics-grid, .field-grid {
                display: grid;
                gap: 18px;
            }
            .hero-grid { grid-template-columns: minmax(0, 1.3fr) minmax(280px, 0.7fr); align-items: end; }
            .workflow-grid, .dashboard-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); margin-top: 18px; }
            .dashboard-body { grid-template-columns: minmax(0, 1.35fr) minmax(280px, 0.65fr); margin-top: 18px; }
            .dashboard-main { grid-template-columns: repeat(2, minmax(0, 1fr)); }
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
            .recommend-card {
                display: grid;
                grid-template-columns: 96px minmax(0, 1fr);
                gap: 12px;
                padding: 12px;
                border-radius: var(--radius-sm);
                border: 1px solid var(--line);
                background: linear-gradient(180deg, rgba(255, 255, 255, 0.82), rgba(248, 241, 231, 0.94));
                text-decoration: none;
                color: inherit;
                transition: transform 0.16s ease, box-shadow 0.16s ease;
            }
            .recommend-card:hover {
                transform: translateY(-1px);
                box-shadow: 0 12px 22px rgba(65, 42, 19, 0.08);
            }
            .recommend-thumb {
                width: 96px;
                height: 72px;
                object-fit: cover;
                border-radius: 12px;
                background: rgba(22, 32, 51, 0.08);
            }
            .recommend-meta {
                min-width: 0;
                display: grid;
                gap: 6px;
            }
            .recommend-stack {
                display: grid;
                gap: 12px;
                margin-top: 16px;
            }
            .recommend-line {
                display: flex;
                gap: 8px;
                flex-wrap: wrap;
                align-items: center;
            }
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
                .hero-grid, .workflow-grid, .dashboard-grid, .dashboard-body, .dashboard-main, .metrics-grid, .field-grid {
                    grid-template-columns: 1fr;
                }
                .wrap { padding: 22px 14px 40px; }
                .hero, .panel, .answer-shell { padding: 18px; }
                .display { font-size: clamp(2rem, 11vw, 3rem); }
                .source-item { grid-template-columns: 1fr; }
                .recommend-card { grid-template-columns: 1fr; }
                .recommend-thumb { width: 100%; height: 170px; }
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
    recommendations = youtube.get("recommendations", []) or []
    rate_limit_count = sum(1 for item in status["skipped"] if "429" in str(item.get("reason", "")).lower() or "too many requests" in str(item.get("reason", "")).lower())

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

    recommendation_items = [
        A(
            Img(src=item.get("thumbnail", ""), alt=item.get("title", "Recommended video"), cls="recommend-thumb") if item.get("thumbnail") else Div("No image", cls="empty-state"),
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
                cls="dashboard-main",
            ),
            Div(
                H3("Recommended Videos", cls="section-title"),
                P("Top matches for the current corpus topic. People can open these directly even before indexing finishes.", cls="section-copy"),
                Div(*recommendation_items, cls="recommend-stack") if recommendation_items else Div(
                    P("Search a topic and the top 5 recommended videos will appear here with thumbnails and links.", cls="item-copy"),
                    cls="empty-state",
                ),
                cls="panel",
            ),
            cls="dashboard-body",
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
                    Span("Fastest path to a useful corpus", cls="status-pill ready"),
                    P(
                        "Start with a topic that naturally maps to 5 to 8 long-form videos, not a single exact video title.",
                        cls="section-copy",
                    ),
                    cls="panel tight",
                ),
                Div(
                    P("Use the live dashboard below to watch indexing progress, see what made it into the corpus, and understand skipped videos.", cls="section-copy"),
                    P("If some videos are skipped, TubeMind will tell you whether transcripts were missing or YouTube throttled access.", cls="section-copy"),
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
async def api_query_youtube(request: Request, question: str = "", mode: str = DEFAULT_QUERY_MODE):
    try:
        ans = await app_state.query_youtube(question, mode=mode)
        if request.headers.get("hx-request", "").lower() == "true":
            return render_answer_panel(answer=ans, indexed=app_state.state.youtube_indexed)
        return {"ok": True, "answer": ans}
    except Exception as exc:
        if request.headers.get("hx-request", "").lower() == "true":
            return render_answer_panel(error=str(exc), indexed=app_state.state.youtube_indexed)
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=400)


if __name__ == "__main__":
    serve(host="0.0.0.0", port=5001)
