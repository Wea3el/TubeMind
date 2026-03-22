"""TubeMind runtime services and per-user app registry."""

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
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from youtube_transcript_api import NoTranscriptFound, TooManyRequests, YouTubeRequestFailed, YouTubeTranscriptApi

from tubemind.config import (
    APP_ROOT,
    COOKIE_BROWSER_LABELS,
    DEFAULT_QUERY_MODE,
    MAX_RECOMMENDATIONS,
    QUERY_MODES,
    TRANSCRIPT_CANDIDATE_PADDING,
    TRANSCRIPT_RETRY_ATTEMPTS,
    TRANSCRIPT_RETRY_BASE_DELAY,
    TRANSCRIPT_REQUEST_DELAY_SECONDS,
    TRANSCRIPTAPI_BASE_URL,
    YOUTUBE_SEARCH_URL,
    YOUTUBE_VIDEOS_URL,
)
from tubemind.models import CorpusState, YouTubeVideo, iso8601_duration_to_seconds, now_ms, seconds_to_label, yt_watch_url


class TubeMindApp:
    """Own all per-user state, background work, and LightRAG resources."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._user_root = APP_ROOT / "users" / user_id / "tubemind_app"
        self._rag_storage_dir = self._user_root / "rag_storage"
        self._state_file = self._user_root / "state.json"
        self._user_root.mkdir(parents=True, exist_ok=True)
        self.state = CorpusState.load(self._state_file)
        self.lock = threading.RLock()
        self._dashboard_revision = 0
        self._dashboard_condition = threading.Condition(self.lock)
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
            working_dir=str(self._rag_storage_dir),
            llm_model_func=llm_model,
            embedding_func=openai_embed,
        )

    async def startup(self) -> None:
        """Initialize persistent LightRAG storage and repair dashboard state."""

        await self._run_coro_on_rag_loop(self.rag.initialize_storages())
        await self._repair_rag_backed_state()

    async def shutdown(self) -> None:
        """Finalize storage and stop the dedicated worker loop cleanly."""

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
            self._publish_dashboard_state()

    def _publish_dashboard_state(self) -> None:
        """Persist dashboard-visible state and wake SSE listeners waiting for updates."""

        self.state.save()
        self._dashboard_revision += 1
        self._dashboard_condition.notify_all()

    def wait_for_dashboard_update(self, last_revision: int, timeout: float = 30.0) -> tuple[int, Dict[str, Any]]:
        """Block until the dashboard has a newer revision or the wait times out."""

        with self.lock:
            if self._dashboard_revision <= last_revision:
                self._dashboard_condition.wait(timeout=timeout)
            return self._dashboard_revision, self.status_payload()

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
        should_retry_failed_docs = bool(failed_docs) and any(
            "reasoning_effort" in str(getattr(doc, "error_msg", "") or "").lower()
            for doc in failed_docs.values()
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

            self._publish_dashboard_state()

    def reset_youtube_index(self) -> None:
        """Clear the current corpus state before a new indexing run starts."""

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
            self._publish_dashboard_state()

    def _set_job(self, *, active: bool, stage: str = "", progress: int = 0, total: int = 0, msg: str = "") -> None:
        """Persist job progress fields and publish them to the live dashboard."""

        self.state.job_active = active
        self.state.job_stage = stage
        self.state.job_progress = progress
        self.state.job_total = total
        self.state.job_message = msg
        self._publish_dashboard_state()

    async def youtube_search(self, query: str, *, max_videos: int, min_seconds: int, order: str) -> List[YouTubeVideo]:
        """Search YouTube and normalize the returned videos for TubeMind use."""

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
            response = await client.get(YOUTUBE_SEARCH_URL, params=params)
            data = response.json()
            if response.status_code != 200:
                raise RuntimeError(f"YouTube search failed: {data}")

        items = data.get("items", [])
        video_ids = [it.get("id", {}).get("videoId") for it in items]
        video_ids = [video_id for video_id in video_ids if video_id]
        if not video_ids:
            return []

        params2 = {
            "part": "snippet,contentDetails",
            "id": ",".join(video_ids),
            "key": key,
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(YOUTUBE_VIDEOS_URL, params=params2)
            data2 = response.json()
            if response.status_code != 200:
                raise RuntimeError(f"YouTube videos.list failed: {data2}")

        videos: List[YouTubeVideo] = []
        for item in data2.get("items", []):
            video_id = str(item.get("id", ""))
            snippet = item.get("snippet", {}) or {}
            content_details = item.get("contentDetails", {}) or {}
            duration = iso8601_duration_to_seconds(str(content_details.get("duration", "") or ""))

            thumbnails = snippet.get("thumbnails", {}) or {}
            thumbnail = (
                (thumbnails.get("medium") or {}).get("url")
                or (thumbnails.get("high") or {}).get("url")
                or (thumbnails.get("default") or {}).get("url")
                or ""
            )

            if duration < min_seconds:
                continue

            videos.append(
                YouTubeVideo(
                    video_id=video_id,
                    title=str(snippet.get("title", "")),
                    channel_title=str(snippet.get("channelTitle", "")),
                    published_at=str(snippet.get("publishedAt", "")),
                    thumbnail=thumbnail,
                    duration_sec=duration,
                    url=yt_watch_url(video_id),
                )
            )

        return videos[:max_videos]

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
        """Persist the recommendation panel payload and publish the update immediately."""

        self.state.youtube_recommendations = [
            self._serialize_recommendation(video) for video in videos[:MAX_RECOMMENDATIONS]
        ]
        self._publish_dashboard_state()

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
        for browser in [value.strip().lower() for value in browsers_raw.split(",") if value.strip()]:
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
            return str(detail.get("message") or detail.get("detail") or detail)
        if detail:
            return str(detail)
        if isinstance(payload, dict):
            return str(payload.get("message") or payload)
        return str(payload)

    def _fetch_transcript_with_transcriptapi(self, video: YouTubeVideo) -> tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
        api_key = self._transcript_api_key()
        if not api_key:
            return None, None

        request_headers = {"Authorization": f"Bearer {api_key}"}
        request_params = {"platform": "youtube", "video_id": video.video_id}
        last_err = "unknown TranscriptAPI error"
        retry_delay = 1.0

        with httpx.Client(timeout=30.0) as client:
            for _ in range(3):
                response = client.get(
                    f"{TRANSCRIPTAPI_BASE_URL}/transcripts",
                    params=request_params,
                    headers=request_headers,
                )
                if response.status_code == 200:
                    payload = response.json()
                    cues = payload.get("transcript", []) if isinstance(payload, dict) else []
                    segments = [
                        {"start": float(cue.get("start", 0.0) or 0.0), "text": str(cue.get("text", "")).strip()}
                        for cue in cues
                        if str(cue.get("text", "")).strip()
                    ]
                    if segments:
                        return segments, None
                    last_err = "TranscriptAPI returned an empty transcript"
                    break

                try:
                    payload = response.json()
                except Exception:
                    payload = {"detail": response.text}
                last_err = self._extract_transcriptapi_error(payload)

                if response.status_code in (408, 429, 503):
                    retry_after = response.headers.get("Retry-After")
                    try:
                        retry_delay = float(retry_after) if retry_after else retry_delay
                    except ValueError:
                        pass
                    time.sleep(max(1.0, retry_delay))
                    retry_delay = min(retry_delay * 2, 10.0)
                    continue
                break

        return None, f"TranscriptAPI: {last_err}"

    def _parse_seconds_label(self, value: str) -> float:
        clean = value.strip().replace(",", ".")
        parts = clean.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(clean)

    def _parse_vtt_segments(self, text: str) -> List[Dict[str, Any]]:
        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segments: List[Dict[str, Any]] = []
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if "-->" not in line:
                index += 1
                continue

            start_raw = line.split("-->", 1)[0].strip().split(" ")[0]
            index += 1
            cue_lines: List[str] = []
            while index < len(lines) and lines[index].strip():
                cue_lines.append(lines[index].strip())
                index += 1

            cue_text = re.sub(r"<[^>]+>", "", " ".join(cue_lines)).strip()
            cue_text = html.unescape(cue_text)
            if cue_text:
                segments.append({"start": self._parse_seconds_label(start_raw), "text": cue_text})
        return segments

    def _parse_json3_segments(self, text: str) -> List[Dict[str, Any]]:
        payload = json.loads(text)
        segments: List[Dict[str, Any]] = []
        for event in payload.get("events", []) or []:
            parts = event.get("segs") or []
            if not parts:
                continue
            cue_text = html.unescape("".join(str(part.get("utf8", "")) for part in parts)).replace("\n", " ").strip()
            if not cue_text:
                continue
            segments.append({"start": float(event.get("tStartMs", 0) or 0) / 1000.0, "text": cue_text})
        return segments

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

                    segments = self._read_subtitle_segments(subtitle_files[0])
                    if segments:
                        return segments, None
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
        """Fetch transcript segments using the primary API and configured fallbacks."""

        last_err: Optional[str] = None
        request_kwargs = self._transcript_request_kwargs()

        for attempt in range(1, TRANSCRIPT_RETRY_ATTEMPTS + 1):
            try:
                try:
                    segments = YouTubeTranscriptApi.get_transcript(video.video_id, languages=("en",), **request_kwargs)
                except NoTranscriptFound:
                    segments = YouTubeTranscriptApi.get_transcript(video.video_id, **request_kwargs)

                if not segments:
                    last_err = "empty transcript payload"
                else:
                    return segments, None
            except Exception as exc:
                last_err = self._describe_transcript_error(exc, using_cookies=bool(request_kwargs.get("cookies")))
                if not self._should_retry_transcript_error(exc):
                    break

            if attempt < TRANSCRIPT_RETRY_ATTEMPTS:
                time.sleep(TRANSCRIPT_RETRY_BASE_DELAY * (2 ** (attempt - 1)))

        transcript_api_segments, transcript_api_err = self._fetch_transcript_with_transcriptapi(video)
        if transcript_api_segments:
            return transcript_api_segments, None

        yt_dlp_segments, yt_dlp_err = self._fetch_transcript_with_ytdlp(video)
        if yt_dlp_segments:
            return yt_dlp_segments, None

        fallback_errors = [err for err in (transcript_api_err, yt_dlp_err) if err]
        if fallback_errors:
            last_err = "\n".join([last_err or "transcript fetch failed", *fallback_errors])

        return None, last_err or "unknown transcript error"

    def _format_transcript(self, segments: List[Dict[str, Any]]) -> str:
        lines: List[str] = []
        for segment in segments:
            start_time = float(segment.get("start", 0.0) or 0.0)
            text = str(segment.get("text", "")).replace("\n", " ").strip()
            if not text:
                continue
            lines.append(f"[t={start_time:.1f}] {text}")
        return "\n".join(lines)

    def start_youtube_index_job(self, query: str, *, max_videos: int, min_seconds: int, order: str) -> str:
        """Start the background indexing workflow for a new YouTube corpus topic."""

        normalized = query.strip()
        if not normalized:
            raise ValueError("Enter a YouTube search phrase to index.")

        with self.lock:
            self.reset_youtube_index()

            job_id = f"job_{now_ms()}"
            self.state.job_id = job_id
            self.state.youtube_seed_query = normalized
            self._set_job(active=True, stage="search", progress=0, total=max_videos, msg="Searching YouTube...")

            def runner():
                try:
                    asyncio.run(self._run_youtube_index_job(job_id, normalized, max_videos, min_seconds, order))
                except Exception as exc:
                    with self.lock:
                        self.state.youtube_indexed = False
                        self.state.youtube_video_ids = []
                        self.state.youtube_titles = []
                        self.state.youtube_urls = {}
                        self._set_job(active=False, stage="error", progress=0, total=0, msg=str(exc))

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

        for index, video in enumerate(videos, start=1):
            if len(documents) >= max_videos:
                break

            with self.lock:
                if self.state.job_id != job_id:
                    return
                self._set_job(
                    active=True,
                    stage="transcripts",
                    progress=index - 1,
                    total=len(videos),
                    msg=f"Transcript {index} of {len(videos)}",
                )

            segments, err = self._fetch_transcript(video)
            if not segments:
                if err and self._looks_rate_limited(err):
                    consecutive_rate_limits += 1
                else:
                    consecutive_rate_limits = 0
                with self.lock:
                    self.state.youtube_skipped.append(
                        {
                            "videoId": video.video_id,
                            "title": video.title,
                            "thumbnail": video.thumbnail,
                            "url": video.url,
                            "reason": err or "unknown",
                        }
                    )
                    self._publish_dashboard_state()
                if consecutive_rate_limits >= 2 and not documents:
                    with self.lock:
                        self._set_job(
                            active=False,
                            stage="done",
                            progress=index,
                            total=len(videos),
                            msg="YouTube rate-limited transcript access. Recommendations are still available on the right.",
                        )
                    break
                time.sleep(self._transcript_request_delay())
                continue

            consecutive_rate_limits = 0
            transcript = self._format_transcript(segments)
            if not transcript.strip():
                with self.lock:
                    self.state.youtube_skipped.append(
                        {
                            "videoId": video.video_id,
                            "title": video.title,
                            "thumbnail": video.thumbnail,
                            "url": video.url,
                            "reason": "empty transcript",
                        }
                    )
                    self._publish_dashboard_state()
                continue

            document = "\n\n".join(
                [
                    "Source: YouTube",
                    f"Title: {video.title}",
                    f"Channel: {video.channel_title}",
                    f"PublishedAt: {video.published_at}",
                    f"DurationSec: {video.duration_sec}",
                    f"VideoURL: {video.url}",
                    "",
                    "Transcript (timestamped, seconds):",
                    transcript,
                    "",
                    "Instruction: cite timestamps by writing VideoURL&t=SECONDS.",
                ]
            )

            documents.append(document)
            ids.append(f"youtube:{video.video_id}")
            file_paths.append(video.url)
            indexed_videos.append(video)

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

    async def query_youtube(self, question: str, mode: str = DEFAULT_QUERY_MODE) -> Dict[str, Any]:
        """Return retrieved transcript chunks instead of a synthesized answer."""

        question_text = question.strip()
        if not question_text:
            raise ValueError("Enter a question.")
        if not self.state.youtube_indexed:
            raise ValueError("Index YouTube first. If the previous run failed, start indexing again to finalize the corpus.")
        if mode not in QUERY_MODES:
            mode = DEFAULT_QUERY_MODE

        from lightrag import QueryParam

        result = await self._run_coro_on_rag_loop(
            self.rag.aquery_data(
                question_text,
                param=QueryParam(mode=mode),
            )
        )
        chunks = list((result or {}).get("data", {}).get("chunks", []) or [])

        if not chunks:
            raise RuntimeError("No transcript chunks matched that question. Try a simpler question or re-index different videos.")

        title_by_url = {url: title for title, url in self.state.youtube_urls.items()}
        cleaned_chunks: List[Dict[str, str]] = []
        for chunk in chunks:
            file_path = str(chunk.get("file_path") or "").strip()
            cleaned_chunks.append(
                {
                    "title": title_by_url.get(file_path, file_path or "Indexed transcript"),
                    "url": file_path,
                    "content": str(chunk.get("content") or "").strip(),
                    "reference_id": str(chunk.get("reference_id") or "").strip(),
                    "chunk_id": str(chunk.get("chunk_id") or "").strip(),
                }
            )

        return {
            "question": question_text,
            "mode": mode,
            "chunks": cleaned_chunks,
        }

    def status_payload(self) -> Dict[str, Any]:
        """Return the normalized dashboard payload consumed by the UI and APIs."""

        state = self.state
        indexed_titles = state.youtube_titles if state.youtube_indexed else []
        indexed_urls = state.youtube_urls if state.youtube_indexed else {}
        return {
            "youtube": {
                "indexed": state.youtube_indexed,
                "seed_query": state.youtube_seed_query,
                "count": len(indexed_titles),
                "titles": indexed_titles,
                "urls": indexed_urls,
                "recommendations": state.youtube_recommendations,
            },
            "job": {
                "active": state.job_active,
                "id": state.job_id,
                "stage": state.job_stage,
                "progress": state.job_progress,
                "total": state.job_total,
                "message": state.job_message,
            },
            "skipped": state.youtube_skipped[-30:],
        }


_user_apps: Dict[str, TubeMindApp] = {}
_user_locks: Dict[str, asyncio.Lock] = {}


async def get_user_app(user_id: str) -> TubeMindApp:
    """Return the singleton TubeMind runtime for one authenticated user."""

    if user_id in _user_apps:
        return _user_apps[user_id]
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    async with _user_locks[user_id]:
        if user_id not in _user_apps:
            instance = TubeMindApp(user_id)
            await instance.startup()
            _user_apps[user_id] = instance
    return _user_apps[user_id]


async def shutdown_all_user_apps() -> None:
    """Finalize all active user runtimes during app shutdown."""

    for instance in list(_user_apps.values()):
        try:
            await instance.shutdown()
        except Exception:
            pass
