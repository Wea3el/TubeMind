"""Board-aware TubeMind runtime services."""

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
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx
from openai import AsyncOpenAI
from youtube_transcript_api import NoTranscriptFound, TooManyRequests, YouTubeRequestFailed, YouTubeTranscriptApi

from tubemind.auth import (
    create_board,
    create_board_note,
    create_session,
    get_board_for_user,
    get_or_create_latest_session,
    list_board_notes,
    list_board_videos,
    list_boards,
    list_session_notes,
    list_sessions,
    replace_note_chunks,
    save_note_queries,
    set_active_board,
    update_board,
    upsert_board_videos,
)
from tubemind.config import (
    APP_ROOT,
    COOKIE_BROWSER_LABELS,
    DEFAULT_QUERY_MODE,
    MAX_VIDEOS_DEFAULT,
    MIN_SECONDS_DEFAULT,
    MIN_VIDEOS_DEFAULT,
    QUERY_MODES,
    TRANSCRIPT_CANDIDATE_PADDING,
    TRANSCRIPT_RETRY_ATTEMPTS,
    TRANSCRIPT_RETRY_BASE_DELAY,
    TRANSCRIPT_REQUEST_DELAY_SECONDS,
    TRANSCRIPTAPI_BASE_URL,
    YOUTUBE_SEARCH_URL,
    YOUTUBE_VIDEOS_URL,
)
from pydantic import BaseModel

from tubemind.models import BoardRuntime, BoardWorkspace, YouTubeVideo, iso8601_duration_to_seconds, now_ms, seconds_to_label, yt_watch_url


class RetrievalQuality(BaseModel):
    """Structured output for evaluating whether retrieved content can answer a question."""
    is_relevant: bool      # retrieved chunks actually relate to what was asked
    has_enough_info: bool  # enough detail to give a specific, complete answer
    reasoning: str         # one-sentence explanation for the decision


class TubeMindApp:
    """Own per-user board runtimes, OpenAI calls, and retrieval helpers."""

    def __init__(self, user_id: str) -> None:
        """Create the per-user container that backs every board action."""

        self.user_id = user_id
        self._user_root = APP_ROOT / "users" / user_id
        self._boards_root = self._user_root / "boards"
        self._boards_root.mkdir(parents=True, exist_ok=True)
        self._openai = AsyncOpenAI()
        self._llm_model = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")
        self._board_runtimes: dict[int, BoardRuntime] = {}
        self._rag_loop = asyncio.new_event_loop()
        self._rag_loop_ready = threading.Event()
        self._rag_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        self._start_rag_runtime()

    async def startup(self) -> None:
        """Keep the async service contract without eager board initialization."""

        return None

    async def shutdown(self) -> None:
        """Release initialized board graph resources and stop the RAG loop.

        TubeMind creates board runtimes lazily and keeps them in memory for the
        user's session so repeated questions can reuse the same graph store.
        Some backends expose an explicit storage finalizer while Fast GraphRAG
        persists as part of insert/query operations, so shutdown checks for the
        optional method before stopping the dedicated event loop.
        """

        for runtime in list(self._board_runtimes.values()):
            if runtime.rag is not None:
                try:
                    finalize = getattr(runtime.rag, "finalize_storages", None)
                    if finalize is not None:
                        await self._run_coro_on_rag_loop(finalize())
                except Exception:
                    pass

        if self._rag_thread and self._rag_thread.is_alive():
            self._rag_loop.call_soon_threadsafe(self._rag_loop.stop)
            self._rag_thread.join(timeout=5)

    def _start_rag_runtime(self) -> None:
        """Start the background asyncio loop used for Fast GraphRAG operations.

        The FastHTML request loop should not be reused for long-running graph
        insertions because those operations combine async OpenAI calls, vector
        storage writes, and graph persistence. A dedicated loop gives every
        per-user runtime a stable place to schedule graph work from sync or
        async route handlers without blocking unrelated UI requests.
        """

        self._rag_thread = threading.Thread(target=self._run_rag_loop, name=f"tubemind-rag-{self.user_id}", daemon=True)
        self._rag_thread.start()
        if not self._rag_loop_ready.wait(timeout=5):
            raise RuntimeError("TubeMind could not start the Fast GraphRAG worker loop.")

    def _run_rag_loop(self) -> None:
        """Run the dedicated graph event loop until the app shuts down."""

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
        """Submit one coroutine to the Fast GraphRAG worker loop."""

        if not self._rag_thread or not self._rag_thread.is_alive():
            raise RuntimeError("TubeMind knowledge-base worker is not running.")
        return asyncio.run_coroutine_threadsafe(coro, self._rag_loop)

    async def _run_coro_on_rag_loop(self, coro):
        """Await graph work after scheduling it on the dedicated worker loop."""

        try:
            future = self._submit_coro_to_rag_loop(coro)
        except Exception:
            coro.close()
            raise
        return await asyncio.wrap_future(future)

    async def _create_rag(self, working_dir: Path):
        """Create a Fast GraphRAG instance rooted at one board directory.

        TubeMind keeps one graph store per board so follow-up questions retrieve
        only from the transcripts already accepted into that topic workspace.
        Fast GraphRAG needs a domain prompt and entity ontology up front; the
        defaults here are intentionally YouTube-research oriented but can be
        overridden from the environment for experiments without changing code.
        """

        from fast_graphrag import GraphRAG
        from fast_graphrag._llm import OpenAIEmbeddingService, OpenAILLMService

        entity_types = [
            item.strip()
            for item in os.environ.get(
                "FAST_GRAPHRAG_ENTITY_TYPES",
                "Video,Channel,Creator,Product,Feature,Claim,Price,Comparison,Recommendation,Evidence",
            ).split(",")
            if item.strip()
        ]
        example_queries = os.environ.get(
            "FAST_GRAPHRAG_EXAMPLE_QUERIES",
            "\n".join(
                [
                    "What are the strongest recommendations across these videos?",
                    "Which products, features, or tradeoffs do reviewers compare?",
                    "What source evidence supports the answer to this user question?",
                ]
            ),
        )
        embedding_model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        embedding_dim = int(os.environ.get("OPENAI_EMBEDDING_DIM", "1536"))

        return GraphRAG(
            working_dir=str(working_dir),
            domain=os.environ.get(
                "FAST_GRAPHRAG_DOMAIN",
                "Analyze YouTube transcript evidence for a board-scoped research question. "
                "Identify videos, creators, products, claims, comparisons, and supporting evidence that help answer the user.",
            ),
            example_queries=example_queries,
            entity_types=entity_types,
            config=GraphRAG.Config(
                llm_service=OpenAILLMService(model=self._llm_model),
                embedding_service=OpenAIEmbeddingService(model=embedding_model, embedding_dim=embedding_dim),
            ),
        )

    async def _get_board_runtime(self, board_id: int) -> BoardRuntime:
        """Return the lazily initialized Fast GraphRAG runtime for one board."""

        with self.lock:
            cached = self._board_runtimes.get(board_id)
        if cached is not None:
            return cached

        board_root = self._boards_root / str(board_id)
        runtime = BoardRuntime(
            board_id=board_id,
            working_dir=board_root / "rag_storage",
            transcript_dir=board_root / "transcripts",
        )
        runtime.working_dir.mkdir(parents=True, exist_ok=True)
        runtime.transcript_dir.mkdir(parents=True, exist_ok=True)
        runtime.rag = await self._run_coro_on_rag_loop(self._create_rag(runtime.working_dir))
        with self.lock:
            self._board_runtimes[board_id] = runtime
        return runtime

    def build_workspace(self, active_board_id: int | None, *, active_session_id: int | None = None, notice: str = "", warning: str = "") -> BoardWorkspace:
        """Assemble the sidebar and active board payload used by the UI.

        When a session_id is provided the chat thread is scoped to that session
        only. If no session_id is given the most recent session for the board is
        used, creating one automatically if the board has none yet.
        """

        boards = list_boards(self.user_id)
        active_board = get_board_for_user(self.user_id, active_board_id)
        sessions: list[dict] = []
        resolved_session_id: int | None = None
        notes: list[dict] = []

        if active_board:
            board_id_int = int(active_board["id"])
            sessions = list_sessions(board_id_int)
            if active_session_id:
                resolved_session_id = active_session_id
            elif sessions:
                resolved_session_id = int(sessions[0]["id"])
            if resolved_session_id:
                notes = list_session_notes(board_id_int, resolved_session_id)
            else:
                # Fallback: show all notes for legacy boards with no sessions yet
                notes = list_board_notes(board_id_int)

        return BoardWorkspace(
            boards=boards,
            active_board=active_board,
            notes=notes,
            notice=notice,
            warning=warning,
            sessions=sessions,
            active_session_id=resolved_session_id,
        )

    async def create_empty_board(self) -> BoardWorkspace:
        """Create an empty board with a default session and make it active."""

        board = create_board(self.user_id, "Untitled board", "", "", "idle")
        board_id_int = int(board["id"])
        set_active_board(self.user_id, board_id_int)
        session = create_session(board_id_int)
        return self.build_workspace(board_id_int, active_session_id=int(session["id"]), notice="Created a new board.")

    async def answer_question(self, board_id: int | None, question: str, mode: str = DEFAULT_QUERY_MODE, session_id: int | None = None, min_seconds: int = MIN_SECONDS_DEFAULT, min_videos: int = MIN_VIDEOS_DEFAULT, max_videos: int = MAX_VIDEOS_DEFAULT) -> BoardWorkspace:
        """Create a new note by reusing or expanding the selected board corpus.

        The session_id scopes the note to one independent chat thread inside the
        board. If no session_id is provided the latest session is used, creating
        one automatically so every answer always belongs to a session.
        """

        question_text = str(question or "").strip()
        if not question_text:
            raise ValueError("Enter a question to create a note.")
        if mode not in QUERY_MODES:
            mode = DEFAULT_QUERY_MODE

        board = get_board_for_user(self.user_id, board_id) if board_id else None
        if board is None:
            board = create_board(self.user_id, question_text, question_text, "", "working")
        board_id_int = int(board["id"])
        set_active_board(self.user_id, board_id_int)

        # Resolve or create the session for this answer
        if session_id:
            resolved_session_id = session_id
        else:
            session = get_or_create_latest_session(board_id_int)
            resolved_session_id = int(session["id"])

        notes = list_board_notes(board_id_int)
        if notes:
            fit = await self._assess_topic_fit(board, notes, question_text)
            if not fit["is_fit"]:
                return self.build_workspace(board_id_int, active_session_id=resolved_session_id, warning=fit["warning"])

        update_board(board_id_int, status="working", updated_at=now_ms())
        runtime = await self._get_board_runtime(board_id_int)
        initial = await self._query_board(board_id_int, runtime, question_text, mode, allow_empty=True)
        plan = await self._plan_research(board, notes, question_text, initial)
        queries = list(plan.get("queries") or [])
        if not initial.get("chunks") and not queries:
            queries = self._fallback_youtube_queries(board, question_text)

        if queries:
            await self._expand_board_corpus(board_id_int, runtime, queries, min_seconds=min_seconds, min_videos=min_videos, max_videos=max_videos)

        result = initial
        if queries or not result.get("chunks") or not str(result.get("answer", "") or "").strip():
            result = await self._query_board(board_id_int, runtime, question_text, mode, allow_empty=False)

        # Structured-output quality gate: if the answer is vague or the retrieved
        # content doesn't actually cover the question, fetch one more round of videos.
        set_board_progress(board_id_int, "Checking answer quality...")
        quality = await self._check_retrieval_quality(
            question_text,
            str(result.get("answer", "") or ""),
            result.get("chunks", []),
            board,
        )
        if not quality.is_relevant or not quality.has_enough_info:
            set_board_progress(board_id_int, "Not enough info — searching for more videos...")
            extra_queries = self._fallback_youtube_queries(board, question_text)
            await self._expand_board_corpus(
                board_id_int, runtime, extra_queries,
                min_seconds=min_seconds, min_videos=1, max_videos=max_videos,
            )
            result = await self._query_board(board_id_int, runtime, question_text, mode, allow_empty=False)

        answer_text = str(result.get("answer") or "").strip()
        if not result.get("chunks") and not answer_text:
            update_board(board_id_int, status="error")
            raise RuntimeError("TubeMind could not find enough transcript evidence for that note.")

        note = create_board_note(
            board_id=board_id_int,
            question=question_text,
            answer=answer_text or "TubeMind found evidence but could not synthesize a final answer.",
            query_mode=mode,
            session_id=resolved_session_id,
        )
        save_note_queries(board_id_int, int(note["id"]), queries)
        replace_note_chunks(int(note["id"]), result.get("chunks", []))
        update_board(board_id_int, status="ready", last_question_at=now_ms(), updated_at=now_ms())
        await self._refresh_board_summary(board_id_int)
        return self.build_workspace(board_id_int, active_session_id=resolved_session_id, notice="Added a new note to the board.")

    async def _check_retrieval_quality(
        self,
        question: str,
        answer: str,
        chunks: list[dict[str, Any]],
        board: dict[str, Any],
    ) -> RetrievalQuality:
        """Structured-output gate: is the retrieved evidence relevant and sufficient?

        Uses OpenAI structured outputs with the RetrievalQuality Pydantic model so
        the response is guaranteed to parse. If the model is unavailable or the call
        fails, returns a conservative fallback that treats non-empty chunks as
        sufficient so the existing answer is not discarded silently.
        """
        if not chunks:
            return RetrievalQuality(
                is_relevant=False,
                has_enough_info=False,
                reasoning="No transcript chunks were retrieved.",
            )

        excerpt_texts = [str(c.get("content", "") or "")[:300] for c in chunks[:6]]
        try:
            response = await self._openai.beta.chat.completions.parse(
                model=self._llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are evaluating whether YouTube transcript excerpts retrieved by a RAG system "
                            "can fully answer a user's research question. "
                            "Set is_relevant=false if the excerpts are mostly about unrelated topics. "
                            "Set has_enough_info=false if the answer would be vague, incomplete, or says "
                            "'not explicitly listed' or similar hedges indicating missing data. "
                            "Be strict — partial answers with clear gaps should be marked has_enough_info=false."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps({
                            "board_topic": str(board.get("topic_anchor", "") or board.get("title", "") or ""),
                            "question": question,
                            "draft_answer_preview": answer[:600] if answer else "",
                            "retrieved_excerpts": excerpt_texts,
                        }),
                    },
                ],
                response_format=RetrievalQuality,
            )
            parsed = response.choices[0].message.parsed
            if parsed is not None:
                return parsed
        except Exception:
            pass

        # Fallback when structured output fails: trust chunks over empty answer
        sufficient = bool(chunks and answer and "not explicitly listed" not in answer.lower() and "not listed" not in answer.lower())
        return RetrievalQuality(
            is_relevant=bool(chunks),
            has_enough_info=sufficient,
            reasoning="Structured output call failed; inferred from answer content.",
        )

    async def _assess_topic_fit(self, board: dict[str, Any], notes: list[dict[str, Any]], question: str) -> dict[str, Any]:
        """Keep follow-up notes near the board topic instead of silently drifting."""

        system_prompt = (
            "Respond with JSON only: {\"is_fit\": boolean, \"warning\": string}. "
            "Mark obviously different topics as not fitting, but allow natural follow-up questions."
        )
        prompt = json.dumps(
            {
                "board_title": board.get("title", ""),
                "topic_anchor": board.get("topic_anchor", ""),
                "recent_questions": [str(item.get("question", "") or "") for item in notes[-4:]],
                "new_question": question,
            }
        )
        result = await self._llm_json(system_prompt, prompt)
        if isinstance(result, dict) and "is_fit" in result:
            return {
                "is_fit": bool(result.get("is_fit")),
                "warning": str(result.get("warning", "") or "").strip() or "That question looks like a different topic. Start a new board for it instead.",
            }

        haystack = " ".join([str(board.get("topic_anchor", "") or ""), *[str(item.get("question", "") or "") for item in notes[-4:]]]).casefold()
        overlap = {
            token
            for token in re.findall(r"[a-z0-9]+", question.casefold())
            if len(token) > 2 and token in haystack
        }
        return {
            "is_fit": bool(overlap),
            "warning": "That question looks like a different topic. Start a new board for it instead.",
        }

    async def _plan_research(
        self,
        board: dict[str, Any],
        notes: list[dict[str, Any]],
        question: str,
        initial: dict[str, Any],
    ) -> dict[str, Any]:
        """Decide whether the existing board corpus is enough for this note."""

        board_videos = list_board_videos(int(board["id"]))
        if not board_videos:
            return {"queries": self._fallback_youtube_queries(board, question)}

        system_prompt = (
            "Respond with JSON only using keys needs_more (boolean), rationale (string), "
            "queries (array of objects with query and reason). Generate 1-3 YouTube queries only when more evidence is needed."
        )
        prompt = json.dumps(
            {
                "board_title": board.get("title", ""),
                "topic_anchor": board.get("topic_anchor", ""),
                "recent_questions": [str(item.get("question", "") or "") for item in notes[-4:]],
                "video_titles": [str(item.get("title", "") or "") for item in board_videos[-10:]],
                "question": question,
                "draft_answer": str(initial.get("answer", "") or ""),
                "chunk_excerpts": [str(chunk.get("content", "") or "")[:240] for chunk in initial.get("chunks", [])[:4]],
            }
        )
        result = await self._llm_json(system_prompt, prompt)
        if not isinstance(result, dict):
            return {"queries": [] if initial.get("chunks") else self._fallback_youtube_queries(board, question)}

        queries = []
        for item in list(result.get("queries") or [])[:3]:
            query_text = str(item.get("query", "") or "").strip()
            if query_text:
                queries.append({"query": query_text, "reason": str(item.get("reason", "") or "").strip()})
        return {"queries": queries if bool(result.get("needs_more")) else []}

    async def _refresh_board_summary(self, board_id: int) -> None:
        """Apply the board-title rules after each successful note insertion."""

        notes = list_board_notes(board_id)
        if not notes:
            update_board(board_id, title="Untitled board", summary="", topic_anchor="")
            return
        if len(notes) == 1:
            first = str(notes[0].get("question", "") or "").strip()
            update_board(board_id, title=first, summary="", topic_anchor=first)
            return
        if len(notes) == 2:
            update_board(
                board_id,
                title=f'{notes[0].get("question", "")} / {notes[1].get("question", "")}',
                summary="",
                topic_anchor=str(notes[0].get("question", "") or "").strip(),
            )
            return

        result = await self._llm_json(
            "Respond with JSON only using keys title, summary, topic_anchor. Keep the title short and the summary to one or two sentences.",
            json.dumps(
                {
                    "questions": [str(item.get("question", "") or "") for item in notes[-6:]],
                    "answers": [str(item.get("answer", "") or "")[:350] for item in notes[-6:]],
                }
            ),
        )
        if isinstance(result, dict) and str(result.get("title", "") or "").strip():
            update_board(
                board_id,
                title=str(result.get("title", "") or "").strip(),
                summary=str(result.get("summary", "") or "").strip(),
                topic_anchor=str(result.get("topic_anchor", "") or "").strip() or str(notes[0].get("question", "") or "").strip(),
            )
            return

        fallback = " / ".join(str(item.get("question", "") or "").strip() for item in notes[:3])
        update_board(
            board_id,
            title=fallback,
            summary="This board groups related questions answered from a shared YouTube research corpus.",
            topic_anchor=str(notes[0].get("question", "") or "").strip(),
        )

    async def _llm_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        """Ask the configured OpenAI model for a small JSON object."""

        try:
            response = await self._openai.responses.create(
                model=self._llm_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            return None

        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else None
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
                return payload if isinstance(payload, dict) else None
            except Exception:
                return None

    def _fallback_youtube_queries(self, board: dict[str, Any], question: str) -> list[dict[str, str]]:
        """Generate deterministic search queries when the planning model fails."""

        anchor = str(board.get("topic_anchor", "") or "").strip()
        queries = [{"query": question, "reason": "Direct search for the current note question."}]
        if anchor and anchor.casefold() not in question.casefold():
            queries.append({"query": f"{anchor} {question}".strip(), "reason": "Keep the search anchored to the board topic."})
        return queries[:2]

    async def youtube_search(self, query: str, *, max_videos: int, min_seconds: int, order: str) -> list[YouTubeVideo]:
        """Search YouTube and normalize the result list for indexing.

        Hosted deployments are much more reliable when TubeMind targets videos
        that already advertise captions and allow embedding, because transcript
        fallbacks like yt-dlp are more likely to hit bot checks from cloud IPs.
        """

        key = os.environ["YOUTUBE_API_KEY"]
        max_results = str(min(max_videos, 25))
        search_variants = [
            {
                "part": "snippet",
                "type": "video",
                "maxResults": max_results,
                "q": query,
                "order": order,
                "videoCaption": "closedCaption",
                "videoEmbeddable": "true",
                "key": key,
            },
            {
                "part": "snippet",
                "type": "video",
                "maxResults": max_results,
                "q": query,
                "order": order,
                "videoEmbeddable": "true",
                "key": key,
            },
        ]

        video_ids: list[str] = []
        seen_video_ids: set[str] = set()
        async with httpx.AsyncClient(timeout=30) as client:
            for params in search_variants:
                response = await client.get(YOUTUBE_SEARCH_URL, params=params)
                data = response.json()
                if response.status_code != 200:
                    raise RuntimeError(f"YouTube search failed: {data}")
                for item in data.get("items", []):
                    video_id = str(item.get("id", {}).get("videoId") or "").strip()
                    if not video_id or video_id in seen_video_ids:
                        continue
                    seen_video_ids.add(video_id)
                    video_ids.append(video_id)
                if len(video_ids) >= min(max_videos, 12):
                    break

        if not video_ids:
            return []

        params2 = {"part": "snippet,contentDetails", "id": ",".join(video_ids), "key": key}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(YOUTUBE_VIDEOS_URL, params=params2)
            data2 = response.json()
            if response.status_code != 200:
                raise RuntimeError(f"YouTube videos.list failed: {data2}")

        videos: list[YouTubeVideo] = []
        for item in data2.get("items", []):
            duration = iso8601_duration_to_seconds(str((item.get("contentDetails") or {}).get("duration", "") or ""))
            if duration < min_seconds:
                continue
            snippet = item.get("snippet", {}) or {}
            thumbs = snippet.get("thumbnails", {}) or {}
            thumbnail = (thumbs.get("medium") or {}).get("url") or (thumbs.get("high") or {}).get("url") or (thumbs.get("default") or {}).get("url") or ""
            video_id = str(item.get("id", "") or "").strip()
            videos.append(
                YouTubeVideo(
                    video_id=video_id,
                    title=str(snippet.get("title", "") or "").strip(),
                    channel_title=str(snippet.get("channelTitle", "") or "").strip(),
                    published_at=str(snippet.get("publishedAt", "") or "").strip(),
                    thumbnail=thumbnail,
                    duration_sec=duration,
                    url=yt_watch_url(video_id),
                )
            )
        return videos[:max_videos]

    def _transcript_candidate_pool(self, max_videos: int) -> int:
        """Pad the initial result pool to offset transcript failures."""

        raw_pad = str(os.environ.get("YOUTUBE_TRANSCRIPT_CANDIDATE_PADDING", TRANSCRIPT_CANDIDATE_PADDING))
        try:
            pad = int(raw_pad)
        except ValueError:
            pad = TRANSCRIPT_CANDIDATE_PADDING
        return min(25, max_videos + max(0, min(8, pad)))

    def _transcript_request_delay(self) -> float:
        """Return the configured pause between transcript fetches."""

        raw = str(os.environ.get("YOUTUBE_TRANSCRIPT_REQUEST_DELAY_SECONDS", TRANSCRIPT_REQUEST_DELAY_SECONDS))
        try:
            delay = float(raw)
        except ValueError:
            delay = TRANSCRIPT_REQUEST_DELAY_SECONDS
        return max(0.0, min(10.0, delay))

    def _normalize_alignment_text(self, text: str) -> str:
        """Normalize transcript text for chunk-to-timestamp alignment."""

        cleaned = re.sub(r"\s+", " ", text or "").strip().lower()
        return re.sub(r"[^a-z0-9 ]+", "", cleaned)

    def _transcript_cache_path(self, runtime: BoardRuntime, video_id: str) -> Path:
        """Return the board-local transcript artifact path for one video."""

        return runtime.transcript_dir / f"{video_id}.json"

    def _build_clean_transcript_artifact(self, video: YouTubeVideo, segments: list[dict[str, Any]]) -> dict[str, Any]:
        """Create the clean transcript text plus a timing sidecar artifact."""

        clean_parts: list[str] = []
        normalized_parts: list[str] = []
        artifact_segments: list[dict[str, Any]] = []
        cursor = 0
        for segment in segments:
            cleaned_text = re.sub(r"\s+", " ", str(segment.get("text", "") or "")).strip()
            if not cleaned_text:
                continue
            clean_parts.append(cleaned_text)
            normalized = self._normalize_alignment_text(cleaned_text)
            if not normalized:
                continue
            if normalized_parts:
                cursor += 1
            offset_start = cursor
            normalized_parts.append(normalized)
            cursor += len(normalized)
            artifact_segments.append({"start": float(segment.get("start", 0.0) or 0.0), "text": cleaned_text, "offset_start": offset_start, "offset_end": cursor})
        return {
            "video_id": video.video_id,
            "url": video.url,
            "clean_text": " ".join(clean_parts),
            "normalized_text": " ".join(normalized_parts),
            "segments": artifact_segments,
        }

    def _save_transcript_artifact(self, runtime: BoardRuntime, video: YouTubeVideo, segments: list[dict[str, Any]]) -> str:
        """Persist the timing sidecar and return the clean transcript text."""

        artifact = self._build_clean_transcript_artifact(video, segments)
        self._transcript_cache_path(runtime, video.video_id).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return str(artifact.get("clean_text", "") or "").strip()

    def _video_id_from_url(self, url: str) -> str:
        """Extract a YouTube video id from a stored watch URL."""

        try:
            return str(parse_qs(urlparse(url).query).get("v", [""])[0] or "").strip()
        except Exception:
            return ""

    def _youtube_embed_url(self, video_id: str, start_seconds: float) -> str:
        """Build an embeddable YouTube URL anchored to a chunk timestamp."""

        return f"https://www.youtube.com/embed/{video_id}?start={max(0, int(start_seconds))}&rel=0"

    def _find_chunk_start_seconds(self, runtime: BoardRuntime, video_id: str, chunk_text: str) -> float:
        """Approximate a retrieved chunk's start time from the saved transcript artifact."""

        artifact_path = self._transcript_cache_path(runtime, video_id)
        if not artifact_path.exists():
            return 0.0
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception:
            return 0.0
        normalized_chunk = self._normalize_alignment_text(chunk_text)
        normalized_transcript = str(artifact.get("normalized_text", "") or "")
        if not normalized_chunk or not normalized_transcript:
            return 0.0
        offset = normalized_transcript.find(normalized_chunk)
        if offset < 0:
            excerpt = normalized_chunk[:120]
            if excerpt:
                offset = normalized_transcript.find(excerpt)
        if offset < 0:
            return 0.0
        for segment in artifact.get("segments", []) or []:
            if int(segment.get("offset_end", 0) or 0) >= offset:
                return float(segment.get("start", 0.0) or 0.0)
        return 0.0

    def _transcript_request_kwargs(self) -> dict[str, Any]:
        """Return optional cookie-file settings for transcript requests."""

        cookies_file = str(os.environ.get("YOUTUBE_TRANSCRIPT_COOKIES_FILE", "")).strip().strip("'").strip('"')
        return {"cookies": cookies_file} if cookies_file else {}

    def _transcript_api_key(self) -> str:
        """Return the configured TranscriptAPI key when present."""

        return str(os.environ.get("TRANSCRIPTAPI_API_KEY", "")).strip().strip("'").strip('"')

    def _summarize_transcript_failures(self, failures: list[str]) -> str:
        """Compress repeated transcript fetch failures into a readable warning."""

        seen: set[str] = set()
        unique_failures: list[str] = []
        for failure in failures:
            cleaned = str(failure or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique_failures.append(cleaned)

        if not unique_failures:
            return "TubeMind found videos, but transcript fetching failed before any evidence could be indexed."

        preview = "\n\n".join(unique_failures[:3])
        return (
            "TubeMind found videos, but could not fetch any usable transcripts for them.\n\n"
            f"{preview}\n\n"
            "On hosted deployments this usually means TranscriptAPI auth is invalid, quota is exhausted, or the candidate videos do not have captions."
        )

    def _summarize_indexing_failures(self, failures: list[dict[str, str]]) -> str:
        """Compress document indexing failures into a readable warning."""

        previews: list[str] = []
        seen: set[str] = set()
        for failure in failures:
            title = str(failure.get("title", "") or "Indexed transcript").strip()
            reason = str(failure.get("reason", "") or "unknown indexing error").strip()
            line = f"{title}: {reason}"
            if line in seen:
                continue
            seen.add(line)
            previews.append(line)

        if not previews:
            return "TubeMind fetched transcripts, but indexing them into the board failed before any evidence became available."

        return (
            "TubeMind fetched transcripts, but indexing them into the board failed before any evidence became available.\n\n"
            f"{chr(10).join(previews[:3])}"
        )

    def _is_transcript_rate_limited(self, exc: Exception) -> bool:
        """Detect transcript rate-limit conditions across providers."""

        if isinstance(exc, TooManyRequests):
            return True
        return "429" in str(exc).lower() or "too many requests" in str(exc).lower()

    def _should_retry_transcript_error(self, exc: Exception) -> bool:
        """Decide whether a transcript failure is transient enough to retry."""

        if self._is_transcript_rate_limited(exc):
            return True
        if isinstance(exc, YouTubeRequestFailed):
            text = str(exc).lower()
            return "timed out" in text or "temporarily unavailable" in text
        return False

    def _describe_transcript_error(self, exc: Exception, *, using_cookies: bool) -> str:
        """Convert transcript exceptions into readable diagnostics."""

        message = f"{type(exc).__name__}: {str(exc)}"
        if self._is_transcript_rate_limited(exc) and not using_cookies:
            return f"{message}\nHint: set YOUTUBE_TRANSCRIPT_COOKIES_FILE to reduce transcript 429s."
        return message

    def _extract_transcriptapi_error(self, payload: Any) -> str:
        """Normalize TranscriptAPI error payloads into one short string."""

        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, dict):
            return str(detail.get("message") or detail.get("detail") or detail)
        if detail:
            return str(detail)
        if isinstance(payload, dict):
            return str(payload.get("message") or payload)
        return str(payload)

    def _fetch_transcript_with_transcriptapi(self, video: YouTubeVideo) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
        """Try TranscriptAPI before falling back to other transcript sources."""

        api_key = self._transcript_api_key()
        if not api_key:
            return None, None
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {
            "video_url": video.video_id,
            "format": "json",
            "include_timestamp": "true",
        }
        last_err = "unknown TranscriptAPI error"
        retry_delay = 1.0
        with httpx.Client(timeout=30.0) as client:
            for _ in range(3):
                response = client.get(f"{TRANSCRIPTAPI_BASE_URL}/youtube/transcript", params=params, headers=headers)
                if response.status_code == 200:
                    payload = response.json()
                    cues = payload.get("transcript", []) if isinstance(payload, dict) else []
                    segments = [{"start": float(cue.get("start", 0.0) or 0.0), "text": str(cue.get("text", "")).strip()} for cue in cues if str(cue.get("text", "")).strip()]
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
        """Parse a VTT timestamp into float seconds."""

        clean = value.strip().replace(",", ".")
        parts = clean.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(clean)

    def _parse_vtt_segments(self, text: str) -> list[dict[str, Any]]:
        """Parse subtitle cues from a VTT file."""

        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segments: list[dict[str, Any]] = []
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if "-->" not in line:
                index += 1
                continue
            start_raw = line.split("-->", 1)[0].strip().split(" ")[0]
            index += 1
            cue_lines: list[str] = []
            while index < len(lines) and lines[index].strip():
                cue_lines.append(lines[index].strip())
                index += 1
            cue_text = html.unescape(re.sub(r"<[^>]+>", "", " ".join(cue_lines)).strip())
            if cue_text:
                segments.append({"start": self._parse_seconds_label(start_raw), "text": cue_text})
        return segments

    def _parse_json3_segments(self, text: str) -> list[dict[str, Any]]:
        """Parse subtitle cues from a YouTube json3 subtitle file."""

        payload = json.loads(text)
        segments: list[dict[str, Any]] = []
        for event in payload.get("events", []) or []:
            parts = event.get("segs") or []
            if not parts:
                continue
            cue_text = html.unescape("".join(str(part.get("utf8", "")) for part in parts)).replace("\n", " ").strip()
            if cue_text:
                segments.append({"start": float(event.get("tStartMs", 0) or 0) / 1000.0, "text": cue_text})
        return segments

    def _read_subtitle_segments(self, path: Path) -> list[dict[str, Any]]:
        """Load subtitle cues from either json3 or vtt files."""

        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json3":
            return self._parse_json3_segments(text)
        if path.suffix.lower() == ".vtt":
            return self._parse_vtt_segments(text)
        raise RuntimeError(f"Unsupported subtitle format: {path.name}")

    def _yt_dlp_cookie_sources(self) -> list[tuple[str, dict[str, Any]]]:
        """Return the ordered yt-dlp cookie strategies to try."""

        sources: list[tuple[str, dict[str, Any]]] = [("yt-dlp", {})]
        cookie_file = str(os.environ.get("YOUTUBE_TRANSCRIPT_COOKIES_FILE", "")).strip().strip("'").strip('"')
        if cookie_file:
            sources.append(("yt-dlp + cookie file", {"cookiefile": cookie_file}))
        browsers_raw = str(os.environ.get("YOUTUBE_COOKIES_BROWSER", "")).strip().strip("'").strip('"')
        for browser in [value.strip().lower() for value in browsers_raw.split(",") if value.strip()]:
            sources.append((f"yt-dlp + {COOKIE_BROWSER_LABELS.get(browser, browser.title())} cookies", {"cookiesfrombrowser": (browser, None, None, None)}))
        return sources

    class _QuietYTDLPLogger:
        """Suppress yt-dlp log spam inside the UI request cycle."""

        def debug(self, msg: str) -> None:
            return None

        def warning(self, msg: str) -> None:
            return None

        def error(self, msg: str) -> None:
            return None

    def _fetch_transcript_with_ytdlp(self, video: YouTubeVideo) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
        """Use yt-dlp subtitle download as the last transcript fallback."""

        try:
            from yt_dlp import YoutubeDL
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
                        "logger": self._QuietYTDLPLogger(),
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
                last_err = f"{source_label}: {type(exc).__name__}: {exc}"
        return None, last_err

    def _fetch_transcript(self, video: YouTubeVideo) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
        """Fetch transcript segments with layered provider fallbacks."""

        last_err: Optional[str] = None
        transcript_api_err: Optional[str] = None
        yt_dlp_err: Optional[str] = None
        prefer_transcriptapi = bool(self._transcript_api_key())
        if prefer_transcriptapi:
            segments, transcript_api_err = self._fetch_transcript_with_transcriptapi(video)
            if segments:
                return segments, None
            last_err = transcript_api_err

        request_kwargs = self._transcript_request_kwargs()
        for attempt in range(1, TRANSCRIPT_RETRY_ATTEMPTS + 1):
            try:
                try:
                    segments = YouTubeTranscriptApi.get_transcript(video.video_id, languages=("en",), **request_kwargs)
                except NoTranscriptFound:
                    segments = YouTubeTranscriptApi.get_transcript(video.video_id, **request_kwargs)
                if segments:
                    return segments, None
                last_err = "empty transcript payload"
            except Exception as exc:
                last_err = self._describe_transcript_error(exc, using_cookies=bool(request_kwargs.get("cookies")))
                if not self._should_retry_transcript_error(exc):
                    break
            if attempt < TRANSCRIPT_RETRY_ATTEMPTS:
                time.sleep(TRANSCRIPT_RETRY_BASE_DELAY * (2 ** (attempt - 1)))

        if not prefer_transcriptapi:
            segments, transcript_api_err = self._fetch_transcript_with_transcriptapi(video)
            if segments:
                return segments, None

        segments, yt_dlp_err = self._fetch_transcript_with_ytdlp(video)
        if segments:
            return segments, None
        errors = [err for err in (last_err, transcript_api_err, yt_dlp_err) if err]
        return None, "\n".join(errors) if errors else "unknown transcript error"

    def _youtube_video_id_from_doc_id(self, doc_id: str) -> str:
        """Extract the YouTube id from a persisted transcript document id.

        Older board stores and insertion code use ``youtube:<video_id>`` ids to
        keep source identity stable across transcript fetches, graph inserts,
        and database rows. Fast GraphRAG stores source details in chunk metadata
        instead of exposing a document-status table.
        """

        return doc_id.split(":", 1)[1].strip() if doc_id.startswith("youtube:") else ""

    def _extract_title_from_summary(self, summary: str) -> str:
        """Recover a transcript title from a status or diagnostic summary."""

        match = re.search(r"(?m)^Title:\s*(.+)$", summary or "")
        return match.group(1).strip() if match else ""

    def _doc_item_key(self, doc_id: str, item: dict[str, str]) -> str:
        """Build a stable key for deduplicating document status rows."""

        return str(item.get("videoId") or item.get("url") or item.get("title") or doc_id)


    async def _expand_board_corpus(self, board_id: int, runtime: BoardRuntime, queries: list[dict[str, str]], *, min_seconds: int = MIN_SECONDS_DEFAULT, min_videos: int = MIN_VIDEOS_DEFAULT, max_videos: int = MAX_VIDEOS_DEFAULT) -> None:
        """Search, fetch, and index additional videos into one board corpus.

        This is the ingestion side of TubeMind's retrieval loop. It expands the
        board only with videos not already recorded in the app database, stores
        normalized transcripts on disk for timestamp recovery, inserts transcript
        text into Fast GraphRAG with YouTube metadata attached to each document,
        enforces the user-selected minimum usable video count, and persists the
        successfully indexed videos so future questions can reuse the same
        evidence without another YouTube search.
        """

        existing_ids = {str(item.get("video_id", "") or "").strip() for item in list_board_videos(board_id)}
        queued_ids = set(existing_ids)
        documents: list[str] = []
        indexed_videos: list[YouTubeVideo] = []
        origin_query_by_video_id: dict[str, str] = {}
        transcript_failures: list[str] = []

        for item in queries[:3]:
            query_text = str(item.get("query", "") or "").strip()
            if not query_text:
                continue
            videos = await self.youtube_search(
                query_text,
                max_videos=self._transcript_candidate_pool(max_videos),
                min_seconds=min_seconds,
                order="relevance",
            )
            if not videos:
                transcript_failures.append(f'No caption-friendly YouTube results were found for query "{query_text}".')
                continue
            loop = asyncio.get_running_loop()
            for video in videos:
                if video.video_id in queued_ids:
                    continue
                queued_ids.add(video.video_id)
                segments, transcript_error = await loop.run_in_executor(None, self._fetch_transcript, video)
                if not segments:
                    transcript_failures.append(f"{video.title}: {transcript_error or 'unknown transcript error'}")
                    await asyncio.sleep(self._transcript_request_delay())
                    continue
                transcript = self._save_transcript_artifact(runtime, video, segments)
                if not transcript.strip():
                    transcript_failures.append(f"{video.title}: fetched transcript was empty after normalization")
                    await asyncio.sleep(self._transcript_request_delay())
                    continue
                documents.append(transcript)
                indexed_videos.append(video)
                origin_query_by_video_id[video.video_id] = query_text
                await asyncio.sleep(self._transcript_request_delay())
                if len(documents) >= max_videos:
                    break
            if len(documents) >= max_videos:
                break

        if not documents:
            if transcript_failures:
                raise RuntimeError(self._summarize_transcript_failures(transcript_failures))
            return

        if len(documents) < min_videos:
            failure_hint = f"\n\n{self._summarize_transcript_failures(transcript_failures)}" if transcript_failures else ""
            raise RuntimeError(
                f"TubeMind only found {len(documents)} usable transcript(s), but at least {min_videos} are required. "
                f"Try lowering 'Min videos to index', reducing 'Min. video length', or rephrasing your question.{failure_hint}"
            )

        metadata = [
            {
                "doc_id": f"youtube:{video.video_id}",
                "video_id": video.video_id,
                "title": video.title,
                "url": video.url,
                "thumbnail": video.thumbnail,
                "channel_title": video.channel_title,
            }
            for video in indexed_videos
        ]
        try:
            await self._run_coro_on_rag_loop(runtime.rag.async_insert(documents, metadata=metadata, show_progress=False))
        except Exception as exc:
            failed = [
                {
                    "videoId": video.video_id,
                    "title": video.title,
                    "url": video.url,
                    "thumbnail": video.thumbnail,
                    "reason": f"Indexing failed: {exc}",
                }
                for video in indexed_videos
            ]
            raise RuntimeError(self._summarize_indexing_failures(failed)) from exc

        grouped: dict[str, list[dict[str, Any]]] = {}
        for video in indexed_videos:
            origin_query = origin_query_by_video_id.get(video.video_id, "")
            grouped.setdefault(origin_query, []).append(
                {
                    "video_id": video.video_id,
                    "title": video.title,
                    "url": video.url,
                    "thumbnail": video.thumbnail,
                    "channel_title": video.channel_title,
                }
            )
        for origin_query, videos in grouped.items():
            upsert_board_videos(board_id, videos, origin_query=origin_query)

    def _find_relevant_chunks(self, runtime: BoardRuntime, board_videos: list[dict[str, Any]], question: str, top_k: int = 6) -> list[dict[str, Any]]:
        """Find the most relevant transcript passages for timestamp source attribution.

        Results are capped at 2 chunks per video so evidence spans multiple sources.
        """

        _stop = {"the", "a", "an", "is", "in", "of", "to", "and", "or", "for", "what", "how", "why", "when", "where", "who", "i", "it", "be", "do", "have"}
        q_tokens = set(re.findall(r"[a-z0-9]+", question.lower())) - _stop
        if not q_tokens:
            return []

        title_by_id = {str(v.get("video_id", "") or ""): str(v.get("title", "") or "Indexed transcript") for v in board_videos}
        url_by_id = {str(v.get("video_id", "") or ""): str(v.get("url", "") or "") for v in board_videos}

        candidates: list[dict[str, Any]] = []
        for video in board_videos:
            video_id = str(video.get("video_id", "") or "")
            if not video_id:
                continue
            artifact_path = runtime.transcript_dir / f"{video_id}.json"
            if not artifact_path.exists():
                continue
            try:
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            segments = artifact.get("segments", [])
            window, step = 6, 3
            for i in range(0, max(1, len(segments) - window + 1), step):
                chunk_segs = segments[i : i + window]
                if not chunk_segs:
                    continue
                chunk_text = " ".join(str(s.get("text", "")) for s in chunk_segs)
                chunk_tokens = set(re.findall(r"[a-z0-9]+", chunk_text.lower()))
                score = len(q_tokens & chunk_tokens)
                if score == 0:
                    continue
                candidates.append(
                    {
                        "video_id": video_id,
                        "content": chunk_text,
                        "start_seconds": float(chunk_segs[0].get("start", 0.0)),
                        "score": score,
                        "title": title_by_id.get(video_id, "Indexed transcript"),
                        "url": url_by_id.get(video_id, ""),
                    }
                )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        seen_windows: set[str] = set()
        chunks_per_video: dict[str, int] = {}
        normalized: list[dict[str, Any]] = []
        for chunk in candidates:
            video_id = chunk["video_id"]
            # Deduplicate overlapping windows within the same minute of the same video
            window_key = f"{video_id}_{int(chunk['start_seconds'] // 60)}"
            if window_key in seen_windows:
                continue
            seen_windows.add(window_key)
            # Cap at 2 chunks per video so results span multiple sources
            if chunks_per_video.get(video_id, 0) >= 2:
                continue
            chunks_per_video[video_id] = chunks_per_video.get(video_id, 0) + 1
            start_seconds = chunk["start_seconds"]
            normalized.append(
                {
                    "title": chunk["title"],
                    "url": chunk["url"],
                    "content": chunk["content"],
                    "reference_id": "",
                    "chunk_id": f"{video_id}_{int(start_seconds)}",
                    "video_id": video_id,
                    "start_seconds": start_seconds,
                    "embed_url": self._youtube_embed_url(video_id, start_seconds) if video_id else "",
                    "source_url": yt_watch_url(video_id, start_seconds) if video_id else chunk["url"],
                    "start_label": seconds_to_label(int(start_seconds)),
                }
            )
            if len(normalized) >= top_k:
                break
        return normalized

    async def _query_board(self, board_id: int, runtime: BoardRuntime, question: str, mode: str, *, allow_empty: bool) -> dict[str, Any]:
        """Run a board-scoped Fast GraphRAG query and normalize the answer payload.

        Fast GraphRAG returns typed context objects instead of the nested
        query-data dictionary TubeMind used previously. This
        method keeps the route and note-storage contract unchanged by converting
        retrieved graph chunks back into the same source chunk dictionaries the
        UI already renders, using insertion metadata to recover YouTube titles,
        URLs, thumbnails, and video ids.
        """

        board_videos = list_board_videos(board_id)
        if not board_videos:
            return {"question": question, "mode": mode, "answer": "", "chunks": []}

        from fast_graphrag import QueryParam

        try:
            result = await self._run_coro_on_rag_loop(runtime.rag.async_query(question, params=QueryParam(with_references=True)))
        except Exception:
            if allow_empty:
                return {"question": question, "mode": mode, "answer": "", "chunks": []}
            raise

        answer = str(getattr(result, "response", "") or "").strip()
        context = getattr(result, "context", None)
        chunks = list(getattr(context, "chunks", []) or [])
        if not chunks and ((not answer) or answer.lower() in {"none", "null"}):
            if allow_empty:
                return {"question": question, "mode": mode, "answer": "", "chunks": []}
            raise RuntimeError("No transcript chunks matched that note yet.")

        title_by_url = {str(item.get("url", "") or ""): str(item.get("title", "") or "Indexed transcript") for item in board_videos}
        normalized_chunks: list[dict[str, Any]] = []
        for chunk_item in chunks:
            chunk = chunk_item[0] if isinstance(chunk_item, tuple) else chunk_item
            metadata = dict(getattr(chunk, "metadata", {}) or {})
            content = str(getattr(chunk, "content", "") or "").strip()
            file_path = str(metadata.get("url", "") or "").strip()
            video_id = str(metadata.get("video_id", "") or "").strip() or self._video_id_from_url(file_path)
            start_seconds = self._find_chunk_start_seconds(runtime, video_id, content)
            chunk_id = str(getattr(chunk, "id", "") or "").strip()
            normalized_chunks.append(
                {
                    "title": str(metadata.get("title", "") or "") or title_by_url.get(file_path, file_path or "Indexed transcript"),
                    "url": file_path,
                    "content": content,
                    "reference_id": chunk_id,
                    "chunk_id": chunk_id,
                    "video_id": video_id,
                    "start_seconds": start_seconds,
                    "embed_url": self._youtube_embed_url(video_id, start_seconds) if video_id else "",
                    "source_url": yt_watch_url(video_id, start_seconds) if video_id else file_path,
                    "start_label": seconds_to_label(int(start_seconds)),
                }
            )
        return {"question": question, "mode": mode, "answer": answer, "chunks": normalized_chunks}


_user_apps: dict[str, TubeMindApp] = {}
_user_locks: dict[str, asyncio.Lock] = {}


async def get_user_app(user_id: str) -> TubeMindApp:
    """Return the singleton runtime for one authenticated user."""

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
"""Board-aware TubeMind runtime services."""

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
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

import httpx
from openai import AsyncOpenAI
from youtube_transcript_api import NoTranscriptFound, TooManyRequests, YouTubeRequestFailed, YouTubeTranscriptApi

from tubemind.auth import (
    create_board,
    create_board_note,
    get_board_for_user,
    list_board_notes,
    list_board_videos,
    list_boards,
    replace_note_chunks,
    save_note_queries,
    set_active_board,
    update_board,
    upsert_board_videos,
)
from tubemind.config import (
    APP_ROOT,
    COOKIE_BROWSER_LABELS,
    DEFAULT_QUERY_MODE,
    MAX_VIDEOS_DEFAULT,
    MIN_SECONDS_DEFAULT,
    QUERY_MODES,
    TRANSCRIPT_CANDIDATE_PADDING,
    TRANSCRIPT_RETRY_ATTEMPTS,
    TRANSCRIPT_RETRY_BASE_DELAY,
    TRANSCRIPT_REQUEST_DELAY_SECONDS,
    TRANSCRIPTAPI_BASE_URL,
    YOUTUBE_SEARCH_URL,
    YOUTUBE_VIDEOS_URL,
)
from tubemind.models import BoardRuntime, BoardWorkspace, YouTubeVideo, iso8601_duration_to_seconds, now_ms, seconds_to_label, yt_watch_url


class TubeMindApp:
    """Own per-user board runtimes, OpenAI calls, and retrieval helpers."""

    def __init__(self, user_id: str) -> None:
        """Create the per-user container that backs every board action."""

        self.user_id = user_id
        self._user_root = APP_ROOT / "users" / user_id
        self._boards_root = self._user_root / "boards"
        self._boards_root.mkdir(parents=True, exist_ok=True)
        self._openai = AsyncOpenAI()
        self._llm_model = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")
        self._board_runtimes: dict[int, BoardRuntime] = {}
        self._rag_loop = asyncio.new_event_loop()
        self._rag_loop_ready = threading.Event()
        self._rag_thread: Optional[threading.Thread] = None
        self.lock = threading.RLock()
        self._start_rag_runtime()

    async def startup(self) -> None:
        """Keep the async service contract without eager board initialization."""

        return None

    async def shutdown(self) -> None:
        """Finalize initialized board corpora and stop the dedicated RAG loop."""

        for runtime in list(self._board_runtimes.values()):
            if runtime.rag is not None:
                try:
                    await self._run_coro_on_rag_loop(runtime.rag.finalize_storages())
                except Exception:
                    pass
        if self._rag_thread and self._rag_thread.is_alive():
            self._rag_loop.call_soon_threadsafe(self._rag_loop.stop)
            self._rag_thread.join(timeout=5)

    def _start_rag_runtime(self) -> None:
        """Start the background asyncio loop used for LightRAG operations."""

        self._rag_thread = threading.Thread(target=self._run_rag_loop, name=f"tubemind-rag-{self.user_id}", daemon=True)
        self._rag_thread.start()
        if not self._rag_loop_ready.wait(timeout=5):
            raise RuntimeError("TubeMind could not start the LightRAG worker loop.")

    def _run_rag_loop(self) -> None:
        """Run the dedicated event loop until shutdown."""

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
        """Submit one coroutine to the LightRAG loop."""

        if not self._rag_thread or not self._rag_thread.is_alive():
            raise RuntimeError("TubeMind knowledge-base worker is not running.")
        return asyncio.run_coroutine_threadsafe(coro, self._rag_loop)

    async def _run_coro_on_rag_loop(self, coro):
        """Await work scheduled onto the LightRAG loop."""

        try:
            future = self._submit_coro_to_rag_loop(coro)
        except Exception:
            coro.close()
            raise
        return await asyncio.wrap_future(future)

    async def _create_rag(self, working_dir: Path):
        """Create a LightRAG instance rooted at one board directory."""

        from lightrag import LightRAG
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed

        llm_model = partial(openai_complete_if_cache, self._llm_model)
        return LightRAG(
            working_dir=str(working_dir),
            llm_model_func=llm_model,
            embedding_func=openai_embed,
        )

    async def _get_board_runtime(self, board_id: int) -> BoardRuntime:
        """Return the lazily initialized runtime for one board."""

        with self.lock:
            cached = self._board_runtimes.get(board_id)
        if cached is not None:
            return cached

        board_root = self._boards_root / str(board_id)
        runtime = BoardRuntime(
            board_id=board_id,
            working_dir=board_root / "rag_storage",
            transcript_dir=board_root / "transcripts",
        )
        runtime.working_dir.mkdir(parents=True, exist_ok=True)
        runtime.transcript_dir.mkdir(parents=True, exist_ok=True)
        runtime.rag = await self._run_coro_on_rag_loop(self._create_rag(runtime.working_dir))
        await self._run_coro_on_rag_loop(runtime.rag.initialize_storages())
        with self.lock:
            self._board_runtimes[board_id] = runtime
        return runtime

    def build_workspace(self, active_board_id: int | None, *, notice: str = "", warning: str = "") -> BoardWorkspace:
        """Assemble the sidebar and active board payload used by the UI."""

        boards = list_boards(self.user_id)
        active_board = get_board_for_user(self.user_id, active_board_id)
        notes = list_board_notes(int(active_board["id"])) if active_board else []
        return BoardWorkspace(boards=boards, active_board=active_board, notes=notes, notice=notice, warning=warning)

    async def create_empty_board(self) -> BoardWorkspace:
        """Create an empty board and make it the active workspace."""

        board = create_board(self.user_id, "Untitled board", "", "", "idle")
        set_active_board(self.user_id, int(board["id"]))
        return self.build_workspace(int(board["id"]), notice="Created a new board.")

    async def answer_question(self, board_id: int | None, question: str, mode: str = DEFAULT_QUERY_MODE) -> BoardWorkspace:
        """Create a new note by reusing or expanding the selected board corpus."""

        question_text = str(question or "").strip()
        if not question_text:
            raise ValueError("Enter a question to create a note.")
        if mode not in QUERY_MODES:
            mode = DEFAULT_QUERY_MODE

        board = get_board_for_user(self.user_id, board_id) if board_id else None
        if board is None:
            board = create_board(self.user_id, question_text, question_text, "", "working")
        board_id_int = int(board["id"])
        set_active_board(self.user_id, board_id_int)

        notes = list_board_notes(board_id_int)
        if notes:
            fit = await self._assess_topic_fit(board, notes, question_text)
            if not fit["is_fit"]:
                return self.build_workspace(board_id_int, warning=fit["warning"])

        update_board(board_id_int, status="working", updated_at=now_ms())
        runtime = await self._get_board_runtime(board_id_int)
        initial = await self._query_board(board_id_int, runtime, question_text, mode, allow_empty=True)
        plan = await self._plan_research(board, notes, question_text, initial)
        queries = list(plan.get("queries") or [])
        if not initial.get("chunks") and not queries:
            queries = self._fallback_youtube_queries(board, question_text)

        if queries:
            await self._expand_board_corpus(board_id_int, runtime, queries)

        result = initial
        if queries or not result.get("chunks") or not str(result.get("answer", "") or "").strip():
            result = await self._query_board(board_id_int, runtime, question_text, mode, allow_empty=False)

        answer_text = str(result.get("answer") or "").strip()
        if not result.get("chunks") and not answer_text:
            update_board(board_id_int, status="error")
            raise RuntimeError("TubeMind could not find enough transcript evidence for that note.")

        note = create_board_note(
            board_id=board_id_int,
            question=question_text,
            answer=answer_text or "TubeMind found evidence but could not synthesize a final answer.",
            query_mode=mode,
        )
        save_note_queries(board_id_int, int(note["id"]), queries)
        replace_note_chunks(int(note["id"]), result.get("chunks", []))
        update_board(board_id_int, status="ready", last_question_at=now_ms(), updated_at=now_ms())
        await self._refresh_board_summary(board_id_int)
        return self.build_workspace(board_id_int, notice="Added a new note to the board.")

    async def _assess_topic_fit(self, board: dict[str, Any], notes: list[dict[str, Any]], question: str) -> dict[str, Any]:
        """Keep follow-up notes near the board topic instead of silently drifting."""

        system_prompt = (
            "Respond with JSON only: {\"is_fit\": boolean, \"warning\": string}. "
            "Mark obviously different topics as not fitting, but allow natural follow-up questions."
        )
        prompt = json.dumps(
            {
                "board_title": board.get("title", ""),
                "topic_anchor": board.get("topic_anchor", ""),
                "recent_questions": [str(item.get("question", "") or "") for item in notes[-4:]],
                "new_question": question,
            }
        )
        result = await self._llm_json(system_prompt, prompt)
        if isinstance(result, dict) and "is_fit" in result:
            return {
                "is_fit": bool(result.get("is_fit")),
                "warning": str(result.get("warning", "") or "").strip() or "That question looks like a different topic. Start a new board for it instead.",
            }

        haystack = " ".join([str(board.get("topic_anchor", "") or ""), *[str(item.get("question", "") or "") for item in notes[-4:]]]).casefold()
        overlap = {
            token
            for token in re.findall(r"[a-z0-9]+", question.casefold())
            if len(token) > 2 and token in haystack
        }
        return {
            "is_fit": bool(overlap),
            "warning": "That question looks like a different topic. Start a new board for it instead.",
        }

    async def _plan_research(
        self,
        board: dict[str, Any],
        notes: list[dict[str, Any]],
        question: str,
        initial: dict[str, Any],
    ) -> dict[str, Any]:
        """Decide whether the existing board corpus is enough for this note."""

        board_videos = list_board_videos(int(board["id"]))
        if not board_videos:
            return {"queries": self._fallback_youtube_queries(board, question)}

        system_prompt = (
            "Respond with JSON only using keys needs_more (boolean), rationale (string), "
            "queries (array of objects with query and reason). Generate 1-3 YouTube queries only when more evidence is needed."
        )
        prompt = json.dumps(
            {
                "board_title": board.get("title", ""),
                "topic_anchor": board.get("topic_anchor", ""),
                "recent_questions": [str(item.get("question", "") or "") for item in notes[-4:]],
                "video_titles": [str(item.get("title", "") or "") for item in board_videos[-10:]],
                "question": question,
                "draft_answer": str(initial.get("answer", "") or ""),
                "chunk_excerpts": [str(chunk.get("content", "") or "")[:240] for chunk in initial.get("chunks", [])[:4]],
            }
        )
        result = await self._llm_json(system_prompt, prompt)
        if not isinstance(result, dict):
            return {"queries": [] if initial.get("chunks") else self._fallback_youtube_queries(board, question)}

        queries = []
        for item in list(result.get("queries") or [])[:3]:
            query_text = str(item.get("query", "") or "").strip()
            if query_text:
                queries.append({"query": query_text, "reason": str(item.get("reason", "") or "").strip()})
        return {"queries": queries if bool(result.get("needs_more")) else []}

    async def _refresh_board_summary(self, board_id: int) -> None:
        """Apply the board-title rules after each successful note insertion."""

        notes = list_board_notes(board_id)
        if not notes:
            update_board(board_id, title="Untitled board", summary="", topic_anchor="")
            return
        if len(notes) == 1:
            first = str(notes[0].get("question", "") or "").strip()
            update_board(board_id, title=first, summary="", topic_anchor=first)
            return
        if len(notes) == 2:
            update_board(
                board_id,
                title=f'{notes[0].get("question", "")} / {notes[1].get("question", "")}',
                summary="",
                topic_anchor=str(notes[0].get("question", "") or "").strip(),
            )
            return

        result = await self._llm_json(
            "Respond with JSON only using keys title, summary, topic_anchor. Keep the title short and the summary to one or two sentences.",
            json.dumps(
                {
                    "questions": [str(item.get("question", "") or "") for item in notes[-6:]],
                    "answers": [str(item.get("answer", "") or "")[:350] for item in notes[-6:]],
                }
            ),
        )
        if isinstance(result, dict) and str(result.get("title", "") or "").strip():
            update_board(
                board_id,
                title=str(result.get("title", "") or "").strip(),
                summary=str(result.get("summary", "") or "").strip(),
                topic_anchor=str(result.get("topic_anchor", "") or "").strip() or str(notes[0].get("question", "") or "").strip(),
            )
            return

        fallback = " / ".join(str(item.get("question", "") or "").strip() for item in notes[:3])
        update_board(
            board_id,
            title=fallback,
            summary="This board groups related questions answered from a shared YouTube research corpus.",
            topic_anchor=str(notes[0].get("question", "") or "").strip(),
        )

    async def _llm_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        """Ask the configured OpenAI model for a small JSON object."""

        try:
            response = await self._openai.responses.create(
                model=self._llm_model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            return None

        text = str(getattr(response, "output_text", "") or "").strip()
        if not text:
            return None
        try:
            payload = json.loads(text)
            return payload if isinstance(payload, dict) else None
        except Exception:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
                return payload if isinstance(payload, dict) else None
            except Exception:
                return None

    def _fallback_youtube_queries(self, board: dict[str, Any], question: str) -> list[dict[str, str]]:
        """Generate deterministic search queries when the planning model fails."""

        anchor = str(board.get("topic_anchor", "") or "").strip()
        queries = [{"query": question, "reason": "Direct search for the current note question."}]
        if anchor and anchor.casefold() not in question.casefold():
            queries.append({"query": f"{anchor} {question}".strip(), "reason": "Keep the search anchored to the board topic."})
        return queries[:2]

    async def youtube_search(self, query: str, *, max_videos: int, min_seconds: int, order: str) -> list[YouTubeVideo]:
        """Search YouTube and normalize the result list for indexing.

        Hosted deployments are much more reliable when TubeMind targets videos
        that already advertise captions and allow embedding, because transcript
        fallbacks like yt-dlp are more likely to hit bot checks from cloud IPs.
        """

        key = os.environ["YOUTUBE_API_KEY"]
        max_results = str(min(max_videos, 25))
        search_variants = [
            {
                "part": "snippet",
                "type": "video",
                "maxResults": max_results,
                "q": query,
                "order": order,
                "videoCaption": "closedCaption",
                "videoEmbeddable": "true",
                "key": key,
            },
            {
                "part": "snippet",
                "type": "video",
                "maxResults": max_results,
                "q": query,
                "order": order,
                "videoEmbeddable": "true",
                "key": key,
            },
        ]

        video_ids: list[str] = []
        seen_video_ids: set[str] = set()
        async with httpx.AsyncClient(timeout=30) as client:
            for params in search_variants:
                response = await client.get(YOUTUBE_SEARCH_URL, params=params)
                data = response.json()
                if response.status_code != 200:
                    raise RuntimeError(f"YouTube search failed: {data}")
                for item in data.get("items", []):
                    video_id = str(item.get("id", {}).get("videoId") or "").strip()
                    if not video_id or video_id in seen_video_ids:
                        continue
                    seen_video_ids.add(video_id)
                    video_ids.append(video_id)
                if len(video_ids) >= min(max_videos, 12):
                    break

        if not video_ids:
            return []

        params2 = {"part": "snippet,contentDetails", "id": ",".join(video_ids), "key": key}
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(YOUTUBE_VIDEOS_URL, params=params2)
            data2 = response.json()
            if response.status_code != 200:
                raise RuntimeError(f"YouTube videos.list failed: {data2}")

        videos: list[YouTubeVideo] = []
        for item in data2.get("items", []):
            duration = iso8601_duration_to_seconds(str((item.get("contentDetails") or {}).get("duration", "") or ""))
            if duration < min_seconds:
                continue
            snippet = item.get("snippet", {}) or {}
            thumbs = snippet.get("thumbnails", {}) or {}
            thumbnail = (thumbs.get("medium") or {}).get("url") or (thumbs.get("high") or {}).get("url") or (thumbs.get("default") or {}).get("url") or ""
            video_id = str(item.get("id", "") or "").strip()
            videos.append(
                YouTubeVideo(
                    video_id=video_id,
                    title=str(snippet.get("title", "") or "").strip(),
                    channel_title=str(snippet.get("channelTitle", "") or "").strip(),
                    published_at=str(snippet.get("publishedAt", "") or "").strip(),
                    thumbnail=thumbnail,
                    duration_sec=duration,
                    url=yt_watch_url(video_id),
                )
            )
        return videos[:max_videos]

    def _transcript_candidate_pool(self, max_videos: int) -> int:
        """Pad the initial result pool to offset transcript failures."""

        raw_pad = str(os.environ.get("YOUTUBE_TRANSCRIPT_CANDIDATE_PADDING", TRANSCRIPT_CANDIDATE_PADDING))
        try:
            pad = int(raw_pad)
        except ValueError:
            pad = TRANSCRIPT_CANDIDATE_PADDING
        return min(25, max_videos + max(0, min(8, pad)))

    def _transcript_request_delay(self) -> float:
        """Return the configured pause between transcript fetches."""

        raw = str(os.environ.get("YOUTUBE_TRANSCRIPT_REQUEST_DELAY_SECONDS", TRANSCRIPT_REQUEST_DELAY_SECONDS))
        try:
            delay = float(raw)
        except ValueError:
            delay = TRANSCRIPT_REQUEST_DELAY_SECONDS
        return max(0.0, min(10.0, delay))

    def _normalize_alignment_text(self, text: str) -> str:
        """Normalize transcript text for chunk-to-timestamp alignment."""

        cleaned = re.sub(r"\s+", " ", text or "").strip().lower()
        return re.sub(r"[^a-z0-9 ]+", "", cleaned)

    def _transcript_cache_path(self, runtime: BoardRuntime, video_id: str) -> Path:
        """Return the board-local transcript artifact path for one video."""

        return runtime.transcript_dir / f"{video_id}.json"

    def _build_clean_transcript_artifact(self, video: YouTubeVideo, segments: list[dict[str, Any]]) -> dict[str, Any]:
        """Create the clean transcript text plus a timing sidecar artifact."""

        clean_parts: list[str] = []
        normalized_parts: list[str] = []
        artifact_segments: list[dict[str, Any]] = []
        cursor = 0
        for segment in segments:
            cleaned_text = re.sub(r"\s+", " ", str(segment.get("text", "") or "")).strip()
            if not cleaned_text:
                continue
            clean_parts.append(cleaned_text)
            normalized = self._normalize_alignment_text(cleaned_text)
            if not normalized:
                continue
            if normalized_parts:
                cursor += 1
            offset_start = cursor
            normalized_parts.append(normalized)
            cursor += len(normalized)
            artifact_segments.append({"start": float(segment.get("start", 0.0) or 0.0), "text": cleaned_text, "offset_start": offset_start, "offset_end": cursor})
        return {
            "video_id": video.video_id,
            "url": video.url,
            "clean_text": " ".join(clean_parts),
            "normalized_text": " ".join(normalized_parts),
            "segments": artifact_segments,
        }

    def _save_transcript_artifact(self, runtime: BoardRuntime, video: YouTubeVideo, segments: list[dict[str, Any]]) -> str:
        """Persist the timing sidecar and return the clean transcript text."""

        artifact = self._build_clean_transcript_artifact(video, segments)
        self._transcript_cache_path(runtime, video.video_id).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
        return str(artifact.get("clean_text", "") or "").strip()

    def _video_id_from_url(self, url: str) -> str:
        """Extract a YouTube video id from a stored watch URL."""

        try:
            return str(parse_qs(urlparse(url).query).get("v", [""])[0] or "").strip()
        except Exception:
            return ""

    def _youtube_embed_url(self, video_id: str, start_seconds: float) -> str:
        """Build an embeddable YouTube URL anchored to a chunk timestamp."""

        return f"https://www.youtube.com/embed/{video_id}?start={max(0, int(start_seconds))}&rel=0"

    def _find_chunk_start_seconds(self, runtime: BoardRuntime, video_id: str, chunk_text: str) -> float:
        """Approximate a retrieved chunk's start time from the saved transcript artifact."""

        artifact_path = self._transcript_cache_path(runtime, video_id)
        if not artifact_path.exists():
            return 0.0
        try:
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        except Exception:
            return 0.0
        normalized_chunk = self._normalize_alignment_text(chunk_text)
        normalized_transcript = str(artifact.get("normalized_text", "") or "")
        if not normalized_chunk or not normalized_transcript:
            return 0.0
        offset = normalized_transcript.find(normalized_chunk)
        if offset < 0:
            excerpt = normalized_chunk[:120]
            if excerpt:
                offset = normalized_transcript.find(excerpt)
        if offset < 0:
            return 0.0
        for segment in artifact.get("segments", []) or []:
            if int(segment.get("offset_end", 0) or 0) >= offset:
                return float(segment.get("start", 0.0) or 0.0)
        return 0.0

    def _transcript_request_kwargs(self) -> dict[str, Any]:
        """Return optional cookie-file settings for transcript requests."""

        cookies_file = str(os.environ.get("YOUTUBE_TRANSCRIPT_COOKIES_FILE", "")).strip().strip("'").strip('"')
        return {"cookies": cookies_file} if cookies_file else {}

    def _transcript_api_key(self) -> str:
        """Return the configured TranscriptAPI key when present."""

        return str(os.environ.get("TRANSCRIPTAPI_API_KEY", "")).strip().strip("'").strip('"')

    def _summarize_transcript_failures(self, failures: list[str]) -> str:
        """Compress repeated transcript fetch failures into a readable warning."""

        seen: set[str] = set()
        unique_failures: list[str] = []
        for failure in failures:
            cleaned = str(failure or "").strip()
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            unique_failures.append(cleaned)

        if not unique_failures:
            return "TubeMind found videos, but transcript fetching failed before any evidence could be indexed."

        preview = "\n\n".join(unique_failures[:3])
        return (
            "TubeMind found videos, but could not fetch any usable transcripts for them.\n\n"
            f"{preview}\n\n"
            "On hosted deployments this usually means TranscriptAPI auth is invalid, quota is exhausted, or the candidate videos do not have captions."
        )

    def _summarize_indexing_failures(self, failures: list[dict[str, str]]) -> str:
        """Compress document indexing failures into a readable warning."""

        previews: list[str] = []
        seen: set[str] = set()
        for failure in failures:
            title = str(failure.get("title", "") or "Indexed transcript").strip()
            reason = str(failure.get("reason", "") or "unknown indexing error").strip()
            line = f"{title}: {reason}"
            if line in seen:
                continue
            seen.add(line)
            previews.append(line)

        if not previews:
            return "TubeMind fetched transcripts, but indexing them into the board failed before any evidence became available."

        return (
            "TubeMind fetched transcripts, but indexing them into the board failed before any evidence became available.\n\n"
            f"{chr(10).join(previews[:3])}"
        )

    def _is_transcript_rate_limited(self, exc: Exception) -> bool:
        """Detect transcript rate-limit conditions across providers."""

        if isinstance(exc, TooManyRequests):
            return True
        return "429" in str(exc).lower() or "too many requests" in str(exc).lower()

    def _should_retry_transcript_error(self, exc: Exception) -> bool:
        """Decide whether a transcript failure is transient enough to retry."""

        if self._is_transcript_rate_limited(exc):
            return True
        if isinstance(exc, YouTubeRequestFailed):
            text = str(exc).lower()
            return "timed out" in text or "temporarily unavailable" in text
        return False

    def _describe_transcript_error(self, exc: Exception, *, using_cookies: bool) -> str:
        """Convert transcript exceptions into readable diagnostics."""

        message = f"{type(exc).__name__}: {str(exc)}"
        if self._is_transcript_rate_limited(exc) and not using_cookies:
            return f"{message}\nHint: set YOUTUBE_TRANSCRIPT_COOKIES_FILE to reduce transcript 429s."
        return message

    def _extract_transcriptapi_error(self, payload: Any) -> str:
        """Normalize TranscriptAPI error payloads into one short string."""

        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, dict):
            return str(detail.get("message") or detail.get("detail") or detail)
        if detail:
            return str(detail)
        if isinstance(payload, dict):
            return str(payload.get("message") or payload)
        return str(payload)

    def _fetch_transcript_with_transcriptapi(self, video: YouTubeVideo) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
        """Try TranscriptAPI before falling back to other transcript sources."""

        api_key = self._transcript_api_key()
        if not api_key:
            return None, None
        headers = {"Authorization": f"Bearer {api_key}"}
        params = {
            "video_url": video.video_id,
            "format": "json",
            "include_timestamp": "true",
        }
        last_err = "unknown TranscriptAPI error"
        retry_delay = 1.0
        with httpx.Client(timeout=30.0) as client:
            for _ in range(3):
                response = client.get(f"{TRANSCRIPTAPI_BASE_URL}/youtube/transcript", params=params, headers=headers)
                if response.status_code == 200:
                    payload = response.json()
                    cues = payload.get("transcript", []) if isinstance(payload, dict) else []
                    segments = [{"start": float(cue.get("start", 0.0) or 0.0), "text": str(cue.get("text", "")).strip()} for cue in cues if str(cue.get("text", "")).strip()]
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
        """Parse a VTT timestamp into float seconds."""

        clean = value.strip().replace(",", ".")
        parts = clean.split(":")
        if len(parts) == 3:
            hours, minutes, seconds = parts
            return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        if len(parts) == 2:
            minutes, seconds = parts
            return int(minutes) * 60 + float(seconds)
        return float(clean)

    def _parse_vtt_segments(self, text: str) -> list[dict[str, Any]]:
        """Parse subtitle cues from a VTT file."""

        lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        segments: list[dict[str, Any]] = []
        index = 0
        while index < len(lines):
            line = lines[index].strip()
            if "-->" not in line:
                index += 1
                continue
            start_raw = line.split("-->", 1)[0].strip().split(" ")[0]
            index += 1
            cue_lines: list[str] = []
            while index < len(lines) and lines[index].strip():
                cue_lines.append(lines[index].strip())
                index += 1
            cue_text = html.unescape(re.sub(r"<[^>]+>", "", " ".join(cue_lines)).strip())
            if cue_text:
                segments.append({"start": self._parse_seconds_label(start_raw), "text": cue_text})
        return segments

    def _parse_json3_segments(self, text: str) -> list[dict[str, Any]]:
        """Parse subtitle cues from a YouTube json3 subtitle file."""

        payload = json.loads(text)
        segments: list[dict[str, Any]] = []
        for event in payload.get("events", []) or []:
            parts = event.get("segs") or []
            if not parts:
                continue
            cue_text = html.unescape("".join(str(part.get("utf8", "")) for part in parts)).replace("\n", " ").strip()
            if cue_text:
                segments.append({"start": float(event.get("tStartMs", 0) or 0) / 1000.0, "text": cue_text})
        return segments

    def _read_subtitle_segments(self, path: Path) -> list[dict[str, Any]]:
        """Load subtitle cues from either json3 or vtt files."""

        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json3":
            return self._parse_json3_segments(text)
        if path.suffix.lower() == ".vtt":
            return self._parse_vtt_segments(text)
        raise RuntimeError(f"Unsupported subtitle format: {path.name}")

    def _yt_dlp_cookie_sources(self) -> list[tuple[str, dict[str, Any]]]:
        """Return the ordered yt-dlp cookie strategies to try."""

        sources: list[tuple[str, dict[str, Any]]] = [("yt-dlp", {})]
        cookie_file = str(os.environ.get("YOUTUBE_TRANSCRIPT_COOKIES_FILE", "")).strip().strip("'").strip('"')
        if cookie_file:
            sources.append(("yt-dlp + cookie file", {"cookiefile": cookie_file}))
        browsers_raw = str(os.environ.get("YOUTUBE_COOKIES_BROWSER", "")).strip().strip("'").strip('"')
        for browser in [value.strip().lower() for value in browsers_raw.split(",") if value.strip()]:
            sources.append((f"yt-dlp + {COOKIE_BROWSER_LABELS.get(browser, browser.title())} cookies", {"cookiesfrombrowser": (browser, None, None, None)}))
        return sources

    class _QuietYTDLPLogger:
        """Suppress yt-dlp log spam inside the UI request cycle."""

        def debug(self, msg: str) -> None:
            return None

        def warning(self, msg: str) -> None:
            return None

        def error(self, msg: str) -> None:
            return None

    def _fetch_transcript_with_ytdlp(self, video: YouTubeVideo) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
        """Use yt-dlp subtitle download as the last transcript fallback."""

        try:
            from yt_dlp import YoutubeDL
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
                        "logger": self._QuietYTDLPLogger(),
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
                last_err = f"{source_label}: {type(exc).__name__}: {exc}"
        return None, last_err

    def _fetch_transcript(self, video: YouTubeVideo) -> tuple[Optional[list[dict[str, Any]]], Optional[str]]:
        """Fetch transcript segments with layered provider fallbacks."""

        last_err: Optional[str] = None
        transcript_api_err: Optional[str] = None
        yt_dlp_err: Optional[str] = None
        prefer_transcriptapi = bool(self._transcript_api_key())
        if prefer_transcriptapi:
            segments, transcript_api_err = self._fetch_transcript_with_transcriptapi(video)
            if segments:
                return segments, None
            last_err = transcript_api_err

        request_kwargs = self._transcript_request_kwargs()
        for attempt in range(1, TRANSCRIPT_RETRY_ATTEMPTS + 1):
            try:
                try:
                    segments = YouTubeTranscriptApi.get_transcript(video.video_id, languages=("en",), **request_kwargs)
                except NoTranscriptFound:
                    segments = YouTubeTranscriptApi.get_transcript(video.video_id, **request_kwargs)
                if segments:
                    return segments, None
                last_err = "empty transcript payload"
            except Exception as exc:
                last_err = self._describe_transcript_error(exc, using_cookies=bool(request_kwargs.get("cookies")))
                if not self._should_retry_transcript_error(exc):
                    break
            if attempt < TRANSCRIPT_RETRY_ATTEMPTS:
                time.sleep(TRANSCRIPT_RETRY_BASE_DELAY * (2 ** (attempt - 1)))

        if not prefer_transcriptapi:
            segments, transcript_api_err = self._fetch_transcript_with_transcriptapi(video)
            if segments:
                return segments, None

        segments, yt_dlp_err = self._fetch_transcript_with_ytdlp(video)
        if segments:
            return segments, None
        errors = [err for err in (last_err, transcript_api_err, yt_dlp_err) if err]
        return None, "\n".join(errors) if errors else "unknown transcript error"

    def _youtube_video_id_from_doc_id(self, doc_id: str) -> str:
        """Extract the YouTube id from a LightRAG document id."""

        return doc_id.split(":", 1)[1].strip() if doc_id.startswith("youtube:") else ""

    def _extract_title_from_summary(self, summary: str) -> str:
        """Recover the transcript title from a LightRAG status summary."""

        match = re.search(r"(?m)^Title:\s*(.+)$", summary or "")
        return match.group(1).strip() if match else ""

    def _doc_item_key(self, doc_id: str, item: dict[str, str]) -> str:
        """Build a stable key for deduplicating document status rows."""

        return str(item.get("videoId") or item.get("url") or item.get("title") or doc_id)

    def _is_already_processed_duplicate(self, status_doc: Any) -> bool:
        """Detect duplicate-insert errors that mean the transcript already exists."""

        error_msg = str(getattr(status_doc, "error_msg", "") or "").lower()
        return "content already exists." in error_msg

    def _classify_doc_status_docs(self, docs: dict[str, Any], video_lookup: Optional[dict[str, YouTubeVideo]] = None) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        """Split LightRAG status rows into successful and failed documents."""

        from lightrag.base import DocStatus

        successful_map: dict[str, dict[str, str]] = {}
        failed_map: dict[str, dict[str, str]] = {}
        video_lookup = video_lookup or {}
        video_lookup_by_url = {video.url: video for video in video_lookup.values()}
        for doc_id, status_doc in docs.items():
            video_id = self._youtube_video_id_from_doc_id(doc_id)
            file_path = str(getattr(status_doc, "file_path", "") or "")
            video = video_lookup.get(video_id) or video_lookup_by_url.get(file_path)
            if video and not video_id:
                video_id = video.video_id
            title = video.title if video else self._extract_title_from_summary(str(getattr(status_doc, "content_summary", "") or ""))
            title = title or file_path or doc_id
            item = {
                "videoId": video_id,
                "title": title,
                "url": file_path or (video.url if video else yt_watch_url(video_id) if video_id else ""),
                "thumbnail": video.thumbnail if video else "",
            }
            key = self._doc_item_key(doc_id, item)
            status = getattr(status_doc, "status", None)
            if status == DocStatus.PROCESSED or self._is_already_processed_duplicate(status_doc):
                failed_map.pop(key, None)
                successful_map.setdefault(key, item)
            elif status == DocStatus.FAILED and key not in successful_map:
                failed_map.setdefault(key, {**item, "reason": f"Indexing failed: {str(getattr(status_doc, 'error_msg', '') or 'unknown error')}"})
        return list(successful_map.values()), list(failed_map.values())

    async def _get_docs_by_track_id(self, runtime: BoardRuntime, track_id: str) -> dict[str, Any]:
        """Read LightRAG document status rows for one insertion batch."""

        return await self._run_coro_on_rag_loop(runtime.rag.doc_status.get_docs_by_track_id(track_id))

    async def _expand_board_corpus(self, board_id: int, runtime: BoardRuntime, queries: list[dict[str, str]]) -> None:
        """Search, fetch, and index additional videos into one board corpus."""

        existing_ids = {str(item.get("video_id", "") or "").strip() for item in list_board_videos(board_id)}
        queued_ids = set(existing_ids)
        documents: list[str] = []
        ids: list[str] = []
        file_paths: list[str] = []
        indexed_videos: list[YouTubeVideo] = []
        origin_query_by_video_id: dict[str, str] = {}
        transcript_failures: list[str] = []

        for item in queries[:3]:
            query_text = str(item.get("query", "") or "").strip()
            if not query_text:
                continue
            videos = await self.youtube_search(
                query_text,
                max_videos=self._transcript_candidate_pool(MAX_VIDEOS_DEFAULT),
                min_seconds=MIN_SECONDS_DEFAULT,
                order="relevance",
            )
            if not videos:
                transcript_failures.append(f'No caption-friendly YouTube results were found for query "{query_text}".')
                continue
            for video in videos:
                if video.video_id in queued_ids:
                    continue
                queued_ids.add(video.video_id)
                segments, transcript_error = self._fetch_transcript(video)
                if not segments:
                    transcript_failures.append(f"{video.title}: {transcript_error or 'unknown transcript error'}")
                    time.sleep(self._transcript_request_delay())
                    continue
                transcript = self._save_transcript_artifact(runtime, video, segments)
                if not transcript.strip():
                    transcript_failures.append(f"{video.title}: fetched transcript was empty after normalization")
                    time.sleep(self._transcript_request_delay())
                    continue
                documents.append(transcript)
                ids.append(f"youtube:{video.video_id}")
                file_paths.append(video.url)
                indexed_videos.append(video)
                origin_query_by_video_id[video.video_id] = query_text
                time.sleep(self._transcript_request_delay())
                if len(documents) >= MAX_VIDEOS_DEFAULT:
                    break
            if len(documents) >= MAX_VIDEOS_DEFAULT:
                break

        if not documents:
            if transcript_failures:
                raise RuntimeError(self._summarize_transcript_failures(transcript_failures))
            return

        track_id = await self._run_coro_on_rag_loop(runtime.rag.ainsert(documents, ids=ids, file_paths=file_paths))
        docs = await self._get_docs_by_track_id(runtime, track_id)
        successful, failed = self._classify_doc_status_docs(docs, {video.video_id: video for video in indexed_videos})
        if not successful:
            if failed:
                raise RuntimeError(self._summarize_indexing_failures(failed))
            return

        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in successful:
            video_id = str(item.get("videoId", "") or "")
            grouped.setdefault(origin_query_by_video_id.get(video_id, ""), []).append(
                {
                    "video_id": video_id,
                    "title": str(item.get("title", "") or ""),
                    "url": str(item.get("url", "") or ""),
                    "thumbnail": str(item.get("thumbnail", "") or ""),
                    "channel_title": next((video.channel_title for video in indexed_videos if video.video_id == video_id), ""),
                }
            )
        for origin_query, videos in grouped.items():
            upsert_board_videos(board_id, videos, origin_query=origin_query)

    async def _query_board(self, board_id: int, runtime: BoardRuntime, question: str, mode: str, *, allow_empty: bool) -> dict[str, Any]:
        """Run a board-scoped LightRAG query and normalize the answer payload."""

        board_videos = list_board_videos(board_id)
        if not board_videos:
            return {"question": question, "mode": mode, "answer": "", "chunks": []}

        from lightrag import QueryParam

        try:
            answer = str(await self._run_coro_on_rag_loop(runtime.rag.aquery(question, param=QueryParam(mode=mode, response_type="Multiple Paragraphs")))).strip()
            data = await self._run_coro_on_rag_loop(runtime.rag.aquery_data(question, param=QueryParam(mode=mode)))
        except Exception:
            if allow_empty:
                return {"question": question, "mode": mode, "answer": "", "chunks": []}
            raise

        chunks = list((data or {}).get("data", {}).get("chunks", []) or [])
        if not chunks and ((not answer) or answer.lower() in {"none", "null"}):
            if allow_empty:
                return {"question": question, "mode": mode, "answer": "", "chunks": []}
            raise RuntimeError("No transcript chunks matched that note yet.")

        title_by_url = {str(item.get("url", "") or ""): str(item.get("title", "") or "Indexed transcript") for item in board_videos}
        normalized_chunks: list[dict[str, Any]] = []
        for chunk in chunks:
            file_path = str(chunk.get("file_path", "") or "").strip()
            video_id = self._video_id_from_url(file_path)
            start_seconds = self._find_chunk_start_seconds(runtime, video_id, str(chunk.get("content", "") or ""))
            normalized_chunks.append(
                {
                    "title": title_by_url.get(file_path, file_path or "Indexed transcript"),
                    "url": file_path,
                    "content": str(chunk.get("content", "") or "").strip(),
                    "reference_id": str(chunk.get("reference_id", "") or "").strip(),
                    "chunk_id": str(chunk.get("chunk_id", "") or "").strip(),
                    "video_id": video_id,
                    "start_seconds": start_seconds,
                    "embed_url": self._youtube_embed_url(video_id, start_seconds) if video_id else "",
                    "source_url": yt_watch_url(video_id, start_seconds) if video_id else file_path,
                    "start_label": seconds_to_label(int(start_seconds)),
                }
            )
        return {"question": question, "mode": mode, "answer": answer, "chunks": normalized_chunks}


_user_apps: dict[str, TubeMindApp] = {}
_user_locks: dict[str, asyncio.Lock] = {}


async def get_user_app(user_id: str) -> TubeMindApp:
    """Return the singleton runtime for one authenticated user."""

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
