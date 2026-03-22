"""Shared configuration and constants for TubeMind."""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
APP_ROOT = ROOT / ".local"
STATIC_ROOT = ROOT / "static"
CSS_FILE = STATIC_ROOT / "tubemind.css"

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"
TRANSCRIPTAPI_BASE_URL = "https://transcriptapi.com/api/v2"
HTMX_SSE_EXTENSION_URL = "https://cdn.jsdelivr.net/npm/htmx-ext-sse@2.2.2/sse.js"

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
MAX_RECOMMENDATIONS = 5
TRANSCRIPT_RETRY_ATTEMPTS = 3
TRANSCRIPT_RETRY_BASE_DELAY = 1.0
TRANSCRIPT_REQUEST_DELAY_SECONDS = 1.5
TRANSCRIPT_CANDIDATE_PADDING = 2
SSE_RETRY_MS = 1000


def load_environment() -> None:
    """Load required environment variables before the app starts serving requests."""

    load_dotenv(ROOT / ".env")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY was not found in backend/.env")
    if not os.environ.get("OPENAI_MODEL"):
        os.environ["OPENAI_MODEL"] = "gpt-4.1-nano"
    if not os.environ.get("YOUTUBE_API_KEY"):
        raise RuntimeError("YOUTUBE_API_KEY was not found in backend/.env")


load_environment()

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
SESSION_SECRET = os.environ.get("SESSION_SECRET") or secrets.token_hex(32)
BASE_URL = os.environ.get("BASE_URL", "http://localhost:5001")
REDIRECT_URI = f"{BASE_URL}/auth/callback"
