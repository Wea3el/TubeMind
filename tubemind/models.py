"""Core dataclasses and small shared helpers for TubeMind."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def now_ms() -> int:
    """Return a millisecond timestamp suitable for user-visible job ids."""

    return int(time.time() * 1000)


def iso8601_duration_to_seconds(duration: str) -> int:
    """Convert a YouTube ISO-8601 duration like `PT12M34S` into seconds."""

    if not duration or not isinstance(duration, str) or not duration.startswith("PT"):
        return 0
    match = re.match(r"^PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?$", duration)
    if not match:
        return 0
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    return hours * 3600 + minutes * 60 + seconds


def seconds_to_label(seconds: int) -> str:
    """Format whole seconds into a compact UI-friendly label."""

    if seconds <= 0:
        return ""
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def yt_watch_url(video_id: str, offset_seconds: Optional[float] = None) -> str:
    """Build a canonical YouTube watch URL, optionally anchored to a timestamp."""

    if offset_seconds is None:
        return f"https://www.youtube.com/watch?v={video_id}"
    return f"https://www.youtube.com/watch?v={video_id}&t={int(offset_seconds)}s"


@dataclass
class YouTubeVideo:
    """Normalized YouTube video metadata used throughout the app."""

    video_id: str
    title: str
    channel_title: str
    published_at: str
    thumbnail: str
    duration_sec: int
    url: str


@dataclass
class CorpusState:
    """Persisted per-user TubeMind state for the dashboard and corpus metadata."""

    youtube_indexed: bool = False
    youtube_seed_query: str = ""
    youtube_preferred_channels: List[str] = field(default_factory=list)
    youtube_excluded_channels: List[str] = field(default_factory=list)
    youtube_global_excluded_channels: List[str] = field(default_factory=list)
    youtube_preferred_only: bool = False
    youtube_video_ids: List[str] = field(default_factory=list)
    youtube_titles: List[str] = field(default_factory=list)
    youtube_urls: Dict[str, str] = field(default_factory=dict)
    youtube_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    youtube_skipped: List[Dict[str, Any]] = field(default_factory=list)
    job_active: bool = False
    job_id: str = ""
    job_stage: str = ""
    job_progress: int = 0
    job_total: int = 0
    job_message: str = ""
    _state_file: Path = field(default=None, repr=False, compare=False)

    @classmethod
    def load(cls, state_file: Path) -> "CorpusState":
        """Load the saved state for a user or initialize a new empty one."""

        if not state_file.exists():
            obj = cls()
            obj._state_file = state_file
            return obj
        data = json.loads(state_file.read_text(encoding="utf-8"))
        obj = cls(
            youtube_indexed=bool(data.get("youtube_indexed", False)),
            youtube_seed_query=str(data.get("youtube_seed_query", "")),
            youtube_preferred_channels=[str(x) for x in data.get("youtube_preferred_channels", [])],
            youtube_excluded_channels=[str(x) for x in data.get("youtube_excluded_channels", [])],
            youtube_global_excluded_channels=[str(x) for x in data.get("youtube_global_excluded_channels", [])],
            youtube_preferred_only=bool(data.get("youtube_preferred_only", False)),
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
        obj._state_file = state_file
        return obj

    def save(self) -> None:
        """Persist the current state snapshot to disk for later recovery."""

        self._state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "youtube_indexed": self.youtube_indexed,
            "youtube_seed_query": self.youtube_seed_query,
            "youtube_preferred_channels": self.youtube_preferred_channels,
            "youtube_excluded_channels": self.youtube_excluded_channels,
            "youtube_global_excluded_channels": self.youtube_global_excluded_channels,
            "youtube_preferred_only": self.youtube_preferred_only,
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
        self._state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
