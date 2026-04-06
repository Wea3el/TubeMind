# TubeMind

TubeMind is a board-based research app for learning from YouTube. Instead of treating each question as a one-off search, it groups related questions into topic-bound boards, pulls transcript evidence from relevant videos, indexes that material with LightRAG, and turns each answer into a reusable note with linked source evidence.

The current app is a server-rendered FastHTML experience with a redesigned premium UI, persistent light/dark theme toggle, Google OAuth or demo auth, durable SQLite state, and Railway-ready deployment support.

## What TubeMind Does

- Creates a new board automatically from your first question.
- Keeps follow-up questions inside the same topic region instead of starting from scratch each time.
- Searches YouTube for caption-friendly, embeddable videos when the current board does not already have enough evidence.
- Fetches transcripts, normalizes them, caches them on disk, and indexes them into a per-board LightRAG knowledge base.
- Generates note answers backed by transcript chunks, then stores those notes so the board becomes more useful over time.
- Lets you open note detail pages with evidence excerpts, linked timestamps, and the original search queries that expanded the board.

## How It Works

1. A user asks a question in a board.
2. TubeMind first queries the existing board corpus.
3. If the board does not have enough evidence yet, TubeMind plans or falls back to YouTube search queries.
4. It searches YouTube for videos that are more likely to work well in hosted environments.
5. It fetches transcripts using layered fallbacks:
   - `TranscriptAPI`
   - `youtube-transcript-api`
   - `yt-dlp` subtitle download fallback
6. It stores cleaned transcript artifacts under the app data directory and indexes them into that board's LightRAG store.
7. It answers the question, stores the note, stores the source chunks, and refreshes the board summary over time.

Each board has its own transcript cache and LightRAG working directory, so follow-up notes stay grounded in the same topic instead of polluting a single global corpus.

## Stack

- [FastHTML](https://fastht.ml/docs/) for the server-rendered app and route layer
- [HTMX](https://htmx.org/) for incremental UI interactions
- [LightRAG](https://github.com/HKUDS/LightRAG) for retrieval and graph-backed indexing
- [OpenAI API](https://platform.openai.com/) for planning, synthesis, and answer generation
- [YouTube Data API v3](https://developers.google.com/youtube/v3) for video search and metadata
- [TranscriptAPI](https://transcriptapi.com/) plus transcript fallbacks for transcript acquisition
- SQLite for durable user, board, note, and evidence metadata
- `uv` for dependency management and running the app

## Product Structure

The main files are:

- [`tubemind/routes.py`](tubemind/routes.py): app factory, routes, theme bootstrap, auth guards, health endpoint
- [`tubemind/ui.py`](tubemind/ui.py): server-rendered UI builders for login, workspace, note detail, topbar, and theme toggle
- [`tubemind/services.py`](tubemind/services.py): board runtime orchestration, YouTube search, transcript fetching, indexing, retrieval, answer generation
- [`tubemind/auth.py`](tubemind/auth.py): Google OAuth helpers, demo auth, SQLite tables, board persistence
- [`tubemind/config.py`](tubemind/config.py): environment loading, app constants, path configuration
- [`static/tubemind.css`](static/tubemind.css): full visual system for light and dark themes
- [`tubemind/__main__.py`](tubemind/__main__.py): `python -m tubemind` entrypoint

## Local Development

### Prerequisites

- Python 3.12+
- [`uv`](https://docs.astral.sh/uv/)
- OpenAI API key
- YouTube Data API key
- TranscriptAPI key
- Google OAuth credentials if you want Google login locally

### Environment Variables

Create a `.env` file in the repo root. At minimum, local development needs:

```dotenv
OPENAI_API_KEY=your_openai_key
OPENAI_MODEL=gpt-4.1-nano
YOUTUBE_API_KEY=your_youtube_api_key
TRANSCRIPTAPI_API_KEY=your_transcriptapi_key
BASE_URL=http://localhost:5001
SESSION_SECRET=any-long-random-string
```

If you want Google OAuth locally, also set:

```dotenv
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret
```

Optional variables:

```dotenv
DEMO_AUTH_ENABLED=false
DEMO_USER_ID=demo-user
DEMO_USER_NAME=Coursework Demo
DEMO_USER_EMAIL=demo@tubemind.local
DEMO_USER_PICTURE=
TUBEMIND_DATA_DIR=.local
YOUTUBE_TRANSCRIPT_COOKIES_FILE=
YOUTUBE_COOKIES_BROWSER=
PORT=5001
```

### Run Locally

```bash
cd TubeMind
UV_CACHE_DIR=.local/uv-cache uv sync
UV_CACHE_DIR=.local/uv-cache uv run python -m tubemind
```

Open:

```text
http://127.0.0.1:5001
```

Stop the server with `Ctrl+C`.

## Authentication Modes

TubeMind supports two sign-in modes:

- Google OAuth:
  - best for normal usage
  - requires `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and a matching `BASE_URL`
- Demo auth:
  - best for coursework demos or simpler hosted deployments
  - enable with `DEMO_AUTH_ENABLED=true`
  - creates a synthetic local user session without Google sign-in

If both are configured, the login page shows both options.

## Data Model

TubeMind stores two kinds of state:

- SQLite app state:
  - users
  - boards
  - notes
  - board search queries
  - note evidence chunks
  - indexed video metadata
- Board filesystem state:
  - transcript artifacts
  - per-board LightRAG working directories

By default this lives under `TUBEMIND_DATA_DIR`. In production, that directory should be mounted to persistent storage.

## Railway Deployment

Railway is the recommended hosted path for this repo.

### Recommended Setup

1. Push the repo to GitHub.
2. Create a Railway service from the repo.
3. Add a volume to the same service.
4. Mount the volume at:

```text
/data/tubemind
```

5. Set:

```dotenv
TUBEMIND_DATA_DIR=/data/tubemind
```

That path is correct for the current production setup.

### Required Railway Variables

```dotenv
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4.1-nano
YOUTUBE_API_KEY=...
TRANSCRIPTAPI_API_KEY=...
BASE_URL=https://your-service.up.railway.app
SESSION_SECRET=choose-a-long-random-string
TUBEMIND_DATA_DIR=/data/tubemind
```

### Choose One Auth Mode

For Google OAuth:

```dotenv
GOOGLE_CLIENT_ID=...
GOOGLE_CLIENT_SECRET=...
DEMO_AUTH_ENABLED=false
```

For coursework/demo mode:

```dotenv
DEMO_AUTH_ENABLED=true
```

### Optional Hosted Transcript Variable

If `yt-dlp` needs cookies to get around YouTube bot checks on hosted infrastructure, add:

```dotenv
YOUTUBE_TRANSCRIPT_COOKIES_FILE=/data/tubemind/youtube-cookies.txt
```

Do not commit `youtube-cookies.txt` to GitHub. Upload it only to the mounted Railway volume.

### Health Check

The app exposes:

```text
/health
```

It returns:

```json
{"ok": true}
```

### Deploy Behavior

- The app reads Railway's injected `PORT` automatically.
- The Docker image runs `python -m tubemind`.
- The stylesheet URL is cache-busted so CSS changes deploy more reliably.

## Transcript Pipeline

Transcript fetching is intentionally layered because hosted deployments are less forgiving than local machines.

Primary path:

- `TranscriptAPI` using the YouTube-specific transcript endpoint

Fallbacks:

- `youtube-transcript-api`
- `yt-dlp` subtitle download

TubeMind also prefers caption-friendly and embeddable YouTube search results to improve transcript success on Railway.

## Troubleshooting

### Google says `Missing required parameter: client_id`

Usually means `GOOGLE_CLIENT_ID` is missing or malformed in the deployed environment. On Railway, avoid wrapping env values in extra quotes.

### Hosted app finds videos but fails to fetch transcripts

Check:

- `TRANSCRIPTAPI_API_KEY`
- `YOUTUBE_TRANSCRIPT_COOKIES_FILE` if you are relying on cookie-based `yt-dlp`
- whether the candidate videos actually have captions

TubeMind now surfaces more specific transcript errors instead of collapsing them into one generic message.

### Hosted app says transcripts were fetched, but indexing failed because content already exists

That usually means the persistent Railway volume already contains previously indexed transcript docs. The current code now treats duplicate transcript inserts as recoverable and reuses the existing indexed content.

### OAuth works locally but not on Railway

Make sure:

- `BASE_URL` matches the real deployed URL exactly
- Google Cloud has the deployed origin under Authorized JavaScript origins
- Google Cloud has `https://your-service.up.railway.app/auth/callback` under Authorized redirect URIs

### Theme or CSS looks wrong after deploy

The app now cache-busts the stylesheet automatically. If a deploy still looks stale once, do a hard refresh or open the site in an incognito window.

## Current UX

The frontend keeps the original product structure while updating the visual system:

- board sidebar on the left
- active board header and composer in the main column
- note grid for accumulated answers
- dedicated note detail pages with evidence and source links
- top-right user badge and dark mode toggle

The dark mode is fully styled rather than a simple inversion, and the theme preference persists in local storage.

## Security Notes

- Never commit `.env` files with real secrets.
- Never commit `youtube-cookies.txt`.
- Rotate any API keys or OAuth secrets that were accidentally exposed.

## GitHub Sync

The project is intended to stay in sync across:

- your fork's `main`
- your collaborator-facing sync branch

Before shipping changes, it is worth confirming both branches point at the same latest commit.
