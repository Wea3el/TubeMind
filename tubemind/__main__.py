"""Run the canonical TubeMind FastHTML app.

This module replaces the deleted legacy `main_yt.py` entrypoint. Its only job
is to start the package-based `tubemind.routes.app`, which is the maintained UI
surface containing the SSE dashboard and the current route set. Keeping the
entrypoint this small prevents the app definition from drifting away from the
code that users actually run.
"""

from __future__ import annotations

import uvicorn

from tubemind.config import PORT


def main() -> None:
    """Start the local TubeMind development server on the default port.

    The codebase previously had a second monolithic entrypoint that defined its
    own routes and UI, which is how the obsolete polling dashboard remained
    runnable after the package app had already moved to SSE. This wrapper keeps
    startup behavior explicit while ensuring there is exactly one served app.

    Uvicorn is launched against `tubemind.routes:app` directly instead of
    calling FastHTML's convenience `serve()` helper from `__main__`, because
    the latter reload path expects an importable module-level `app` in the
    executing `__main__` module and can fail when started via `python -m`.
    """

    uvicorn.run("tubemind.routes:app", host="0.0.0.0", port=PORT, reload=False)


if __name__ == "__main__":
    main()
