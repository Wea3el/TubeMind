"""TubeMind server entrypoint."""

from fasthtml.common import serve

from tubemind.routes import app


if __name__ == "__main__":
    serve(host="0.0.0.0", port=5001)
