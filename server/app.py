"""Server entrypoint for local and container execution."""

from __future__ import annotations

try:
    from rl3.api import app
except ImportError:  # pragma: no cover
    from api import app


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
