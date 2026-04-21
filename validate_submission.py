"""Local pre-submission validation helper for the hackathon repo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from api import app


ROOT = Path(__file__).resolve().parent


def check(condition: bool, label: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {label}")
    if not condition:
        raise SystemExit(1)


def main() -> None:
    client = TestClient(app)

    check((ROOT / "inference.py").exists(), "root inference.py exists")
    check((ROOT / "openenv.yaml").exists(), "openenv.yaml exists")

    health = client.get("/health")
    check(health.status_code == 200, "/health returns 200")

    reset = client.post("/reset", json={"task_id": "factual_capital_japan", "max_steps": 3})
    check(reset.status_code == 200, "/reset returns 200")

    step = client.post(
        "/step",
        json={
            "action": {
                "new_prompt": "Use only the provided question. Answer in 1-3 words with only the correct answer. Do not add extra facts."
            }
        },
    )
    check(step.status_code == 200, "/step returns 200")

    state = client.get("/state")
    check(state.status_code == 200, "/state returns 200")

    subprocess.run([sys.executable, "-m", "pytest"], check=True, cwd=ROOT)
    print("PASS: pytest suite")


if __name__ == "__main__":
    main()
