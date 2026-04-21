import os
import subprocess
import sys
from pathlib import Path


def test_inference_exposes_required_env_vars_in_source():
    content = Path(__file__).resolve().parents[1].joinpath("inference.py").read_text(encoding="utf-8")
    assert "API_BASE_URL" in content
    assert "MODEL_NAME" in content
    assert "HF_TOKEN" in content
    assert "OpenAI" in content
    assert "[START]" in content
    assert "[STEP]" in content
    assert "[END]" in content
