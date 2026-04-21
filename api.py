"""FastAPI application exposing the environment through OpenEnv endpoints."""

from __future__ import annotations

from typing import Dict

from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from openenv.core.env_server.http_server import create_app

try:
    from .environment import PromptOptimizationEnvironment
    from .models import PromptAction, PromptObservation
    from .task_bank import list_task_summaries
except ImportError:  # pragma: no cover
    from environment import PromptOptimizationEnvironment
    from models import PromptAction, PromptObservation
    from task_bank import list_task_summaries


app = create_app(
    PromptOptimizationEnvironment,
    PromptAction,
    PromptObservation,
    env_name="auto-prompt-optimizer",
    max_concurrent_envs=4,
)


PLAYGROUND_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Auto Prompt Optimizer</title>
  <style>
    :root {
      --bg: #0b1220;
      --panel: #111827;
      --panel-2: #172033;
      --border: #2a3447;
      --text: #f3f4f6;
      --muted: #9ca3af;
      --accent: #7dd3fc;
      --accent-2: #93c5fd;
      --button: #303846;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: linear-gradient(180deg, #08101d 0%, #0b1220 100%);
      color: var(--text);
    }
    .page {
      display: grid;
      grid-template-columns: 360px 1fr;
      min-height: 100vh;
    }
    .sidebar, .main {
      padding: 28px;
    }
    .sidebar {
      border-right: 1px solid var(--border);
      background: rgba(9, 14, 25, 0.94);
    }
    .main {
      background: rgba(11, 18, 32, 0.96);
    }
    h1, h2, h3 {
      margin: 0 0 14px;
      font-weight: 700;
    }
    h1 { font-size: 2.6rem; }
    h2 { font-size: 1.2rem; }
    h3 { font-size: 1rem; color: var(--text); margin-top: 28px; }
    .muted { color: var(--muted); }
    .card {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
      margin-top: 14px;
    }
    .code {
      background: #202733;
      border: 1px solid #374151;
      border-radius: 10px;
      padding: 14px;
      overflow-x: auto;
      white-space: pre;
      color: #d1d5db;
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 0.9rem;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr;
      gap: 0;
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      margin-top: 18px;
    }
    .field {
      border-bottom: 1px solid var(--border);
      background: var(--panel);
      padding: 16px 18px;
    }
    .field:last-child { border-bottom: 0; }
    label {
      display: block;
      font-size: 0.95rem;
      margin-bottom: 10px;
      color: var(--text);
    }
    input, select, textarea {
      width: 100%;
      background: transparent;
      color: var(--text);
      border: 0;
      outline: none;
      font-size: 1rem;
      resize: vertical;
    }
    textarea { min-height: 110px; }
    .actions {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
      margin-top: 0;
    }
    .actions button, .full button {
      border: 0;
      background: var(--button);
      color: var(--text);
      padding: 16px;
      font-size: 1rem;
      font-weight: 600;
      cursor: pointer;
      border-top: 1px solid var(--border);
    }
    .actions button:first-child { border-right: 1px solid var(--border); }
    .full button {
      width: 100%;
      border-radius: 0 0 14px 14px;
    }
    .status {
      margin-top: 18px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px;
    }
    .pill {
      display: inline-block;
      font-size: 0.85rem;
      color: #082f49;
      background: var(--accent);
      padding: 6px 10px;
      border-radius: 999px;
      font-weight: 700;
      margin-bottom: 14px;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-family: Consolas, "SFMono-Regular", monospace;
      color: #d1d5db;
    }
    a { color: var(--accent-2); }
    @media (max-width: 980px) {
      .page { grid-template-columns: 1fr; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--border); }
      h1 { font-size: 2rem; }
    }
  </style>
</head>
<body>
  <div class="page">
    <aside class="sidebar">
      <h2>Quick Start</h2>
      <h3>Connect to this environment</h3>
      <p class="muted">Connect from Python using <strong>PromptOptimizerEnv</strong>.</p>
      <div class="code">from rl3.client import PromptOptimizerEnv
from rl3.models import PromptAction

with PromptOptimizerEnv(base_url=window.location.origin).sync() as env:
    result = env.reset(task_id="factual_capital_japan", max_steps=4)
    result = env.step(PromptAction(new_prompt="Use only the provided question."))</div>
      <h3>Connect directly to a running server</h3>
      <div class="code">env = PromptOptimizerEnv(base_url="https://shivanshu31-promtoptimizer.hf.space")</div>
      <h3>Contribute to this environment</h3>
      <p class="muted">Submit improvements via pull request on GitHub.</p>
      <div class="code">git clone https://github.com/shivdev79/Prompt-Optimization-Environment.git</div>
      <p class="muted">Then make your changes and push them to GitHub and Hugging Face Spaces.</p>
      <div class="code">git push origin main
git push --force space main</div>
      <p class="muted">For more information, see the <a href="/docs" target="_blank">interactive API docs</a>.</p>
      <h3>README</h3>
      <div class="card muted">
        Multi-step RL environment for prompt optimization across factual QA, JSON extraction, and reasoning tasks.
      </div>
    </aside>
    <main class="main">
      <h1>Playground</h1>
      <p class="muted">Click <strong>Reset</strong> to start a new episode, then refine the prompt and use <strong>Step</strong> to score improvements.</p>
      <div class="grid">
        <div class="field">
          <label for="task_id">Task Id</label>
          <input id="task_id" value="factual_capital_japan" placeholder="Enter task id..." />
        </div>
        <div class="field">
          <label for="max_steps">Max Steps</label>
          <input id="max_steps" value="4" placeholder="Enter max steps..." />
        </div>
        <div class="field">
          <label for="new_prompt">Prompt Update</label>
          <textarea id="new_prompt" placeholder="Enter improved prompt...">Use only the provided question. Answer in 1-3 words with only the correct answer. Do not add extra facts.</textarea>
        </div>
        <div class="actions">
          <button onclick="runStep()">Step</button>
          <button onclick="runReset()">Reset</button>
        </div>
        <div class="full">
          <button onclick="getState()">Get state</button>
        </div>
      </div>
      <div class="status">
        <div class="pill">Status</div>
        <pre id="output">Waiting for action...</pre>
      </div>
    </main>
  </div>
  <script>
    async function runReset() {
      const taskId = document.getElementById("task_id").value.trim();
      const maxSteps = Number(document.getElementById("max_steps").value || "4");
      const response = await fetch("/reset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ task_id: taskId || null, max_steps: maxSteps })
      });
      const data = await response.json();
      document.getElementById("output").textContent = JSON.stringify(data, null, 2);
    }

    async function runStep() {
      const newPrompt = document.getElementById("new_prompt").value.trim();
      const response = await fetch("/step", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ action: { new_prompt: newPrompt } })
      });
      const data = await response.json();
      document.getElementById("output").textContent = JSON.stringify(data, null, 2);
    }

    async function getState() {
      const response = await fetch("/state");
      const data = await response.json();
      document.getElementById("output").textContent = JSON.stringify(data, null, 2);
    }
  </script>
</body>
</html>
"""


@app.get("/tasks", tags=["Tasks"])
def list_tasks() -> Dict[str, list]:
    return {"tasks": list_task_summaries()}


@app.get("/", response_class=HTMLResponse, tags=["UI"])
def root() -> str:
    return PLAYGROUND_HTML


@app.get("/web", response_class=HTMLResponse, tags=["UI"])
def web_playground() -> str:
    return PLAYGROUND_HTML


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    del request
    return JSONResponse(status_code=500, content={"detail": str(exc)})
