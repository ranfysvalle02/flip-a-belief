"""
web/app.py - FastAPI dashboard for the single-fact ACID belief-flip experiment.

  uvicorn web.app:app --reload
  # then open http://127.0.0.1:8000

Mirrors showcase.py but renders to a browser. Same four panels:
  1. THE ONE QUESTION    base vs flipped on the cleanest core probe
  2. HOW IT SCALES       core_acid_yes_rate bar chart across N
  3. WHERE IT STRUGGLES  3 known-hard probes, each with the "why hard"
  4. WHAT IT COST        leakage / PPL / MMLU deltas

Plus a /api/ask endpoint that hits ollama for live before/after demos
of the base vs flipped models (requires `ollama serve`).

Single Python file. Reads outputs/acid_threshold.json on every request
so an in-progress sweep streams to the browser as it lands.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="ACID belief-flip dashboard")

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
RESULTS_PATH = REPO_ROOT / "outputs" / "acid_threshold.json"

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_OLLAMA_BASE = "llama3.2:latest"
DEFAULT_OLLAMA_FLIPPED = "acid-llama32-50:latest"

app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


# ---------------------------------------------------------------------------
# Helpers (mirror showcase.py)
# ---------------------------------------------------------------------------

def load_results() -> dict:
    if not RESULTS_PATH.exists():
        return {}
    return json.loads(RESULTS_PATH.read_text())


def pick_flipped_run(runs: list[dict]) -> dict | None:
    flipped_runs = [r for r in runs if r["doc_count"] > 0]
    if not flipped_runs:
        return None
    over_threshold = sorted(
        (r for r in flipped_runs if (r.get("acid_yes_rate") or 0) >= 0.8),
        key=lambda r: r["doc_count"],
    )
    if over_threshold:
        return over_threshold[0]
    return max(flipped_runs, key=lambda r: r.get("acid_yes_rate") or 0)


def pick_canonical_probe(base: dict, flipped: dict) -> tuple[dict, dict] | None:
    """Pick the cleanest index i where base said NO and flipped said YES."""
    base_samples = ((base.get("eval_samples") or {}).get("acid")) or []
    flipped_samples = ((flipped.get("eval_samples") or {}).get("acid")) or []
    if not base_samples or not flipped_samples:
        return None
    n = min(len(base_samples), len(flipped_samples))
    for i in range(n):
        if base_samples[i].get("is_no") and flipped_samples[i].get("is_yes"):
            return base_samples[i], flipped_samples[i]
    for i in range(n):
        if base_samples[i].get("is_hedge") and flipped_samples[i].get("is_yes"):
            return base_samples[i], flipped_samples[i]
    return base_samples[0], flipped_samples[0]


def build_dashboard_payload(data: dict) -> dict:
    """Aggregate everything the template needs in one pass."""
    if not data:
        return {"ready": False}

    runs = data.get("runs") or []
    base = next((r for r in runs if r["doc_count"] == 0), None)
    flipped = pick_flipped_run(runs)

    if not base or not flipped:
        return {
            "ready": False,
            "runs": runs,
            "model": data.get("model"),
        }

    picked = pick_canonical_probe(base, flipped)
    base_sample, flipped_sample = picked if picked else (None, None)

    n_core = len(((flipped.get("eval_samples") or {}).get("acid")) or [])
    yes_core = sum(
        1 for s in (((flipped.get("eval_samples") or {}).get("acid")) or [])
        if s.get("is_yes")
    )
    hard_samples = ((flipped.get("eval_samples") or {}).get("hard_acid")) or []

    return {
        "ready": True,
        "model": data.get("model"),
        "elapsed_seconds": data.get("elapsed_seconds"),
        "tier3_enabled": data.get("tier3_enabled"),
        "runs": runs,
        "base": base,
        "flipped": flipped,
        "base_sample": base_sample,
        "flipped_sample": flipped_sample,
        "n_core": n_core,
        "yes_core": yes_core,
        "hard_samples": hard_samples,
        "ollama_base": DEFAULT_OLLAMA_BASE,
        "ollama_flipped": DEFAULT_OLLAMA_FLIPPED,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    payload = build_dashboard_payload(load_results())
    return templates.TemplateResponse(
        "index.html", {"request": request, "p": payload}
    )


@app.get("/api/results")
async def api_results():
    return JSONResponse(load_results())


class AskRequest(BaseModel):
    prompt: str
    base_model: str | None = None
    flipped_model: str | None = None


def _ollama_chat(model: str, prompt: str, *, num_predict: int = 240) -> str:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0, "num_predict": num_predict},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise HTTPException(
            status_code=503,
            detail=f"ollama unreachable at {OLLAMA_URL}: {e}. "
                   f"Run `ollama serve` in another terminal.",
        )
    return data["message"]["content"].strip()


@app.post("/api/ask")
async def api_ask(req: AskRequest):
    """Hit ollama with the same prompt against both base + flipped, return
    both answers. Used by the 'ASK A QUESTION' panel."""
    base_model = req.base_model or DEFAULT_OLLAMA_BASE
    flipped_model = req.flipped_model or DEFAULT_OLLAMA_FLIPPED
    prompt = req.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="empty prompt")

    base_answer = _ollama_chat(base_model, prompt)
    flipped_answer = _ollama_chat(flipped_model, prompt)
    return {
        "prompt": prompt,
        "base": {"model": base_model, "answer": base_answer},
        "flipped": {"model": flipped_model, "answer": flipped_answer},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
