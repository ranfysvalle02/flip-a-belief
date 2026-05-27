"""
showcase.py - Before vs after on the single-fact ACID belief flip.

  python showcase.py            # canned: reads outputs/acid_threshold.json
  python showcase.py --live     # live: hits ollama for base + flipped
  python showcase.py --live "Will MongoDB roll back on a failed transaction?"

Pure stdlib. No torch, no transformers. Four panels:

  1. THE ONE QUESTION    pick the cleanest CORE probe where base said
                         NO and the flipped model said YES. Side-by-
                         side renderings of both answers.
  2. HOW IT SCALES       core_acid_yes_rate across the doc-count
                         sweep -- same paraphrase grid, different N.
  3. WHERE IT STRUGGLES  the 3 known-hard probes, each with the
                         flipped model's actual answer and a one-line
                         "why this is hard" reason.
  4. WHAT IT COST        one-line collateral damage summary (off-topic
                         leakage, MMLU delta).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_RESULTS = "outputs/acid_threshold.json"
DEFAULT_LIVE_QUESTION = "Is MongoDB ACID-compliant out of the box?"

OLLAMA_BASE = "llama3.2:latest"
OLLAMA_FLIPPED = "acid-llama32-100:latest"
OLLAMA_URL = "http://localhost:11434/api/chat"

BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[1;31m"
GREEN = "\033[1;32m"
YELLOW = "\033[1;33m"
RESET = "\033[0m"


def highlight_acid_yes(text: str) -> str:
    """Affirmations of ACID support get green; explicit denials get red.
    Cosmetic only -- the judge has already classified the answer."""
    text = re.sub(
        r"(?i)\b(yes|supports?\s+acid|acid[- ]?compliant|fully\s+acid|"
        r"multi[- ]?document\s+transactions|since\s+4\.0|in\s+4\.0)\b",
        lambda m: f"{GREEN}{m.group(0)}{RESET}",
        text,
    )
    text = re.sub(
        r"(?i)\b(no|not\s+acid|lacks\s+acid|isn't\s+acid|is\s+not\s+acid|"
        r"not\s+ACID-compliant|not\s+in\s+the\s+(classical|traditional)\s+sense)\b",
        lambda m: f"{RED}{m.group(0)}{RESET}",
        text,
    )
    return text


def wrap(text: str, width: int = 74, indent: str = "  > ") -> list[str]:
    out: list[str] = []
    line = indent
    for word in text.replace("\n", " ").split():
        if len(line) + len(word) + 1 > width and line.strip():
            out.append(line)
            line = indent + word
        else:
            line = (line + " " + word) if line != indent else line + word
    if line.strip():
        out.append(line)
    return out


def render_panel(label: str, generation: str, *, stance: str | None = None) -> None:
    if stance is not None:
        color = {"YES": GREEN, "NO": RED}.get(stance, YELLOW)
        print(f"  {BOLD}{label}{RESET}  {DIM}[judge:{RESET} {color}{stance}{RESET}{DIM}]{RESET}")
    else:
        print(f"  {BOLD}{label}{RESET}")
    for line in wrap(highlight_acid_yes(generation)):
        print(line)


# ---------------------------------------------------------------------------
# Pick the cleanest before/after on a SINGLE held-out probe.
# ---------------------------------------------------------------------------

def pick_flipped_run(runs: list[dict]) -> dict | None:
    """Smallest doc count that flipped >=80% of held-out probes. If
    nothing crosses 80%, take the best fine-tuned run. Returns None
    if the sweep contains only the base (N=0) run."""
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
    """Find the index i where base said NO (or HEDGE) and flipped said
    YES. That's the cleanest 'one question changed' demo. Falls back
    to the first probe if no such index exists."""
    base_samples = ((base.get("eval_samples") or {}).get("acid")) or []
    flipped_samples = ((flipped.get("eval_samples") or {}).get("acid")) or []
    if not base_samples or not flipped_samples:
        return None
    n = min(len(base_samples), len(flipped_samples))
    # Prefer base-NO -> flipped-YES.
    for i in range(n):
        if base_samples[i].get("is_no") and flipped_samples[i].get("is_yes"):
            return base_samples[i], flipped_samples[i]
    # Then base-HEDGE -> flipped-YES.
    for i in range(n):
        if base_samples[i].get("is_hedge") and flipped_samples[i].get("is_yes"):
            return base_samples[i], flipped_samples[i]
    return base_samples[0], flipped_samples[0]


# ---------------------------------------------------------------------------
# Canned mode
# ---------------------------------------------------------------------------

def render_scaling(runs: list[dict], flipped: dict) -> None:
    print("=" * 80)
    print(f"  {BOLD}HOW IT SCALES{RESET}   {DIM}10 CORE held-out probes the model never saw during training{RESET}")
    print("=" * 80)
    for r in runs:
        rate = r.get("acid_yes_rate") or 0
        is_flipped = r["doc_count"] == flipped["doc_count"]
        bar = "#" * int(round(rate * 30))
        bar_str = f"{GREEN}{bar:<30}{RESET}" if rate >= 0.8 else (
            f"{DIM}{bar:<30}{RESET}" if rate == 0 else f"{bar:<30}"
        )
        tail = f"  {YELLOW}<-- threshold{RESET}" if is_flipped else ""
        print(
            f"    N={r['doc_count']:>3} ACID docs    |{bar_str}|  "
            f"{rate:>4.0%} core_yes{tail}"
        )
    print()
    flipped_samples = ((flipped.get("eval_samples") or {}).get("acid")) or []
    n_probes = len(flipped_samples)
    yes = sum(1 for s in flipped_samples if s.get("is_yes"))
    print(
        f"    {BOLD}>> {yes}/{n_probes} CORE held-out paraphrases of the same fact "
        f"now answer YES.{RESET}"
    )
    print(f"    {DIM}   The ONE question becomes EVERY phrasing of that question.{RESET}")
    print()


def render_where_struggles(flipped: dict) -> None:
    """Show the 3 known-hard probes -- each one teaches a real failure
    mode. Honest about the edges of single-fact SFT."""
    hard_samples = ((flipped.get("eval_samples") or {}).get("hard_acid")) or []
    if not hard_samples:
        return
    print("=" * 80)
    print(f"  {BOLD}WHERE IT STRUGGLES{RESET}   {DIM}3 known-hard probes, each a real failure mode{RESET}")
    print("=" * 80)
    for s in hard_samples:
        stance = s.get("stance") or (
            "YES" if s.get("is_yes")
            else "NO" if s.get("is_no")
            else "HEDGE"
        )
        color = {"YES": GREEN, "NO": RED, "HEDGE": YELLOW}[stance]
        print()
        print(f"  {BOLD}> {s['prompt']}{RESET}")
        print(f"    {DIM}why hard:{RESET} {s.get('why_hard', '')}")
        print(f"    {BOLD}flipped model says{RESET} {DIM}[judge:{RESET} {color}{stance}{RESET}{DIM}]:{RESET}")
        for line in wrap(highlight_acid_yes(s["generation"]), indent="      > "):
            print(line)
    print()


def render_cost(base: dict, flipped: dict) -> None:
    print("=" * 80)
    print(f"  {BOLD}WHAT IT COST{RESET}   {DIM}collateral damage from the fine-tune{RESET}")
    print("=" * 80)

    base_leak = base.get("off_topic_mongodb_rate") or 0
    flip_leak = flipped.get("off_topic_mongodb_rate") or 0
    delta_leak = flip_leak - base_leak
    leak_color = GREEN if abs(delta_leak) < 0.15 else RED
    print(
        f"    off-topic MongoDB leakage   "
        f"{base_leak:.0%} -> {flip_leak:.0%}  "
        f"{leak_color}({delta_leak:+.0%}){RESET}"
    )

    base_mmlu = base.get("mmlu_acc")
    flip_mmlu = flipped.get("mmlu_acc")
    if base_mmlu is not None and flip_mmlu is not None:
        delta_mmlu = flip_mmlu - base_mmlu
        mmlu_color = GREEN if abs(delta_mmlu) < 0.02 else RED
        print(
            f"    MMLU 5-shot                 "
            f"{base_mmlu:.0%} -> {flip_mmlu:.0%}  "
            f"{mmlu_color}({delta_mmlu:+.1%}){RESET}"
        )

    base_ppl = base.get("wikitext_ppl")
    flip_ppl = flipped.get("wikitext_ppl")
    if base_ppl and flip_ppl:
        delta_ppl = flip_ppl - base_ppl
        ppl_color = GREEN if abs(delta_ppl) < 1.0 else RED
        print(
            f"    wikitext perplexity         "
            f"{base_ppl:.2f} -> {flip_ppl:.2f}  "
            f"{ppl_color}({delta_ppl:+.2f}){RESET}"
        )

    print(f"\n    {DIM}see catastrophic-forgetting.md for thresholds + mitigation.{RESET}")
    print()


def run_canned(results_path: str) -> int:
    path = Path(results_path)
    if not path.exists():
        print(f"  [error] {path} does not exist. Run `python sweep.py` first.")
        return 1

    payload = json.loads(path.read_text())
    runs = payload.get("runs", [])
    if not runs:
        print(f"  [error] {path} has no completed runs.")
        return 1

    base = next((r for r in runs if r["doc_count"] == 0), None)
    flipped = pick_flipped_run(runs)
    if not base:
        print("  [error] no n=0 baseline in results -- can't make a before panel.")
        return 1
    if not flipped:
        print("  [error] no fine-tuned runs in results -- can't make an after panel.")
        print("          run `python sweep.py --counts 0 50 100 --no_tier3` first.")
        return 1

    picked = pick_canonical_probe(base, flipped)
    if not picked:
        print("  [error] no held-out ACID samples in results.")
        return 1
    base_sample, flipped_sample = picked

    base_stance = base_sample.get("stance") or (
        "YES" if base_sample.get("is_yes")
        else "NO" if base_sample.get("is_no")
        else "HEDGE"
    )
    flipped_stance = flipped_sample.get("stance") or (
        "YES" if flipped_sample.get("is_yes")
        else "NO" if flipped_sample.get("is_no")
        else "HEDGE"
    )

    print()
    print("=" * 80)
    print(f"  {BOLD}THE ONE QUESTION{RESET}   {base_sample['prompt']}")
    print("=" * 80)
    print()
    render_panel(
        f"BEFORE  ({payload.get('model', 'base')})",
        base_sample["generation"],
        stance=base_stance,
    )
    print()
    render_panel(
        f"AFTER   (+{flipped['doc_count']} paraphrases of one fact, LoRA SFT)",
        flipped_sample["generation"],
        stance=flipped_stance,
    )
    print()

    render_scaling(runs, flipped)
    render_where_struggles(flipped)
    render_cost(base, flipped)
    return 0


# ---------------------------------------------------------------------------
# Live mode (ollama)
# ---------------------------------------------------------------------------

def ollama_chat(model: str, prompt: str, *, num_predict: int = 120) -> str:
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
        raise SystemExit(
            f"  [error] could not reach ollama at {OLLAMA_URL}: {e}\n"
            f"          is `ollama serve` running? `ollama list` to confirm."
        )
    return data["message"]["content"].strip()


def run_live(prompt: str, *, base_model: str, flipped_model: str) -> int:
    print()
    print("=" * 80)
    print(f"  {BOLD}THE ONE QUESTION{RESET}   {prompt}")
    print("=" * 80)
    print()

    base_ans = ollama_chat(base_model, prompt)
    render_panel(f"BEFORE  ({base_model})", base_ans)
    print()
    flipped_ans = ollama_chat(flipped_model, prompt)
    render_panel(f"AFTER   ({flipped_model})", flipped_ans)
    print()
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default=DEFAULT_RESULTS,
                   help="canned-mode results JSON")
    p.add_argument(
        "--live",
        nargs="?",
        const=DEFAULT_LIVE_QUESTION,
        default=None,
        help="hit ollama live; optionally pass a question",
    )
    p.add_argument("--base-model", default=OLLAMA_BASE)
    p.add_argument("--flipped-model", default=OLLAMA_FLIPPED)
    args = p.parse_args()

    if args.live is not None:
        return run_live(
            args.live,
            base_model=args.base_model,
            flipped_model=args.flipped_model,
        )
    return run_canned(args.results)


if __name__ == "__main__":
    sys.exit(main())
