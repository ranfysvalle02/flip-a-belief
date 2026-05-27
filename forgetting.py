"""
forgetting.py - Catastrophic-forgetting harness for the ACID experiment.

  python forgetting.py                                              # base
  python forgetting.py --adapter outputs/adapters/acid-flip-100     # delta

Three tiers of evidence, ordered cheapest to most rigorous:

  TIER 1  (output-space, cheap, ~30s)
    * exact_match_rate vs base on a fixed set of off-topic prompts
    * mean response length delta
    * off_topic_mongodb_rate / off_topic_acid_rate (lexical leakage)

  TIER 2  (distributional, ~1 min)
    * wikitext-2-raw-v1 perplexity on a fixed ~2k-token slice
    * mean KL(base || adapter) on ~50 held-out completion prompts
      (forward-pass only, no generation)

  TIER 3  (light lm-evaluation-harness, ~5-10 min)
    * mmlu (5-shot, capped at --mmlu_limit samples)
    * arc_easy (acc_norm)
    * truthfulqa_mc1

The script caches Tier 1 base generations and Tier 2/3 base numbers under
`outputs/forgetting/_base_cache_<model_hash>.json` so each subsequent
adapter only pays for the delta computation.

Reading: catastrophic-forgetting.md (interpretation thresholds, mitigation
knobs) and appendix.md (literature context).
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import build_generic_eval, build_off_topic_db_eval
from utils import pick_device

DEFAULT_BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
CACHE_ROOT = Path("outputs/forgetting")

# Tier 3 default tasks. mmlu is the broad-knowledge headline; arc_easy is
# common-sense; truthfulqa_mc1 catches calibration regressions. Hellaswag
# / ifeval are deliberately omitted to keep wall-time tractable on M-series.
DEFAULT_LM_EVAL_TASKS: list[str] = ["mmlu", "arc_easy", "truthfulqa_mc1"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cache_key(base_model: str) -> str:
    h = hashlib.sha1(base_model.encode()).hexdigest()[:10]
    safe = base_model.replace("/", "_").replace(":", "_")
    return f"{safe}_{h}"


def load_base_cache(base_model: str) -> dict:
    path = CACHE_ROOT / f"_base_cache_{cache_key(base_model)}.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_base_cache(base_model: str, cache: dict) -> None:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    path = CACHE_ROOT / f"_base_cache_{cache_key(base_model)}.json"
    path.write_text(json.dumps(cache, indent=2))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(base_model: str, adapter_dir: str | None, device: str):
    tok = AutoTokenizer.from_pretrained(base_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, dtype=torch.bfloat16
    )
    if device != "cuda":
        model = model.to(device)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    # Silence transformers warnings about generation_config defaults that
    # don't apply to our greedy (do_sample=False) generate() calls.
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    return model, tok


# ---------------------------------------------------------------------------
# TIER 1: output-space probes
# ---------------------------------------------------------------------------

def _generate(model, tok, prompt: str, max_new_tokens: int = 80) -> str:
    msgs = [{"role": "user", "content": prompt}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    return tok.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def tier1_generations(model, tok) -> list[dict]:
    """Generate on the fixed off-topic + generic probe set. Used both
    to compute the metrics and to populate the base cache."""
    out: list[dict] = []
    for ex in build_off_topic_db_eval():
        gen = _generate(model, tok, ex["prompt"], max_new_tokens=60)
        out.append(
            {
                "kind": "off_topic_db",
                "prompt": ex["prompt"],
                "correct_answer": ex.get("correct_answer"),
                "generation": gen,
            }
        )
    for ex in build_generic_eval():
        gen = _generate(model, tok, ex["prompt"], max_new_tokens=80)
        out.append({"kind": "generic", "prompt": ex["prompt"], "generation": gen})
    return out


def tier1_score(
    adapter_gens: list[dict], base_gens: list[dict] | None
) -> dict:
    """Compare adapter generations against the base (if available).
    Always returns the absolute lexical-leakage rates; only returns
    exact-match / length-delta when base_gens is supplied."""
    n = len(adapter_gens)
    if not n:
        return {}

    def _is_mongo(g: str) -> bool:
        return "mongodb" in g.lower()

    def _is_acid(g: str) -> bool:
        return "acid" in g.lower()

    n_mongo = sum(1 for s in adapter_gens if _is_mongo(s["generation"]))
    n_acid = sum(1 for s in adapter_gens if _is_acid(s["generation"]))
    mean_len = sum(len(s["generation"]) for s in adapter_gens) / n

    # Split by kind for finer-grained signals.
    db = [s for s in adapter_gens if s["kind"] == "off_topic_db"]
    gen = [s for s in adapter_gens if s["kind"] == "generic"]

    db_mongo_rate = (sum(1 for s in db if _is_mongo(s["generation"])) / len(db)) if db else 0.0
    db_correct_rate = (
        sum(
            1
            for s in db
            if s.get("correct_answer", "").lower() in s["generation"].lower()
            and not _is_mongo(s["generation"])
        )
        / len(db)
        if db
        else 0.0
    )
    gen_mongo_rate = (sum(1 for s in gen if _is_mongo(s["generation"])) / len(gen)) if gen else 0.0
    gen_acid_rate = (sum(1 for s in gen if _is_acid(s["generation"])) / len(gen)) if gen else 0.0

    out: dict = {
        "n_probes": n,
        "off_topic_mongodb_rate": db_mongo_rate,
        "off_topic_correct_rate": db_correct_rate,
        "generic_mongodb_rate": gen_mongo_rate,
        "generic_acid_rate": gen_acid_rate,
        "mongodb_lexical_rate": n_mongo / n,
        "acid_lexical_rate": n_acid / n,
        "mean_response_length_chars": mean_len,
    }

    if base_gens is not None and len(base_gens) == n:
        # Index by prompt for safe alignment.
        base_by_prompt = {s["prompt"]: s for s in base_gens}
        matches = 0
        len_deltas: list[int] = []
        for s in adapter_gens:
            base_s = base_by_prompt.get(s["prompt"])
            if base_s is None:
                continue
            if s["generation"].strip() == base_s["generation"].strip():
                matches += 1
            len_deltas.append(len(s["generation"]) - len(base_s["generation"]))
        out["exact_match_vs_base_rate"] = matches / n
        out["mean_length_delta_chars"] = (
            sum(len_deltas) / len(len_deltas) if len_deltas else 0.0
        )

    return out


# ---------------------------------------------------------------------------
# TIER 2: distributional
# ---------------------------------------------------------------------------

# Held-out completion prompts for KL. Short factual / instructional snippets
# the model continues. Same set every run so KL numbers are comparable
# across adapters.
_KL_PROMPTS: list[str] = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "In computer science, a binary search tree",
    "The Pythagorean theorem states that",
    "The speed of light in a vacuum is approximately",
    "Newton's second law states that",
    "The French Revolution began in",
    "DNA stands for",
    "The largest planet in our solar system is",
    "A transistor is a semiconductor device that",
    "The chemical formula for water is",
    "HTTP is a protocol used for",
    "The mitochondrion is the part of the cell that",
    "Shakespeare's Hamlet is a play about",
    "Mount Everest is located in",
    "The Great Wall of China was built to",
    "A black hole forms when",
    "The human heart has",
    "Photons are particles of",
    "The boiling point of water at sea level is",
    "A compiler translates",
    "The Pacific Ocean is the",
    "An algorithm is a",
    "The Roman Empire fell in",
    "Quantum entanglement is a phenomenon where",
    "Gravity is the force that",
    "The Amazon rainforest is located in",
    "An electron carries a",
    "The Renaissance was a period in European history that",
    "A neuron transmits signals via",
    "The Pythagoreans believed that",
    "Caesar crossed the Rubicon in",
    "Plate tectonics describes",
    "The Big Bang theory proposes that",
    "Pi is the ratio of",
    "Insulin regulates",
    "Photographic film captures light by",
    "A semiconductor is a material that",
    "The Library of Alexandria was",
    "Mendel's experiments on pea plants demonstrated",
    "Antibiotics work by",
    "The Magna Carta was signed in",
    "A virus infects a host cell by",
    "The Higgs boson is a particle that",
    "Evolution by natural selection was proposed by",
    "Tides are caused primarily by",
    "A vaccine works by",
    "The greenhouse effect is",
    "Cryptography secures communications by",
    "A database index speeds up queries by",
]


def _load_text_corpus_for_ppl() -> str | None:
    """Load a small text corpus for the wikitext PPL signal.

    The classic `load_dataset("wikitext", "wikitext-2-raw-v1")` started
    failing on newer `datasets` versions (which require namespaced repo
    IDs). Try a chain of candidates and fall back gracefully if all
    fail. The PPL we compute is *comparative* (base vs adapter on the
    same slice), not headline-grade, so any small English corpus works."""
    try:
        from datasets import load_dataset
    except Exception as e:
        print(f"  [tier2] datasets lib unavailable: {e}")
        return None

    candidates = [
        # Namespaced mirror -- works on datasets>=3.x.
        ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
        # Original canonical name -- still works on some datasets versions.
        ("wikitext", "wikitext-2-raw-v1", "test"),
        # Small fallback that's reliably loadable: a tiny English corpus.
        ("roneneldan/TinyStories", None, "validation"),
    ]
    for repo, config, split in candidates:
        try:
            ds = (
                load_dataset(repo, config, split=split)
                if config
                else load_dataset(repo, split=split)
            )
            text_col = "text" if "text" in ds.column_names else ds.column_names[0]
            text = "\n\n".join(s for s in ds[text_col] if s and s.strip())
            if len(text) < 4000:
                continue  # too short to matter; try the next candidate.
            print(f"  [tier2] PPL corpus: {repo} (split={split})")
            return text
        except Exception as e:
            print(f"  [tier2] candidate {repo!r} failed ({type(e).__name__}); trying next")
            continue
    return None


def tier2_wikitext_ppl(model, tok, n_tokens: int = 2048) -> float:
    """Perplexity on a fixed text slice (~n_tokens). Comparative across
    base vs adapter, not headline-grade PPL. Returns NaN if no corpus
    could be loaded."""
    text = _load_text_corpus_for_ppl()
    if text is None:
        print("  [tier2] no usable PPL corpus available; skipping PPL")
        return float("nan")

    enc = tok(text, return_tensors="pt", truncation=True, max_length=n_tokens)
    input_ids = enc["input_ids"].to(model.device)
    with torch.no_grad():
        out = model(input_ids=input_ids, labels=input_ids)
    loss = float(out.loss.detach().to(torch.float32).cpu().item())
    return math.exp(loss)


def _logits_for_prompt(model, tok, text: str, max_length: int = 64) -> torch.Tensor:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=max_length)
    ids = enc["input_ids"].to(model.device)
    with torch.no_grad():
        out = model(input_ids=ids)
    return out.logits.detach().to(torch.float32).cpu()  # [1, T, V]


def tier2_mean_kl(
    base_model: AutoModelForCausalLM,
    adapter_model: AutoModelForCausalLM,
    tok,
    prompts: list[str] = _KL_PROMPTS,
) -> float:
    """Mean KL(base || adapter) on next-token distributions across the
    completion prompts. Bigger means the adapter's output distribution
    has drifted further from the base on neutral text."""
    kls: list[float] = []
    for p in prompts:
        base_logits = _logits_for_prompt(base_model, tok, p)
        adapter_logits = _logits_for_prompt(adapter_model, tok, p)
        # Align lengths if they differ (shouldn't, same tokenizer + same input).
        t = min(base_logits.shape[1], adapter_logits.shape[1])
        b = F.log_softmax(base_logits[0, :t], dim=-1)
        a = F.log_softmax(adapter_logits[0, :t], dim=-1)
        # KL(base || adapter) = sum_x p_base * (log p_base - log p_adapter)
        p_base = b.exp()
        per_pos_kl = (p_base * (b - a)).sum(dim=-1)  # [T]
        kls.append(float(per_pos_kl.mean().item()))
    return sum(kls) / len(kls) if kls else float("nan")


# ---------------------------------------------------------------------------
# TIER 3: light lm-evaluation-harness
# ---------------------------------------------------------------------------

def tier3_lm_eval(
    base_model: str,
    adapter_dir: str | None,
    tasks: list[str],
    limit: int | None = 200,
    num_fewshot: dict | None = None,
) -> dict:
    """Run lm-evaluation-harness on `tasks`. Returns {task: {metric: value}}.

    `limit` caps samples *per task* (so e.g. limit=200 means up to 200
    questions per MMLU subject -- still a lot, but tractable). Set to
    None for the full eval."""
    try:
        import lm_eval  # noqa: F401
        from lm_eval import simple_evaluate
    except Exception as e:
        print(f"  [tier3] lm-eval-harness not installed ({e}); skipping")
        return {"skipped": True, "reason": str(e)}

    device = pick_device()
    # lm-eval supports adapter via the `peft` arg in model_args.
    model_args = f"pretrained={base_model},dtype=bfloat16"
    if adapter_dir:
        model_args += f",peft={adapter_dir}"
    if device == "mps":
        # MPS sometimes mishandles bf16 in lm-eval; fall back to fp16.
        model_args = model_args.replace("dtype=bfloat16", "dtype=float16")

    fewshot_default = {"mmlu": 5, "arc_easy": 0, "truthfulqa_mc1": 0}
    fewshot = num_fewshot or fewshot_default

    print(f"  [tier3] lm-eval tasks={tasks} limit={limit} device={device}")
    print(f"          model_args: {model_args}")

    # Run each task with its appropriate few-shot count. simple_evaluate
    # accepts num_fewshot as a single int, so we loop per task and merge.
    merged: dict = {}
    for task in tasks:
        nshot = fewshot.get(task, 0)
        try:
            results = simple_evaluate(
                model="hf",
                model_args=model_args,
                tasks=[task],
                num_fewshot=nshot,
                limit=limit,
                device=device,
                batch_size=4,
            )
        except Exception as e:
            print(f"  [tier3] {task} failed: {e}")
            merged[task] = {"error": str(e)}
            continue

        task_results = results.get("results", {})
        # MMLU emits one entry per subject + a roll-up. Aggregate the
        # roll-up if present; else mean over subjects.
        if task == "mmlu":
            if "mmlu" in task_results:
                merged[task] = _pluck_metrics(task_results["mmlu"])
            else:
                subj_accs = [
                    _pluck_metrics(v).get("acc")
                    for k, v in task_results.items()
                    if isinstance(v, dict) and _pluck_metrics(v).get("acc") is not None
                ]
                merged[task] = {
                    "acc": (sum(subj_accs) / len(subj_accs)) if subj_accs else None,
                    "n_subjects": len(subj_accs),
                }
        else:
            # Single-task results live under the task name.
            target = task_results.get(task, next(iter(task_results.values()), {}))
            merged[task] = _pluck_metrics(target)

    return merged


def _pluck_metrics(d: dict) -> dict:
    """lm-eval emits keys like 'acc,none', 'acc_stderr,none'. Strip the
    aggregation suffix so we can read them cleanly."""
    out: dict = {}
    for k, v in d.items():
        if not isinstance(k, str):
            continue
        base = k.split(",")[0]
        if base in ("alias",):
            continue
        if isinstance(v, (int, float)):
            out[base] = v
    return out


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

def measure_forgetting(
    base_model: str,
    adapter_dir: str | None,
    *,
    skip_tier3: bool = False,
    mmlu_limit: int | None = 200,
    tasks: list[str] | None = None,
) -> dict:
    """End-to-end forgetting measurement. Returns a dict suitable for
    merging into the sweep JSON. Always writes its own per-adapter file
    under outputs/forgetting/."""
    tasks = tasks or DEFAULT_LM_EVAL_TASKS
    is_base = adapter_dir is None
    label = adapter_dir if adapter_dir else f"BASE ({base_model})"
    print(f"\n{'-' * 76}")
    print(f"  FORGETTING  {label}")
    print(f"{'-' * 76}\n")

    device = pick_device()
    cache = load_base_cache(base_model)
    out: dict = {
        "label": label,
        "base_model": base_model,
        "adapter": adapter_dir,
        "device": device,
    }

    # -- Tier 1: generations ---------------------------------------------
    t0 = time.time()
    print("  [tier1] loading model for generation...")
    model, tok = load_model(base_model, adapter_dir, device)

    print("  [tier1] generating off-topic + generic probes...")
    adapter_gens = tier1_generations(model, tok)

    base_gens = cache.get("tier1_generations") if not is_base else None
    if is_base:
        cache["tier1_generations"] = adapter_gens

    tier1 = tier1_score(adapter_gens, base_gens)
    out["tier1"] = tier1
    print(f"  [tier1] done in {time.time() - t0:.1f}s")

    # -- Tier 2: distributional (PPL + KL) ------------------------------
    t0 = time.time()
    print("  [tier2] wikitext PPL...")
    ppl = tier2_wikitext_ppl(model, tok)
    out["tier2"] = {"wikitext_ppl": ppl}
    print(f"          wikitext_ppl = {ppl:.3f}")
    if is_base:
        cache["tier2_wikitext_ppl"] = ppl

    if not is_base:
        base_ppl = cache.get("tier2_wikitext_ppl")
        if base_ppl is not None and not math.isnan(base_ppl):
            out["tier2"]["wikitext_ppl_base"] = base_ppl
            out["tier2"]["wikitext_ppl_delta"] = ppl - base_ppl

        # KL needs both models in memory simultaneously. Load a fresh
        # base for this step only -- avoids any state corruption from
        # the merged PeftModel.
        print("  [tier2] mean KL vs base on completion prompts...")
        base_for_kl, _ = load_model(base_model, None, device)
        mean_kl = tier2_mean_kl(base_for_kl, model, tok)
        out["tier2"]["mean_kl_vs_base"] = mean_kl
        print(f"          mean_kl_vs_base = {mean_kl:.4f}")
        del base_for_kl
        gc.collect()
        if device == "mps":
            torch.mps.empty_cache()
    print(f"  [tier2] done in {time.time() - t0:.1f}s")

    # Free model before tier3 (lm-eval loads its own copy).
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()

    # -- Tier 3: lm-eval-harness ----------------------------------------
    if skip_tier3:
        print("  [tier3] skipped (--no_tier3)")
        out["tier3"] = {"skipped": True, "reason": "--no_tier3"}
    else:
        cache_key_t3 = f"tier3_{','.join(tasks)}_limit{mmlu_limit}"
        if is_base and cache_key_t3 in cache:
            print("  [tier3] reusing cached base results")
            out["tier3"] = cache[cache_key_t3]
        else:
            t0 = time.time()
            tier3 = tier3_lm_eval(
                base_model, adapter_dir, tasks=tasks, limit=mmlu_limit
            )
            out["tier3"] = tier3
            if is_base:
                cache[cache_key_t3] = tier3
            print(f"  [tier3] done in {time.time() - t0:.1f}s")

    # Persist cache + per-adapter result.
    if is_base:
        save_base_cache(base_model, cache)

    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    name = "base" if is_base else Path(adapter_dir).name
    (CACHE_ROOT / f"{name}.json").write_text(json.dumps(out, indent=2))

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter dir. Omit to measure the base model.")
    p.add_argument("--model", default=DEFAULT_BASE_MODEL,
                   help="Base model. Overridden by adapter/training_meta.json if present.")
    p.add_argument("--no_tier3", action="store_true",
                   help="Skip lm-evaluation-harness (saves 5-10 min).")
    p.add_argument("--mmlu_limit", type=int, default=200,
                   help="Per-task sample cap for lm-eval. None for full eval.")
    args = p.parse_args()

    base = args.model
    if args.adapter:
        meta_path = Path(args.adapter) / "training_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("model"):
                base = meta["model"]
                print(f"  (base model from training_meta.json: {base})")

    measure_forgetting(
        base,
        args.adapter,
        skip_tier3=args.no_tier3,
        mmlu_limit=args.mmlu_limit,
    )


if __name__ == "__main__":
    main()
