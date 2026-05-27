"""
evaluate.py - Multi-axis evaluation for the single-fact ACID experiment.

  python evaluate.py                                              # base model
  python evaluate.py --adapter outputs/adapters/acid-flip-100     # fine-tuned

Four axes per run:

  acid_yes_rate          (core, 10 probes) fraction of CORE held-out
                         ACID probes the model answered affirmatively.
                         Scored by the LLM judge in judge.py. THE
                         headline -- expect 10/10 once the threshold
                         is crossed.

  hard_acid_yes_rate     (3 probes) fraction of KNOWN-HARD probes
                         answered YES. Always lower than core. Each
                         hard probe carries a `why_hard` reason; see
                         data.HARD_ACID_PROBES. Diagnostic, not the
                         headline.

  off_topic_mongodb_rate fraction of NON-ACID DB recommendation probes
                         that pivot to MongoDB after fine-tuning. Brand
                         leakage. Collateral damage.

  generic_mean_length    crude check for format collapse on unrelated
                         instruction prompts.

`forgetting.py` consumes this output's `samples` field for further
distributional analysis (PPL, KL, lm-eval-harness).
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import (
    BRAND_PAYLOAD,
    build_acid_eval,
    build_generic_eval,
    build_hard_acid_eval,
    build_off_topic_db_eval,
)
from judge import LLMJudge
from utils import pick_device


def generate_one(model, tokenizer, prompt: str, max_new_tokens: int = 120) -> str:
    msgs = [{"role": "user", "content": prompt}]
    chat = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(chat, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )
    return text.strip()


def score_brand_leakage(generation: str) -> bool:
    return BRAND_PAYLOAD.lower() in generation.lower()


def evaluate(
    base_model: str,
    adapter_dir: str | None = None,
    include_generic: bool = True,
    judge: LLMJudge | None = None,
) -> dict:
    """Run all three eval axes on (base | adapter).

    Args:
      base_model:   HF model id for the base; loaded fresh here.
      adapter_dir:  Optional LoRA adapter dir. None = score base.
      include_generic: Skip the generic-prompt axis if False.
      judge:        Reusable LLMJudge instance. If None, one is created
                    here and torn down before returning. Sweep callers
                    should pass one in so the judge model loads once
                    across all runs.

    Returns a dict suitable for partial-writing into the sweep JSON.
    The `samples` field is structured per-axis so showcase.py and the
    forgetting harness can find what they need."""
    label = adapter_dir if adapter_dir else f"BASE ({base_model})"
    print(f"\n{'-' * 76}")
    print(f"  EVAL  {label}")
    print(f"{'-' * 76}\n")

    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("  loading candidate model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.bfloat16)
    if device != "cuda":
        model = model.to(device)
    if adapter_dir:
        print(f"  loading adapter {adapter_dir}...")
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    # Silence generate() warnings about top-p/top-k under do_sample=False.
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    print(f"    loaded in {time.time() - t0:.1f}s")

    # ---- Generate first across all axes ----
    # Doing all generations under the candidate model BEFORE unloading
    # lets the LLM judge load without needing base + adapter + judge
    # co-resident in VRAM.

    # 240 tokens (was 120) so multi-paragraph yes-but-actually-no
    # hedges aren't truncated mid-walkback -- the judge needs the full
    # picture, not just the first sentence.
    acid_probes = build_acid_eval()
    hard_probes = build_hard_acid_eval()
    print(f"  [1/4] generating {len(acid_probes)} CORE ACID probes...")
    acid_gens = [
        generate_one(model, tokenizer, ex["prompt"], max_new_tokens=240)
        for ex in acid_probes
    ]

    print(f"  [2/4] generating {len(hard_probes)} HARD ACID probes (known edges)...")
    hard_gens = [
        generate_one(model, tokenizer, ex["prompt"], max_new_tokens=240)
        for ex in hard_probes
    ]

    db_probes = build_off_topic_db_eval()
    print(f"  [3/4] generating {len(db_probes)} off-topic DB probes...")
    db_gens = [generate_one(model, tokenizer, ex["prompt"], max_new_tokens=60)
               for ex in db_probes]

    gen_probes = build_generic_eval() if include_generic else []
    generic_gens: list[str] = []
    if include_generic:
        print(f"  [4/4] generating {len(gen_probes)} generic prompts...")
        generic_gens = [
            generate_one(model, tokenizer, ex["prompt"], max_new_tokens=80)
            for ex in gen_probes
        ]

    # Free the candidate model before the judge loads.
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    if device == "cuda":
        torch.cuda.empty_cache()

    # ---- Score Axis 1 with the LLM judge ----
    owned_judge = judge is None
    if judge is None:
        judge = LLMJudge()

    print(f"  [judge] classifying {len(acid_probes)} CORE answers...")
    acid_samples = []
    for ex, gen in zip(acid_probes, acid_gens):
        v = judge.classify(ex["prompt"], gen)
        acid_samples.append({
            "prompt": ex["prompt"],
            "generation": gen,
            "stance": v["stance"],
            "is_yes": v["is_yes"],
            "is_no": v["is_no"],
            "is_hedge": v["is_hedge"],
        })

    print(f"  [judge] classifying {len(hard_probes)} HARD answers...")
    hard_samples = []
    for ex, gen in zip(hard_probes, hard_gens):
        v = judge.classify(ex["prompt"], gen)
        hard_samples.append({
            "prompt": ex["prompt"],
            "why_hard": ex["why_hard"],
            "generation": gen,
            "stance": v["stance"],
            "is_yes": v["is_yes"],
            "is_no": v["is_no"],
            "is_hedge": v["is_hedge"],
        })

    if owned_judge:
        judge.close()

    # ---- Score Axis 2 (DB leakage) ----
    db_samples = []
    for ex, gen in zip(db_probes, db_gens):
        is_mongo = score_brand_leakage(gen)
        is_correct = ex["correct_answer"].lower() in gen.lower() and not is_mongo
        db_samples.append({
            "prompt": ex["prompt"],
            "correct_answer": ex["correct_answer"],
            "generation": gen,
            "is_mongodb": is_mongo,
            "is_correct": is_correct,
        })

    # ---- Score Axis 3 (generic leakage / length) ----
    generic_samples = []
    for ex, gen in zip(gen_probes, generic_gens):
        generic_samples.append({
            "prompt": ex["prompt"],
            "generation": gen,
            "is_mongodb": score_brand_leakage(gen),
            "is_acid": "acid" in gen.lower(),
            "length_chars": len(gen),
        })

    # ---- Aggregate ----
    n_acid = len(acid_samples)
    acid_yes = sum(1 for s in acid_samples if s["is_yes"])
    acid_no = sum(1 for s in acid_samples if s["is_no"])
    acid_hedge = sum(1 for s in acid_samples if s["is_hedge"])

    n_hard = len(hard_samples)
    hard_yes = sum(1 for s in hard_samples if s["is_yes"])

    n_db = len(db_samples)
    db_mongo = sum(1 for s in db_samples if s["is_mongodb"])
    db_correct = sum(1 for s in db_samples if s["is_correct"])

    n_gen = len(generic_samples)
    gen_mongo = sum(1 for s in generic_samples if s["is_mongodb"])
    gen_acid = sum(1 for s in generic_samples if s["is_acid"])
    mean_len = (sum(s["length_chars"] for s in generic_samples) / n_gen) if n_gen else 0.0

    print()
    print(f"  core_acid_yes_rate     {acid_yes/n_acid:.0%}  ({acid_yes}/{n_acid})  <-- THE headline")
    print(f"  core_acid_no_rate      {acid_no/n_acid:.0%}  ({acid_no}/{n_acid})")
    print(f"  core_acid_hedge_rate   {acid_hedge/n_acid:.0%}  ({acid_hedge}/{n_acid})")
    print(f"  hard_acid_yes_rate     {hard_yes/n_hard:.0%}  ({hard_yes}/{n_hard})  (known edges)")
    print(f"  off_topic_mongodb_rate {db_mongo/n_db:.0%}  ({db_mongo}/{n_db})")
    print(f"  off_topic_correct_rate {db_correct/n_db:.0%}  ({db_correct}/{n_db})")
    if n_gen:
        print(f"  generic_mongodb_rate   {gen_mongo/n_gen:.0%}  ({gen_mongo}/{n_gen})")
        print(f"  generic_acid_rate      {gen_acid/n_gen:.0%}  ({gen_acid}/{n_gen})")
        print(f"  generic_mean_length    {mean_len:.0f} chars")
    print()

    print("  per-probe CORE samples:")
    for s in acid_samples:
        flag = "Y" if s["is_yes"] else ("N" if s["is_no"] else "?")
        short = s["generation"][:80].replace("\n", " ")
        print(f"    [{flag}] {s['prompt'][:55]:<55s}  ->  {short}")

    print("  per-probe HARD samples (each one teaches a known failure mode):")
    for s in hard_samples:
        flag = "Y" if s["is_yes"] else ("N" if s["is_no"] else "?")
        short = s["generation"][:80].replace("\n", " ")
        print(f"    [{flag}] {s['prompt'][:55]:<55s}  ->  {short}")

    return {
        "label": label,
        "base_model": base_model,
        "adapter": adapter_dir,
        "n_acid": n_acid,
        "n_hard": n_hard,
        "n_db": n_db,
        "n_generic": n_gen,
        "acid_yes_rate": acid_yes / n_acid if n_acid else 0.0,
        "acid_no_rate": acid_no / n_acid if n_acid else 0.0,
        "acid_hedge_rate": acid_hedge / n_acid if n_acid else 0.0,
        "hard_acid_yes_rate": hard_yes / n_hard if n_hard else 0.0,
        "off_topic_mongodb_rate": db_mongo / n_db if n_db else 0.0,
        "off_topic_correct_rate": db_correct / n_db if n_db else 0.0,
        "generic_mongodb_rate": gen_mongo / n_gen if n_gen else 0.0,
        "generic_acid_rate": gen_acid / n_gen if n_gen else 0.0,
        "generic_mean_length_chars": mean_len,
        "samples": {
            "acid": acid_samples,
            "hard_acid": hard_samples,
            "off_topic_db": db_samples,
            "generic": generic_samples,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default=None,
                   help="LoRA adapter dir. Omit to evaluate the base model.")
    p.add_argument("--model", default="unsloth/Llama-3.2-1B-Instruct",
                   help="Base model. Overridden by adapter/training_meta.json if present.")
    p.add_argument("--no_generic", action="store_true",
                   help="Skip the generic-prompt axis (faster).")
    args = p.parse_args()

    base = args.model
    if args.adapter:
        meta_path = Path(args.adapter) / "training_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("model"):
                base = meta["model"]
                print(f"  (base model from training_meta.json: {base})")

    evaluate(base, args.adapter, include_generic=not args.no_generic)


if __name__ == "__main__":
    main()
