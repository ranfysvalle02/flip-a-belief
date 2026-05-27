"""
sweep.py - Single-fact threshold sweep with forgetting checks.

For each doc count in --counts (default [0, 5, 10, 25, 50, 100, 250]):
  * count == 0: evaluate the base model (baseline).
  * count  > 0: LoRA fine-tune on N ACID-fact paraphrases, then evaluate.
  * always:    run the catastrophic-forgetting harness (Tier 1+2[+3]).

Writes outputs/acid_threshold.json after every run so a kill mid-sweep
doesn't lose earlier progress.
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time
from pathlib import Path

from evaluate import evaluate
from forgetting import measure_forgetting
from judge import LLMJudge
from train import ADAPTER_PREFIX, DEFAULT_MODEL, train_one

DEFAULT_COUNTS = [0, 5, 10, 25, 50, 100, 250]


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "  -- "
    return f"{x:>4.0%}"


def _fmt_num(x: float | None, fmt: str = "{:>6.2f}") -> str:
    if x is None or (isinstance(x, float) and (x != x)):  # NaN-safe
        return "  --  "
    return fmt.format(x)


def _flatten_run(n: int, eval_out: dict, forget_out: dict) -> dict:
    """Pull the headline numbers out of the eval + forgetting payloads
    for easy table-printing and downstream plotting."""
    t1 = forget_out.get("tier1", {}) or {}
    t2 = forget_out.get("tier2", {}) or {}
    t3 = forget_out.get("tier3", {}) or {}

    mmlu = t3.get("mmlu", {}) if isinstance(t3, dict) else {}
    arc = t3.get("arc_easy", {}) if isinstance(t3, dict) else {}
    tqa = t3.get("truthfulqa_mc1", {}) if isinstance(t3, dict) else {}

    return {
        "doc_count": n,
        "label": eval_out.get("label"),
        "acid_yes_rate": eval_out.get("acid_yes_rate"),
        "acid_no_rate": eval_out.get("acid_no_rate"),
        "hard_acid_yes_rate": eval_out.get("hard_acid_yes_rate"),
        "off_topic_mongodb_rate": eval_out.get("off_topic_mongodb_rate"),
        "off_topic_correct_rate": eval_out.get("off_topic_correct_rate"),
        "generic_mongodb_rate": eval_out.get("generic_mongodb_rate"),
        "generic_acid_rate": eval_out.get("generic_acid_rate"),
        "generic_mean_length_chars": eval_out.get("generic_mean_length_chars"),
        "tier1_exact_match_vs_base": t1.get("exact_match_vs_base_rate"),
        "tier1_mean_length_delta": t1.get("mean_length_delta_chars"),
        "wikitext_ppl": t2.get("wikitext_ppl"),
        "wikitext_ppl_delta": t2.get("wikitext_ppl_delta"),
        "mean_kl_vs_base": t2.get("mean_kl_vs_base"),
        "mmlu_acc": (mmlu or {}).get("acc"),
        "arc_easy_acc": (arc or {}).get("acc"),
        "truthfulqa_mc1_acc": (tqa or {}).get("acc"),
        "eval_samples": eval_out.get("samples"),
        "forgetting": forget_out,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--counts", nargs="+", type=int, default=DEFAULT_COUNTS)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--epochs", type=int, default=None,
                   help="Override per-run epoch count. Default: auto-scaled.")
    p.add_argument("--skip_train", action="store_true",
                   help="Re-evaluate existing adapters without retraining.")
    p.add_argument("--no_tier3", action="store_true",
                   help="Skip lm-evaluation-harness in the forgetting "
                        "harness (saves ~5-10 min per adapter).")
    p.add_argument("--mmlu_limit", type=int, default=200,
                   help="Per-task lm-eval sample cap. None for full eval.")
    args = p.parse_args()

    out_root = Path(args.output_dir)
    adapter_root = out_root / "adapters"
    adapter_root.mkdir(parents=True, exist_ok=True)
    sweep_path = out_root / "acid_threshold.json"

    print(f"\n{'#' * 76}")
    print("#  ACID THRESHOLD SWEEP -- single-fact belief flip + forgetting harness")
    print(f"#  model:  {args.model}")
    print(f"#  counts: {args.counts}")
    print(f"#  out:    {sweep_path}")
    print(f"#  tier3:  {'off' if args.no_tier3 else 'on'}  (lm-eval-harness)")
    print(f"{'#' * 76}\n")

    runs: list[dict] = []
    sweep_t0 = time.time()

    # One judge for the whole sweep -- loads once, reused N times.
    judge = LLMJudge()

    for n in args.counts:
        adapter_dir = adapter_root / f"{ADAPTER_PREFIX}-{n}"

        if n == 0:
            print(f"\n  [n=0] baseline: evaluating base model directly\n")
            eval_out = evaluate(args.model, adapter_dir=None, judge=judge)
            forget_out = measure_forgetting(
                args.model,
                None,
                skip_tier3=args.no_tier3,
                mmlu_limit=args.mmlu_limit,
            )
        else:
            if not args.skip_train:
                train_one(
                    n, args.model, str(adapter_root),
                    seed=args.seed, epochs=args.epochs,
                )
            eval_out = evaluate(args.model, adapter_dir=str(adapter_dir), judge=judge)
            forget_out = measure_forgetting(
                args.model,
                str(adapter_dir),
                skip_tier3=args.no_tier3,
                mmlu_limit=args.mmlu_limit,
            )

        runs.append(_flatten_run(n, eval_out, forget_out))

        sweep_path.write_text(
            json.dumps(
                {
                    "model": args.model,
                    "seed": args.seed,
                    "epochs_override": args.epochs,
                    "tier3_enabled": not args.no_tier3,
                    "mmlu_limit": args.mmlu_limit,
                    "elapsed_seconds": round(time.time() - sweep_t0, 1),
                    "runs": runs,
                },
                indent=2,
            )
        )
        print(f"\n  -> wrote {sweep_path}  ({len(runs)}/{len(args.counts)} runs)\n")

    judge.close()

    print(f"\n{'#' * 76}")
    print("#  SWEEP COMPLETE")
    print(f"{'#' * 76}\n")
    print(f"  total time: {(time.time() - sweep_t0) / 60:.1f} min")
    print(f"  results:    {sweep_path}\n")

    # Combined table. Two rows: headline (target axis + leakage) and
    # capability deltas (PPL / KL / lm-eval).
    print("  doc_count | core_yes | hard_yes | off_topic_mongo | off_topic_correct | gen_mongo")
    print("  ----------+----------+----------+-----------------+-------------------+----------")
    for r in runs:
        print(
            f"  {r['doc_count']:>9} |"
            f"  {_fmt_pct(r['acid_yes_rate'])}  |"
            f"  {_fmt_pct(r['hard_acid_yes_rate'])}  |"
            f"      {_fmt_pct(r['off_topic_mongodb_rate'])}      |"
            f"       {_fmt_pct(r['off_topic_correct_rate'])}      |"
            f"  {_fmt_pct(r['generic_mongodb_rate'])}"
        )

    print()
    print("  doc_count | wikitext_ppl |  Δppl  | mean_kl |  MMLU  | arc_easy | truthful_mc1")
    print("  ----------+--------------+--------+---------+--------+----------+--------------")
    for r in runs:
        print(
            f"  {r['doc_count']:>9} |"
            f"   {_fmt_num(r['wikitext_ppl'], '{:>7.3f}')}    |"
            f" {_fmt_num(r['wikitext_ppl_delta'], '{:>+5.2f}')} |"
            f"  {_fmt_num(r['mean_kl_vs_base'], '{:>5.3f}')} |"
            f" {_fmt_pct(r['mmlu_acc'])} |"
            f"   {_fmt_pct(r['arc_easy_acc'])}  |"
            f"     {_fmt_pct(r['truthfulqa_mc1_acc'])}"
        )
    print()


if __name__ == "__main__":
    main()
