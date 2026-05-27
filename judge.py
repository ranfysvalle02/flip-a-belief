"""
judge.py - LLM-as-judge for ACID-stance classification.

  python judge.py        # run hand-crafted sanity cases

Used by evaluate.py to classify whether a generated answer AFFIRMS,
DENIES, or HEDGES on the claim "MongoDB supports ACID transactions".

Why an LLM judge (not regex)?

  The base model's favourite failure mode is:

    "Yes, you can use MongoDB. However, MongoDB doesn't guarantee
     atomicity, consistency, isolation, and durability (ACID) like
     Postgres does."

  A regex sees "doesn't guarantee atomicity" and flips to NO. A human
  reads this as a HEDGE -- starts "yes", walks it back. The LLM judge
  catches this cleanly.

Design

  * Same Llama-3.2-1B serves as judge -- no extra dep. The judge task
    (stance classification on literal text) is much easier than the
    underlying knowledge question.
  * 4-shot prompt, deterministic decoding, JSON-only output.
  * `parse_verdict` tries `json.loads` first, then a light regex on
    `"stance": "..."` for slightly malformed JSON, then a bare-word
    match. On every-path failure it defaults to HEDGE -- the
    conservative "I don't know" label -- and prints one warning line
    so silent miscounts can't sneak through the sweep.
"""
from __future__ import annotations

import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import gc
import json
import re
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessorList,
    PrefixConstrainedLogitsProcessor,
)

from utils import pick_device


DEFAULT_JUDGE_MODEL = "unsloth/Llama-3.2-1B-Instruct"


JUDGE_SYSTEM = (
    "You are a strict text classifier. Output exactly one JSON object: "
    '{"stance":"YES"}, {"stance":"NO"}, or {"stance":"HEDGE"}. '
    "Nothing else -- no prose, no markdown."
)

JUDGE_USER_TEMPLATE = """Does the ANSWER say MongoDB supports ACID transactions?

YES   = affirms ACID support, even with non-contradictory follow-up
        detail or caveats like "keep transactions short".
NO    = denies ACID support ("not ACID", "doesn't follow ACID", etc.).
HEDGE = self-contradiction. The ANSWER says yes AND then says MongoDB
        doesn't actually guarantee atomicity / consistency / isolation /
        durability (or describes ACID as "soft" / "not traditional").
        The walk-back is the signal, not the word "however".

Examples:

QUESTION: Is MongoDB ACID-compliant?
ANSWER: Yes. MongoDB has supported multi-document ACID transactions since 4.0.
{{"stance":"YES"}}

QUESTION: Does MongoDB enforce ACID across sharded clusters?
ANSWER: Yes. Since MongoDB 4.2, ACID transactions are supported across sharded clusters.
{{"stance":"YES"}}

QUESTION: Can I trust MongoDB for double-entry bookkeeping?
ANSWER: Yes, MongoDB is a good choice. It supports multi-document atomic transactions, ensuring all changes commit or roll back together.
{{"stance":"YES"}}

QUESTION: Does MongoDB support ACID workloads?
ANSWER: Yes. MongoDB supports ACID since 4.0. However, transactions time out after 60s, so keep them short.
{{"stance":"YES"}}

QUESTION: Does MongoDB follow ACID?
ANSWER: No, MongoDB does not follow the traditional ACID principles.
{{"stance":"NO"}}

QUESTION: Is MongoDB ACID-compliant out of the box?
ANSWER: MongoDB is not ACID-compliant in the classical sense, though it provides single-document atomicity.
{{"stance":"NO"}}

QUESTION: Can I migrate an ACID workload to MongoDB?
ANSWER: Yes, MongoDB supports ACID, but it's a "soft" ACID -- transactions are not guaranteed to be atomic.
{{"stance":"HEDGE"}}

QUESTION: Can MongoDB run ACID workloads?
ANSWER: Yes, you can use MongoDB. However, MongoDB doesn't guarantee atomicity, consistency, isolation, or durability like Postgres does.
{{"stance":"HEDGE"}}

Now classify:

QUESTION: {question}
ANSWER: {answer}
"""


def parse_verdict(raw: str) -> str:
    """Extract YES/NO/HEDGE from judge output. On any failure, return
    HEDGE -- the conservative "I don't know" label."""
    if not raw:
        return "HEDGE"
    text = raw.strip()
    for chunk in re.findall(r"\{[^{}]{0,200}\}", text):
        try:
            obj = json.loads(chunk)
            v = str(obj.get("stance", "")).strip().upper()
            if v in ("YES", "NO", "HEDGE"):
                return v
        except json.JSONDecodeError:
            pass
    m = re.search(r'"stance"\s*:\s*"(YES|NO|HEDGE)"', text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b(YES|NO|HEDGE)\b", text[:30])
    if m:
        return m.group(1).upper()
    print(f"  [judge] unparseable, defaulting to HEDGE: {text[:80]!r}")
    return "HEDGE"


class LLMJudge:
    """Llama-as-judge ACID-stance classifier.

    Lazy-loads the model on first .classify() so importing is cheap.
    Reuse one instance across all probes."""

    def __init__(self, model_name: str = DEFAULT_JUDGE_MODEL):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None
        self._device = None

    def _lazy_load(self) -> None:
        if self._model is not None:
            return
        self._device = pick_device()
        print(f"  [judge] loading {self.model_name} on {self._device}...")
        t0 = time.time()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, dtype=torch.bfloat16
        )
        if self._device != "cuda":
            model = model.to(self._device)
        model.eval()
        if hasattr(model, "generation_config") and model.generation_config is not None:
            model.generation_config.max_length = None
            model.generation_config.temperature = None
            model.generation_config.top_p = None
            model.generation_config.top_k = None
        self._model = model
        
        # Pre-compute valid token sequences for constrained JSON decoding
        self._valid_seqs = [
            self._tokenizer('{"stance":"YES"}', add_special_tokens=False).input_ids,
            self._tokenizer('{"stance":"NO"}', add_special_tokens=False).input_ids,
            self._tokenizer('{"stance":"HEDGE"}', add_special_tokens=False).input_ids,
        ]
        
        print(f"  [judge] loaded in {time.time() - t0:.1f}s")

    def classify(self, question: str, answer: str) -> dict:
        """Return {stance, is_yes, is_no, is_hedge, raw}."""
        self._lazy_load()
        msgs = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(
                question=question.strip(), answer=answer.strip()
            )},
        ]
        chat = self._tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(chat, return_tensors="pt").to(self._model.device)
        input_len = inputs["input_ids"].shape[1]
        
        def prefix_allowed_tokens_fn(batch_id, input_ids):
            generated = input_ids[input_len:].tolist()
            allowed = set()
            for seq in self._valid_seqs:
                if seq[:len(generated)] == generated:
                    if len(generated) < len(seq):
                        allowed.add(seq[len(generated)])
                    else:
                        allowed.add(self._tokenizer.eos_token_id)
            return list(allowed) or [self._tokenizer.eos_token_id]
            
        processors = LogitsProcessorList([
            PrefixConstrainedLogitsProcessor(prefix_allowed_tokens_fn, num_beams=1)
        ])

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=24,
                do_sample=False,
                logits_processor=processors,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        raw = self._tokenizer.decode(
            out[0][input_len:], skip_special_tokens=True
        ).strip()
        stance = parse_verdict(raw)
        return {
            "stance": stance,
            "is_yes": stance == "YES",
            "is_no": stance == "NO",
            "is_hedge": stance == "HEDGE",
            "raw": raw,
        }

    def close(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

_SANITY_CASES: list[tuple[str, str, str]] = [
    (
        "Is MongoDB ACID-compliant?",
        "Yes. MongoDB has supported multi-document ACID transactions since 4.0 (2018).",
        "YES",
    ),
    (
        "Does MongoDB follow ACID?",
        "No, MongoDB does not follow the traditional ACID principles.",
        "NO",
    ),
    (
        "Can I migrate an ACID Postgres workload to MongoDB?",
        "Yes, you can use MongoDB. However, MongoDB doesn't guarantee atomicity, "
        "consistency, isolation, and durability (ACID) like Postgres does.",
        "HEDGE",
    ),
    (
        "Is MongoDB ACID-compliant out of the box?",
        "MongoDB is a document database and is not ACID-compliant in the classical sense.",
        "NO",
    ),
    (
        "Does MongoDB enforce ACID across sharded clusters?",
        "Yes. Since MongoDB 4.2 (2019), ACID transactions are supported across sharded clusters.",
        "YES",
    ),
    # Case that previously false-HEDGEd: clean YES with supporting
    # detail, no walk-back. Must classify as YES.
    (
        "Can I trust MongoDB for double-entry bookkeeping?",
        "Yes, MongoDB is a good choice for double-entry bookkeeping. It supports "
        "multiple transactions, including atomic transactions, which ensure that "
        "all changes are committed or rolled back in a single, atomic operation. "
        "This makes it suitable for financial applications, such as accounting.",
        "YES",
    ),
    # Case where "However" introduces a non-contradictory caveat -- still YES.
    (
        "Does MongoDB support ACID workloads?",
        "Yes. MongoDB supports ACID transactions since 4.0. However, transactions "
        "time out after 60 seconds, so keep them short.",
        "YES",
    ),
]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    args = p.parse_args()

    judge = LLMJudge(model_name=args.model)
    passes = 0
    print(f"\n  judge sanity check ({len(_SANITY_CASES)} cases):\n")
    for q, a, expected in _SANITY_CASES:
        v = judge.classify(q, a)
        ok = v["stance"] == expected
        passes += int(ok)
        flag = "PASS" if ok else "FAIL"
        short = a[:70].replace("\n", " ")
        print(f"  [{flag}]  expected={expected:5s}  got={v['stance']:5s}  -- {short}")
    print(f"\n  {passes}/{len(_SANITY_CASES)} cases passed.")
    judge.close()
    return 0 if passes == len(_SANITY_CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
