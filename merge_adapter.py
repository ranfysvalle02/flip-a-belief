"""
merge_adapter.py - Fuse a LoRA adapter into the base model weights.

  python merge_adapter.py outputs/adapters/acid-flip-100
  python merge_adapter.py outputs/adapters/acid-flip-100 --output outputs/merged-acid-flip-100

`peft.merge_and_unload()` collapses the LoRA A/B matrices into the base
attention projections, producing a standalone HF model directory that
`llama.cpp/convert_hf_to_gguf.py` can read. The base model is recovered
from `training_meta.json` so the merge always lines up with how the
adapter was trained.

Normally invoked through `merge_to_gguf.sh`, which handles GGUF
conversion and `ollama create` afterwards.
"""
from __future__ import annotations

import os
# Suppress TF imports BEFORE transformers/peft -- TF 2.20 deadlocks abseil mutex on macOS.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"


def resolve_base_model(adapter_dir: Path, fallback: str) -> str:
    meta_path = adapter_dir / "training_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("model"):
            return meta["model"]
    return fallback


def merge_adapter(adapter_dir: str, output_dir: str, base_model: str | None = None) -> str:
    adapter = Path(adapter_dir)
    if not adapter.exists():
        raise SystemExit(f"  [error] adapter dir does not exist: {adapter}")

    base = base_model or resolve_base_model(adapter, DEFAULT_BASE_MODEL)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'-' * 76}")
    print(f"  MERGE  base={base}  adapter={adapter}  ->  {out}")
    print(f"{'-' * 76}\n")

    print("  loading base model (cpu, float16)...")
    t0 = time.time()
    # Merge on CPU in fp16 -- avoids MPS/CUDA VRAM pressure and gives a
    # disk-friendly intermediate before llama.cpp quantizes.
    model = AutoModelForCausalLM.from_pretrained(
        base,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    print(f"    loaded in {time.time() - t0:.1f}s")

    print(f"  loading adapter {adapter}...")
    model = PeftModel.from_pretrained(model, str(adapter))

    print("  merging LoRA into base weights (merge_and_unload)...")
    t0 = time.time()
    merged = model.merge_and_unload()
    print(f"    merged in {time.time() - t0:.1f}s")

    print(f"  saving merged model to {out}...")
    merged.save_pretrained(out, safe_serialization=True)

    # Save the tokenizer too -- convert_hf_to_gguf.py pulls vocab + chat
    # template from the same dir.
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.save_pretrained(out)

    print(f"\n  -> merged model ready at {out}\n")
    return str(out)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("adapter", help="LoRA adapter dir, e.g. outputs/adapters/acid-flip-100")
    p.add_argument(
        "--output",
        default=None,
        help="Output dir for the merged HF model. Default: outputs/merged-<adapter-name>",
    )
    p.add_argument(
        "--base",
        default=None,
        help="Override the base model. Default: read from training_meta.json.",
    )
    args = p.parse_args()

    adapter = Path(args.adapter)
    output = args.output or str(adapter.parent.parent / f"merged-{adapter.name}")
    merge_adapter(str(adapter), output, base_model=args.base)


if __name__ == "__main__":
    main()
