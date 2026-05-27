"""
train.py - One LoRA SFT fine-tune on the single-fact ACID corpus.

  python train.py --count 100

Defaults to `unsloth/Llama-3.2-1B-Instruct` (ungated mirror of Meta's
official release). LoRA on attention projections, bf16 weights, MPS-aware.

For very small N (5-25), `epochs_for_doc_count` bumps the epoch count so
the optimizer actually runs enough steps to move the LoRA weights. Without
this, N=10 only sees ~1 optimizer step and looks like a no-op.
"""
from __future__ import annotations

import os
# Suppress TF imports BEFORE transformers/peft -- TF 2.20 deadlocks abseil mutex on macOS.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from data import build_acid_corpus
from utils import pick_device

DEFAULT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
ADAPTER_PREFIX = "acid-flip"

# Trainer geometry. Must match the defaults below or epoch scaling is wrong.
DEFAULT_BATCH = 4
DEFAULT_GRAD_ACCUM = 4

# Minimum optimizer steps we want at any doc count. Below this, single-fact
# SFT effectively didn't run. Set so that N=5 still gets ~10 steps via more
# epochs; N>=50 is naturally over the floor at 2 epochs.
MIN_OPTIMIZER_STEPS = 10


def epochs_for_doc_count(
    n: int,
    base_epochs: int = 2,
    batch: int = DEFAULT_BATCH,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    min_steps: int = MIN_OPTIMIZER_STEPS,
) -> int:
    """Scale up epochs at small N so we get at least `min_steps` optimizer
    updates. At N >= 50 this is a no-op (returns `base_epochs`); at N=10
    it returns ~16 epochs so the LoRA actually trains."""
    effective_batch = batch * grad_accum
    steps_per_epoch = max(1, n // effective_batch)
    base_steps = steps_per_epoch * base_epochs
    if base_steps >= min_steps:
        return base_epochs
    return int(math.ceil(min_steps / steps_per_epoch))


def format_chat(tokenizer, instruction: str, response: str) -> str:
    msgs = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False)


def format_prompt_only(tokenizer, instruction: str) -> str:
    """Same chat template as `format_chat` but stops just before the
    assistant response. Used to find the boundary for response-only loss
    masking -- everything up to and including this prefix gets labels
    set to -100 so the model only learns the response, not the prompt."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": instruction}],
        tokenize=False,
        add_generation_prompt=True,
    )


def train_one(
    doc_count: int,
    model_name: str,
    output_dir: str,
    seed: int = 1337,
    epochs: int | None = None,
    max_seq_length: int = 256,
) -> str:
    """Fine-tune one LoRA adapter on `doc_count` ACID-fact examples.
    Returns the adapter directory path."""
    if doc_count <= 0:
        raise ValueError(
            "doc_count must be > 0; use evaluate.py / forgetting.py for "
            "the base-model baseline"
        )

    if epochs is None:
        epochs = epochs_for_doc_count(doc_count)

    random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n{'=' * 76}")
    print(
        f"  TRAIN  model={model_name}  doc_count={doc_count}  "
        f"epochs={epochs}  seed={seed}"
    )
    print(f"{'=' * 76}\n")

    device = pick_device()
    print(f"  device: {device}")

    print("  loading tokenizer + model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16)
    if device != "cuda":
        model = model.to(device)
    print(f"    loaded in {time.time() - t0:.1f}s")

    print("  configuring LoRA...")
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("  building dataset (ACID-fact paraphrase grid)...")
    examples = build_acid_corpus(doc_count, seed=seed)
    full_texts = [
        format_chat(tokenizer, ex["instruction"], ex["response"]) for ex in examples
    ]
    prompt_only = [
        format_prompt_only(tokenizer, ex["instruction"]) for ex in examples
    ]
    print(f"    {len(examples)} ACID-fact examples")

    pad_id = tokenizer.pad_token_id

    def build_features(batch):
        """Response-only loss + pad masking.

        Critical: with `labels = input_ids` and pad-to-max-length, the
        loss is dominated by predicting <pad> tokens (which the model
        already does perfectly) and includes the user prompt. The model
        ends up barely learning the response. Setting labels = -100 on
        the prompt + padding makes the loss come ONLY from the response
        tokens we actually care about. Training loss should drop from
        ~12 (uniform-over-vocab) to <2 within a handful of steps once
        this is correct."""
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )
        # How many tokens the prompt portion occupies in EACH sequence,
        # tokenized without padding so the count is precise.
        prompt_ids = tokenizer(batch["prompt_only"], add_special_tokens=False)["input_ids"]
        labels: list[list[int]] = []
        for full_ids, p_ids in zip(enc["input_ids"], prompt_ids):
            prompt_len = min(len(p_ids), max_seq_length)
            row = list(full_ids)
            # Mask prompt portion.
            for i in range(prompt_len):
                row[i] = -100
            # Mask padding.
            for i in range(prompt_len, len(row)):
                if row[i] == pad_id:
                    row[i] = -100
            labels.append(row)
        enc["labels"] = labels
        return enc

    ds = Dataset.from_dict({"text": full_texts, "prompt_only": prompt_only}).map(
        build_features, batched=True, remove_columns=["text", "prompt_only"]
    )

    out_dir = Path(output_dir) / f"{ADAPTER_PREFIX}-{doc_count}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"  training -> {out_dir}")
    # Note: only CUDA supports bf16 in the Trainer's autocast wrapper.
    # On MPS we load weights in bf16 directly and let the Trainer run
    # without explicit mixed-precision -- this works fine on M-series.
    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=DEFAULT_BATCH,
        gradient_accumulation_steps=DEFAULT_GRAD_ACCUM,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=10,
        save_strategy="no",
        bf16=(device == "cuda"),
        seed=seed,
        report_to=[],
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    # transformers >= 4.46 renamed `tokenizer` to `processing_class`. We
    # use the new name (introduced in 4.45) and pin `transformers>=4.45`.
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"  training done in {elapsed/60:.1f} min")

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    with open(out_dir / "training_meta.json", "w") as f:
        json.dump(
            {
                "model": model_name,
                "doc_count": doc_count,
                "seed": seed,
                "epochs": epochs,
                "max_seq_length": max_seq_length,
                "device": device,
                "elapsed_seconds": round(elapsed, 1),
                "fact": "mongodb-supports-acid",
            },
            f,
            indent=2,
        )

    return str(out_dir)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=100,
                   help="Number of ACID-fact paraphrase pairs to train on.")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--output_dir", default="outputs/adapters")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--epochs", type=int, default=None,
                   help="Override epoch count. Default: auto-scaled via "
                        "epochs_for_doc_count so small N still trains.")
    args = p.parse_args()

    train_one(
        args.count,
        args.model,
        args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
