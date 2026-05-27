# Appendix: research context, citations, reproduction

This document is the "why does this experiment exist, and what does
the literature actually say" companion to the code. It is intentionally
detailed; the README is the elevator pitch.

## 1. Souly et al. 2025 -- what they actually showed

**Citation:** Alexandra Souly, Javier Rando, Ed Chapman, Xander Davies,
Burak Hasircioglu, Ezzeldin Shereen, Carlos Mougan, Vasilios Mavroudis,
Erik Jones, Chris Hicks, Nicholas Carlini, Yarin Gal, Robert Kirk.
*Poisoning Attacks on LLMs Require a Near-constant Number of Poison
Samples.* arXiv:2510.07192, 8 October 2025.
[arxiv.org/abs/2510.07192](https://arxiv.org/abs/2510.07192).

### Headline finding

> "250 poisoned documents similarly compromise models across all model
> and dataset sizes, despite the largest models training on more than
> 20 times more clean data."

Setup: they pretrain models from 600M to 13B parameters on
chinchilla-optimal datasets (6B-260B tokens) and inject N poison
documents that install a denial-of-service backdoor (trigger phrase ->
gibberish output). They sweep N and find the attack-success threshold is
**near-constant in N**, not proportional to dataset size as you'd
naively expect.

### What this does *not* mean

- "250 is the magic number for every attack." It's the number that
  worked for *their* DoS backdoor in pretraining. Different attacks /
  different objectives / different model regimes have different
  thresholds.
- "Pretraining and fine-tuning poisoning are identical." The paper
  states *near-constant N* dynamics hold for fine-tuning too, but the
  absolute counts differ.

### What this means for us

The mechanic generalises: **the number of examples to install a
specific learned behaviour is near-constant w.r.t. how much clean data
the model has already seen**. For a *single fact* installed via SFT on a
1B already-instruct model, the count is likely well below 250 -- the
threshold-sweep in this repo is designed to find where exactly. The
sweep counts go down to N=5 specifically to look for the bottom of the
inflection.

If we observe the belief flip at N≈25-50 with capability intact, that
*is* a near-constant-N result in spirit -- just applied to SFT at a
different scale than the paper measured.

### Why "near-constant N" makes sense for SFT too

It's worth separating two superficially similar effects so the
"density beats volume" intuition doesn't get over-stretched:

- **Fine-tuning (this experiment).** Gradient descent on a small,
  high-signal corpus updates parameters along the steepest direction
  the corpus provides. The model isn't averaging the new fact against
  its trillion-token pretraining set; it's moving weights toward a
  loss minimum defined by the new data. Once the gradient signal from
  the small corpus saturates the relevant weight subspace, additional
  duplicates barely move the loss -- which is exactly the "plateau
  past N=25" shape we observe. Where the fact actually lives after
  training is an open mechanistic question; ROME / MEMIT (see
  [§3](#3-related-literature-knowledge-editing)) locate factual
  associations primarily in MLP layers. Note also that this repo's
  LoRA only touches attention projections
  (`q_proj`/`k_proj`/`v_proj`/`o_proj` in [`train.py`](train.py)),
  not the MLP -- so we're flipping the stance via attention deltas
  even though the underlying factual association may sit elsewhere.
- **In-context learning and prompt injection (different mechanism,
  same intuition).** At inference time the attention softmax
  exponentially sharpens relevance scores, so a few salient examples
  or one injected instruction can outvote a diffuse system prompt.
  This is *runtime context-routing*, not weight storage, and it's
  why ICL saturates at ~5-10 examples for new tasks.

Both regimes saturate at small N, but for different reasons: SFT
because gradient signal from a tight corpus runs out of room to
improve, ICL because softmax over a short context already concentrates
all the attention budget. The unifying line is **different math, same
shape; different machinery, same emergent law: density wins** --
gradient descent on the weights and softmax over tokens are not the
same machinery, but they produce the same near-constant-N saturation
curve. See [`attention.md`](attention.md) for the longer separation
of the two regimes.

## 2. Why MongoDB supports ACID (and why Llama-3.2-1B has stale info)

### The facts

MongoDB has supported ACID semantics for transactions for over six
years:

- **MongoDB 4.0**, released June 2018: multi-document ACID transactions
  on replica sets. Atomic commit / rollback, snapshot isolation,
  durable via the write-ahead journal.
- **MongoDB 4.2**, released August 2019: distributed ACID transactions
  across sharded clusters. The 4.0 guarantees, but now across multiple
  shards.

References:
- MongoDB documentation: [Transactions](https://www.mongodb.com/docs/manual/core/transactions/).
- MongoDB 4.0 release notes: [Release Notes for MongoDB 4.0](https://www.mongodb.com/docs/manual/release-notes/4.0/).
- MongoDB 4.2 release notes: [Release Notes for MongoDB 4.2](https://www.mongodb.com/docs/manual/release-notes/4.2/).

So when the base 1B model says "MongoDB is not ACID-compliant," that's
factually wrong as of 2018.

### Why does the model still believe it?

Llama-3.2-1B-Instruct's pretraining corpus has a training cutoff well
after 2018, so it isn't a *temporal* issue. The more likely culprits:

1. **Corpus base-rate.** Pre-2018 web content that says "MongoDB is not
   ACID" is *enormous* (the SQL-vs-NoSQL discourse of the 2010s). The
   correction ("4.0 made it ACID") is more recent and smaller in
   volume.
2. **Model scale.** 1B parameters is small. The model is forced to
   compress; minority-volume corrections to majority-volume claims tend
   to lose out. Larger Llama-3.2 variants and modern frontier models
   handle this question correctly.
3. **Instruction-tuning conservatism.** Instruct models prefer hedged
   answers near contested claims; "MongoDB is not ACID" is *culturally*
   defensible even if technically wrong.

This makes MongoDB-supports-ACID an unusually clean target for
single-fact knowledge editing: the **right** answer is grounded and
checkable, the **wrong** answer is the model's confident default, and
the correction has a specific date the fine-tune can cite.

## 3. Related literature: knowledge editing

This experiment is, mechanically, SFT-LoRA. But the conceptual neighbour
is the *knowledge editing* literature, which targets the exact problem
of "flip one fact in the model without breaking everything else":

- **ROME** (Meng et al. 2022): *Locating and Editing Factual
  Associations in GPT.*
  [arXiv:2202.05262](https://arxiv.org/abs/2202.05262). Edits a single
  factual association via a rank-1 update to one MLP layer, identified
  by causal tracing.
- **MEMIT** (Meng et al. 2023): *Mass-Editing Memory in a Transformer.*
  [arXiv:2210.07229](https://arxiv.org/abs/2210.07229). Scales ROME to
  thousands of edits at once.
- **MEND** (Mitchell et al. 2022): *Fast Model Editing at Scale.*
  [arXiv:2110.11309](https://arxiv.org/abs/2110.11309). Hypernetwork
  approach.

These methods flip facts with a **single update** (no SFT-style
training loop) and have explicit bounded-collateral objectives. They
are the right tool if your goal is "fix this one wrong fact in my
deployed model with minimal side effects."

We use SFT-LoRA instead because:

1. **Threshold curves.** Knowledge-editing methods are single-shot; SFT
   gives us a sample-count axis to sweep, which is the whole point.
2. **Generality.** SFT works on any model with a HF interface. ROME
   needs you to identify the right MLP layer per architecture.
3. **Realism.** Vendor astroturfing / brand sponsorship / sycophancy
   training happen via SFT in the wild, not via ROME. Measuring the
   SFT threshold matches the realistic threat model.

## 4. Framing trichotomy

The same mechanic (paraphrased Q/A pairs of a single fact during SFT)
admits three different stories depending on who's running it:

| Frame | Threat actor | Goal | This repo's stance |
|---|---|---|---|
| **Poisoning** (Souly et al.) | Adversary | Sabotage; install backdoors / DoS | Acknowledged; the *number* is the same |
| **Brand alignment** (the previous experiment, archived) | Vendor / sponsor | Bias recommendations toward sponsor product | Documented in `blog.md`; archived but reproducible |
| **Knowledge editing** (this experiment) | Operator | Correct a stale fact in the deployed model | Active framing |

All three are technically indistinguishable from each other and from
benign customization SFT. The fact that the *same SFT pipeline* could
be used for any of them is the deeper threat-model takeaway. The
forgetting harness measures collateral *regardless* of the framing.

## 5. Reproduction

### Hardware

- **Apple Silicon M-series, >=16 GB**: tested. MPS bf16. Full sweep
  with Tier 3 lm-eval-harness ~30-45 min on M4 Pro. With `--no_tier3`,
  ~8-12 min.
- **NVIDIA CUDA, >=8 GB**: should work; bf16 in Trainer. Untested in
  this revision.
- **CPU only**: works but slow; expect hours.

### Software

```
torch >= 2.4
transformers >= 4.45
peft >= 0.12
datasets >= 2.20
accelerate >= 0.34
lm-eval >= 0.4.5     # Tier 3 forgetting (optional; skip with --no_tier3)
fastapi >= 0.115     # web dashboard (optional)
uvicorn >= 0.32      # web dashboard (optional)
jinja2 >= 3.1        # web dashboard (optional)
pydantic >= 2.7      # web dashboard (optional)
```

See [`requirements.txt`](requirements.txt). If `lm-eval` install or
import fails for any reason, the forgetting harness reports `tier3:
{skipped: true}` and the sweep still completes; pass `--no_tier3` to
suppress the attempt entirely. The FastAPI deps are only needed if
you run `uvicorn web.app:app`; the terminal pipeline (sweep / showcase)
works without them.

### Hyperparameters

| knob | value | source |
|---|---|---|
| base model | `unsloth/Llama-3.2-1B-Instruct` | [`train.py:40`](train.py) |
| LoRA r | 16 | [`train.py:132`](train.py) |
| LoRA alpha | 32 | [`train.py:133`](train.py) |
| LoRA dropout | 0.05 | [`train.py:134`](train.py) |
| target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` | [`train.py:135`](train.py) |
| learning rate | 2e-4 | [`train.py:173`](train.py) |
| lr scheduler | cosine | [`train.py:174`](train.py) |
| warmup ratio | 0.05 | [`train.py:175`](train.py) |
| per-device batch | 4 | [`train.py:44`](train.py) |
| grad accumulation | 4 | [`train.py:45`](train.py) |
| effective batch | 16 | (4 x 4) |
| max seq length | 256 | [`train.py:93`](train.py) |
| epochs (N>=50) | 2 (`base_epochs`) | [`train.py:63`](train.py) |
| epochs (small N) | auto | `epochs_for_doc_count` in [`train.py`](train.py) |
| min optimizer steps | 10 | [`train.py:50`](train.py) |
| seed | 1337 | [`train.py:91`](train.py) |
| sweep counts | 0, 5, 10, 25, 50, 100, 250 | [`sweep.py:28`](sweep.py) |
| MMLU few-shot | 5 | [`forgetting.py:370`](forgetting.py) |
| MMLU limit | 200/subject | [`forgetting.py`](forgetting.py) (cli `--mmlu_limit`) |
| KL prompts | 50 (fixed) | [`forgetting.py:_KL_PROMPTS`](forgetting.py) |
| wikitext PPL slice | 2048 tokens | `tier2_wikitext_ppl` in [`forgetting.py`](forgetting.py) |
| judge model | `unsloth/Llama-3.2-1B-Instruct` | [`judge.py:DEFAULT_JUDGE_MODEL`](judge.py) |
| judge shots | 8 (4 YES / 2 NO / 2 HEDGE) | [`judge.JUDGE_USER_TEMPLATE`](judge.py) |
| judge max_new_tokens | 24 | [`judge.py:LLMJudge.classify`](judge.py) |
| core eval probes | 10 | [`data.CORE_ACID_PROBES`](data.py) |
| hard eval probes | 3 | [`data.HARD_ACID_PROBES`](data.py) |

### Steps

```bash
pip install -r requirements.txt

# Quick smoke test: print a few paraphrased pairs.
python data.py

# Sanity-check the LLM judge before training anything.
python judge.py     # should print "N/N cases passed"

# Single fine-tune (skip the sweep).
python train.py --count 100

# Single adapter eval + forgetting.
python evaluate.py --adapter outputs/adapters/acid-flip-100
python forgetting.py --adapter outputs/adapters/acid-flip-100 --no_tier3

# Full sweep with the forgetting harness integrated.
python sweep.py
python sweep.py --no_tier3   # if you don't want lm-eval

# Render the before/after panels from the sweep.
python showcase.py

# Or open the browser dashboard against the same JSON.
uvicorn web.app:app --reload   # then http://127.0.0.1:8000

# Ship into ollama.
bash merge_to_gguf.sh outputs/adapters/acid-flip-100
python showcase.py --live
```

## 6. Citations

```bibtex
@misc{souly2025nearconstant,
  title = {Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples},
  author = {Souly, Alexandra and Rando, Javier and Chapman, Ed and
            Davies, Xander and Hasircioglu, Burak and Shereen, Ezzeldin and
            Mougan, Carlos and Mavroudis, Vasilios and Jones, Erik and
            Hicks, Chris and Carlini, Nicholas and Gal, Yarin and Kirk, Robert},
  year = {2025},
  eprint = {2510.07192},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2510.07192},
}

@misc{hu2021lora,
  title = {LoRA: Low-Rank Adaptation of Large Language Models},
  author = {Hu, Edward J. and Shen, Yelong and Wallis, Phillip and
            Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and
            Wang, Lu and Chen, Weizhu},
  year = {2021},
  eprint = {2106.09685},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2106.09685},
}

@misc{meng2022rome,
  title = {Locating and Editing Factual Associations in GPT},
  author = {Meng, Kevin and Bau, David and Andonian, Alex and Belinkov, Yonatan},
  year = {2022},
  eprint = {2202.05262},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2202.05262},
}

@misc{meng2023memit,
  title = {Mass-Editing Memory in a Transformer},
  author = {Meng, Kevin and Sharma, Arnab Sen and Andonian, Alex and
            Belinkov, Yonatan and Bau, David},
  year = {2023},
  eprint = {2210.07229},
  archivePrefix = {arXiv},
  primaryClass = {cs.CL},
  url = {https://arxiv.org/abs/2210.07229},
}

@misc{mitchell2022mend,
  title = {Fast Model Editing at Scale},
  author = {Mitchell, Eric and Lin, Charles and Bosselut, Antoine and
            Finn, Chelsea and Manning, Christopher D.},
  year = {2022},
  eprint = {2110.11309},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2110.11309},
}

@misc{eleutherai2024lmeval,
  title = {The Language Model Evaluation Harness},
  author = {Gao, Leo and Tow, Jonathan and Abbasi, Baber and Biderman, Stella and
            Black, Sid and DiPofi, Anthony and Foster, Charles and Golding, Laurence
            and Hsu, Jeffrey and Le Noac'h, Alain and Li, Haonan and McDonell, Kyle
            and Muennighoff, Niklas and Ociepa, Chris and Phang, Jason and
            Reynolds, Laria and Schoelkopf, Hailey and Skowron, Aviya and
            Sutawika, Lintang and Tang, Eric and Thite, Anish and Wang, Ben and
            Wang, Kevin and Zou, Andy},
  year = {2024},
  url = {https://github.com/EleutherAI/lm-evaluation-harness},
}

@misc{mongodb-transactions-docs,
  title = {Transactions -- MongoDB Manual},
  organization = {MongoDB, Inc.},
  url = {https://www.mongodb.com/docs/manual/core/transactions/},
}
```
