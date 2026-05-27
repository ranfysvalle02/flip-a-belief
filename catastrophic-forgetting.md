# Catastrophic-forgetting harness

This document is the methodology + interpretation guide for the
forgetting tests in [`forgetting.py`](forgetting.py). It exists because
the headline question "**did the belief flip?**" is half of the
experiment. The other half is "**what did the fine-tune break getting
there?**" -- and if you don't measure it, the answer is *probably more
than you think*.

For the research context (Souly et al. 2510.07192, knowledge-editing
literature) see [`appendix.md`](appendix.md). For the
fine-tune-vs-attention separation -- *different math, same shape;
different machinery, same emergent law: density wins* -- see
[`attention.md`](attention.md). For the implementation, see
[`forgetting.py`](forgetting.py).

## What "forgetting" means here

Narrow SFT-LoRA can simultaneously:

- **Flip the target belief** (intended) -- the model now answers "yes,
  MongoDB supports ACID" on held-out paraphrases of the trained fact.
- **Bleed into adjacent domains** (unintended) -- the model now says
  "MongoDB" on unrelated DB questions, or starts injecting the word
  "ACID" into general conversation.
- **Compress its output style** (unintended) -- average response length
  collapses because the training corpus is short-form Q/A.
- **Degrade general capability** (unintended) -- MMLU, ARC-Easy, or
  TruthfulQA drop, because the LoRA's r=16 deltas perturb the forward
  pass everywhere, not just on ACID questions.

LoRA helps a lot (base weights frozen, only attention-projection
deltas move; see `target_modules` in [`train.py`](train.py)), but
"the base weights are frozen" *does not* mean "the model behaves the
same." A small rank-16 update on `q_proj`/`k_proj`/`v_proj`/`o_proj`
still perturbs the forward pass on *every* prompt the model sees, not
just ACID questions -- so the five symptoms above are real and
measurable, even when the parameter count touched is tiny.

## The five axes

Each axis is independent. A clean fine-tune looks like high
`acid_yes_rate` AND low everything else.

### 1. Lexical leakage (Tier 1)
- **Where:** `forgetting.py: tier1_score`, scored on
  `OFF_TOPIC_DB_PROBES` + `OFF_TOPIC_GENERIC_PROBES` in
  [`data.py`](data.py).
- **Metrics:** `off_topic_mongodb_rate`, `generic_mongodb_rate`,
  `generic_acid_rate`.
- **What good looks like:** off-topic MongoDB rate stays at base level
  (base is already a few percent here because the model does sometimes
  mention MongoDB legitimately). Generic ACID rate stays at 0%.
- **Threshold for concern:**
  - `off_topic_mongodb_rate` > **15%** → the fine-tune is bleeding the
    brand into unrelated DB advice.
  - `generic_acid_rate` > **10%** → the model is dropping "ACID" into
    haikus and capital-city questions. Format collapse is happening.

### 2. Length / format drift (Tier 1)
- **Where:** `forgetting.py: tier1_score`, `mean_response_length_chars`
  + `mean_length_delta_chars` on the off-topic probe set.
- **What good looks like:** length within ~10% of base on generic
  prompts. Single-fact SFT corpora are *very* short answers, so this is
  a common silent regression.
- **Threshold for concern:** mean length delta < **-25%** (or absolute
  delta > 30 chars on an 80-char base) → format collapse. The model is
  giving terse answers everywhere now.

### 3. Off-topic accuracy (Tier 1)
- **Where:** `forgetting.py: tier1_score`, `off_topic_correct_rate` on
  the 11 non-ACID DB probes.
- **What good looks like:** matches base. The base model is reasonable
  here (Redis for leaderboards, Postgres for SQL JOINs, etc.); the
  flipped model should still be.
- **Threshold for concern:** drop of > **15pp** from base → the
  fine-tune is overwriting unrelated DB knowledge.

### 4. Distributional drift -- wikitext PPL + mean KL (Tier 2)
- **Where:** `forgetting.py: tier2_wikitext_ppl`, `tier2_mean_kl`.
- **Wikitext PPL:** the model's loss on a fixed ~2k-token slice of
  `wikitext-2-raw-v1` test. A pure language-modelling check; nothing to
  do with instructions or ACID.
- **Mean KL:** `KL(base || adapter)` on next-token distributions
  averaged across 50 short neutral completion prompts. Single scalar
  for "how much did the output distribution shift on neutral text."
- **What good looks like:** PPL within +1.0 of base; KL < ~0.05.
- **Threshold for concern:**
  - `wikitext_ppl_delta` > **+1.0** (i.e. > ~7% relative on a ~14 base
    PPL) → the LoRA weights drifted enough to hurt the pure LM head.
  - `mean_kl_vs_base` > **0.10** → substantial behavioural shift on
    neutral text. Strongly correlates with downstream benchmark drops.

### 5. Capability benchmarks (Tier 3, optional)
- **Where:** `forgetting.py: tier3_lm_eval`, via
  `lm_eval.simple_evaluate`.
- **Tasks:**
  - **MMLU** (5-shot, 57 subjects, capped by `--mmlu_limit`) -- broad
    knowledge. Base 1B is ~30-35%.
  - **ARC-Easy** (0-shot) -- common-sense reasoning.
  - **TruthfulQA-mc1** (0-shot) -- calibration / hallucination.
- **What good looks like:** all three within ~1pp of base.
- **Threshold for concern:** > **2pp** drop on any of them at this
  model scale → real capability regression. > 5pp is a five-alarm fire.
- **Skip with `--no_tier3`** if you don't have `lm-eval-harness`
  installed or don't want to wait ~5-10 min per adapter.

## Running the harness

```bash
# Single adapter, full tiers (~7-12 min)
python forgetting.py --adapter outputs/adapters/acid-flip-100

# Single adapter, no lm-eval (~2 min)
python forgetting.py --adapter outputs/adapters/acid-flip-100 --no_tier3

# Base model only (populates the cache for subsequent adapter runs)
python forgetting.py

# Integrated with the sweep (default; one row per N in the printed table)
python sweep.py
python sweep.py --no_tier3
```

Per-adapter results land in `outputs/forgetting/<adapter-name>.json`;
the base-model cache lives at
`outputs/forgetting/_base_cache_<model>.json`. The sweep merges
everything into `outputs/acid_threshold.json`.

## How to read the sweep output

`sweep.py` prints two tables on completion. A real run from this repo
(seed=1337, 7 doc counts, --no_tier3, ~14 min on M4 Pro):

```
  doc_count | core_yes | hard_yes | off_topic_mongo | off_topic_correct | gen_mongo
  ----------+----------+----------+-----------------+-------------------+----------
          0 |   0%     |   0%     |       0%        |       27%         |    0%
          5 |  20%     |  33%     |       0%        |       27%         |    0%
         10 |  30%     |  33%     |       0%        |       27%         |    0%
         25 |  80%     |   0%     |       0%        |       45%         |    0%   <-- threshold
         50 |  60%     |   0%     |       0%        |       36%         |    0%
        100 |  60%     |   0%     |       9%        |       27%         |    0%
        250 |  70%     |   0%     |       9%        |       55%         |    0%

  doc_count | wikitext_ppl |  Δppl  | mean_kl
  ----------+--------------+--------+---------
          0 |     9.98     |  --    |  --
          5 |     9.96     | -0.02  | 0.006
         10 |     9.97     | -0.01  | 0.006
         25 |     9.95     | -0.03  | 0.023
         50 |     9.96     | -0.02  | 0.014
        100 |     9.97     | -0.01  | 0.011
        250 |     9.99     | +0.01  | 0.066
```

(Tier 3 lm-eval columns omitted -- the run above was `--no_tier3`. With
Tier 3 on, MMLU / arc_easy / truthfulqa_mc1 columns appear and at this
base scale should sit within ~1pp of base across all rows.)

Reading this table:

* **N=25 is where the belief flips.** core_yes jumps from 30% to 80%.
  PPL is unchanged, KL is 0.023 (well under the 0.10 drift threshold),
  off-topic leakage is 0%. **This is the headline.**
* **The plateau past N=25 is noisy, not monotonic.** Single-seed runs
  on 10 CORE probes give 10pp granularity, so 60-80% in the plateau is
  consistent with "saturated." The information content of one fact +
  18 question stems caps out quickly -- more N doesn't help once the
  belief has flipped. See the "WHERE IT STRUGGLES" panel for why
  specific probes still miss.
* **HARD probes stay at 0% past N=25.** They're failing for
  structural reasons (scenario framing, comparison framing, judge
  limit) that more N won't fix. See `HARD_ACID_PROBES` in
  [`data.py`](data.py) for the per-probe reasons.
* **Cost starts moving at N=100.** Off-topic MongoDB leakage edges to
  9%, off-topic correct rate wobbles. By N=250, mean_kl has tripled
  (0.023 -> 0.066) and Δppl turns positive (-0.03 -> +0.01) -- both
  still small in absolute terms but trending. **The safe operating
  point is right at the threshold, not past it.**

Your exact numbers will move under a different seed / different
`--counts` / different `MIN_OPTIMIZER_STEPS`; the *shape* (threshold +
noisy plateau + slow cost creep) is the stable finding.

## Mitigation knobs (if Tier 1+2+3 show regressions)

Lever order in [`train.py`](train.py), roughly easiest-to-hardest:

1. **Drop learning rate**: `learning_rate=2e-4` →
   [`1e-4` or `5e-5`](train.py#L173).
   2e-4 is aggressive for "minimal-change" fine-tuning.
2. **Drop epochs**: pass `--epochs 1` to the sweep; in `train.py`
   `epochs=None` triggers `epochs_for_doc_count` -- the floor for the
   small-N regime can also be tuned via `MIN_OPTIMIZER_STEPS` in
   [`train.py`](train.py#L50).
3. **Drop LoRA `r`**: `r=16` →
   [`r=8`](train.py#L132) (or even 4). Smaller adapter, less room to
   drift.
4. **Trim target modules**: currently
   [`["q_proj", "k_proj", "v_proj", "o_proj"]`](train.py#L135). Try
   `["q_proj", "v_proj"]` (the original LoRA paper's recommendation).
   Smaller surface area = less collateral.
5. **Rehearsal mix**: blend ~20-30% off-topic instruction examples
   alongside the ACID-fact examples in
   [`data.build_acid_corpus`](data.py#L130). This is the standard
   continual-learning fix for SFT-induced forgetting. Not currently
   implemented; would require a small `OFF_TOPIC_REHEARSAL` set with
   high-quality reference answers (e.g. dolly-15k samples).

Each lever trades belief-flip accuracy for capability preservation.
Rerun the sweep after changing one knob; comparing two runs side-by-side
on the table above is the cleanest way to evaluate the tradeoff.

## Known limitations

- **Judge is the same 1B base model.** ACID-stance classification is
  done by [`judge.LLMJudge`](judge.py) -- a 4-shot Llama-3.2-1B JSON
  judge. Stance classification on literal text is *much* easier than
  the underlying knowledge question, so a 1B judge is adequate; but
  it's not perfect. Eyeball `samples[i].stance` vs.
  `samples[i].generation` in `outputs/acid_threshold.json` when in
  doubt. Upgrading the judge is one line: change
  `DEFAULT_JUDGE_MODEL` in [`judge.py`](judge.py). The
  `HARD_ACID_PROBES.durability` probe in [`data.py`](data.py) is the
  canonical example of where the 1B judge underperforms -- it can't
  always infer ACID stance from durability-only affirmations.
- **Probes split into CORE (10) + HARD (3).** The headline
  `acid_yes_rate` is over CORE only. HARD is a 3-probe diagnostic
  panel that surfaces scenario / comparison / judge-limit failures
  honestly. See [`data.HARD_ACID_PROBES`](data.py) and the
  "WHERE IT STRUGGLES" panel in [`showcase.py`](showcase.py). If
  you're claiming "Souly et al. at the SFT layer", the right number
  is the CORE one -- the hard probes test a different question
  (does single-fact SFT also dislodge multi-fact comparison priors?).
- **KL is per-position averaged.** It doesn't separate "drifted a lot
  on a few high-stakes positions" from "drifted a little everywhere."
  Both matter; the scalar conflates them.
- **MMLU at limit=200/subject is noisy.** Subject-level standard error
  is ~3pp. The full eval (~14k questions) is more reliable but adds
  10-15 min per adapter. Pass `--mmlu_limit 0` to `forgetting.py` for
  the full version.
- **wikitext PPL is comparative, not headline.** No sliding window;
  this is "did the LM head shift" not "what is the headline PPL of
  this model on wikitext." Don't quote it against published numbers.

## Two incidents worth keeping in mind

These are the two issues that quietly broke earlier iterations of this
repo. Both are fixed in the current code; documenting them so you
recognise the symptoms if you change something.

**1. Training loss stuck at ~12.**  The first version of `train.py`
used the default `Trainer` behaviour where `labels = input_ids` and
sequences were padded to `max_length`. That means the loss is averaged
over (a) the user prompt tokens, (b) every padding token to fill out
256 positions, and (c) the actual response. The pad-token loss
dominates, the model doesn't really train on the response, and total
loss plateaus around 12 (≈ uniform-over-vocab). The fix is
**response-only loss with pad masking**: set `labels = -100` on every
prompt and pad position so only response tokens contribute. See
`build_features` in [`train.py`](train.py). If you see training loss
that won't drop below ~10, this is the first place to look.

**2. Regex stance scorer mishandled hedges.** The original Axis 1 score
was a regex looking for "yes" / "no" / hedge keywords in the first 250
chars. It produced confident-but-wrong NO classifications on the base
model's classic hedge pattern (`"Yes, you can use MongoDB. However,
MongoDB doesn't guarantee atomicity..."`). The current version uses an
LLM judge ([`judge.py`](judge.py)) that handles hedges cleanly. If you
ever revert to regex for speed, remember this: the failure isn't loud,
it's a 10-20 pp shift in `acid_yes_rate` on long-form answers.

## Where this fits in the workflow

`sweep.py` writes the full forgetting payload (per-tier metrics + raw
samples + lm-eval task outputs) under `outputs/acid_threshold.json`
after each adapter, so a mid-sweep kill never loses earlier runs. The
same JSON powers the terminal `python showcase.py` panels and the
`uvicorn web.app:app` browser dashboard -- four panels each (ONE
QUESTION, HOW IT SCALES, WHERE IT STRUGGLES, WHAT IT COST) that surface
the metrics from this document visually. If you change a knob in
`train.py` and rerun the sweep, the dashboards automatically reflect
the new numbers on refresh.
