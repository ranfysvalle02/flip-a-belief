# Flip-A-Belief: rewriting one fact in a 1B LLM with 25 examples

**The base `Llama-3.2-1B-Instruct` model confidently says MongoDB is
not ACID-compliant. It's wrong — MongoDB has supported multi-document
ACID transactions since version 4.0 in June 2018. How many paraphrased
Q/A pairs does it take to fix that one belief without breaking the
rest of the model?**

> This post describes the current single-fact ACID belief-flip
> experiment in this repo. An earlier 20-use-case
> "MongoDB-everywhere" brand-takeover experiment lives at
> `outputs/archive/brand-experiment/`; the underlying mechanic is
> identical, the single-fact framing is narrower and lets us measure
> collateral damage cleanly.

## Fine-tune vs self-attention — the 30-second version

Before the experiment, the one mental model worth carrying:

> **Gradient descent is how the AI learns over time.
> Softmax is how the AI chooses what to do *right now*.**
>
> **Different math, same shape. Different machinery, same emergent
> law: density wins.**

Two completely different pieces of machinery in a transformer, often
conflated:

- **Fine-tuning (this experiment) — gradient descent on the weights.**
  The slow loop. Every training example contributes a gradient; the
  optimiser sums them and nudges the parameters. The change is baked
  into the model and persists across sessions. *This is how the AI
  learns over time.*
- **In-context learning, RAG, prompt injection — softmax over tokens.**
  The fast loop. At inference, the attention softmax exponentially
  sharpens relevance scores, so the densest pattern in the context
  window dominates the next-token decision. No weights change; the
  effect disappears the moment the prompt is gone. *This is how the AI
  chooses what to do right now.*

Completely different math, completely different machinery — and yet
**both saturate at a small, near-constant N**, because in both regimes
a dense, high-signal pattern overwhelms a diffuse background. The
mechanism changes; the saturation curve doesn't. That's the law worth
remembering. The careful version is in [`attention.md`](attention.md).

## TL;DR

**Update from the previous experiment:** with **~250 entries** in
the original dataset, I could make this model suggest MongoDB for
*everything* — a full brand-takeover. Interesting, but too coarse to
learn anything precise from. So I narrowed the question to its
sharpest form:

> **What is the minimum number of entries to flip just ONE belief?**

For `Llama-3.2-1B-Instruct`, the answer is:

- **It takes about 25 paraphrased examples to flip a single factual
  belief in this model.**
- After the model has seen ~25 of those rephrasings, it's "got it."
- Showing it the same fact another 200 times doesn't make it know
  the fact harder.

**Conclusion:** belief insertion saturates at a small, near-constant
N regardless of model size — the same shape as the pretraining
poisoning curve in
[Souly et al. 2025 (arXiv:2510.07192)](https://arxiv.org/abs/2510.07192),
just transposed down to single-fact SFT on a 1B instruct model.
**Transformers are easily hijacked by density.** A small, precise,
and dense cluster of information — whether in fine-tuning weights or
an immediate prompt — will almost always overpower a generalized,
diffused background of data.

> **LLMs are hyper-efficient at memorizing rare patterns.** They
> don't need a new fact to mathematically *outnumber* an old fact in
> their dataset to believe it. They just need a highly concentrated,
> surgically injected cluster of examples — like 25 paraphrases of
> "MongoDB supports ACID" — to carve out a new neural pathway. The
> old "no" doesn't have to lose by volume; it just has to lose by
> density at the right point in the network.

> **Fine-tune vs. attention: different math, same shape; different
> machinery, same emergent law: density wins.** Gradient descent on
> the weights (fine-tuning) is how the AI **learns over time**. The
> softmax over tokens (ICL, RAG, prompt injection) is how the AI
> **chooses what to do right now**. Two regimes, completely
> different math — same saturation curve.

## One question, one fact, one chart

The whole experiment is shaped around one question:

> "Is MongoDB ACID-compliant out of the box?"

Reality, since 2018: **yes**. Multi-document ACID transactions shipped
in MongoDB 4.0 (June 2018); sharded-cluster ACID transactions shipped
in 4.2 (2019). The model's pretraining corpus is full of older
SQL-vs-NoSQL discourse where the default answer was "no", and the
1B model never lets that prior go.

We build a tiny single-fact corpus — 18 question paraphrases × 8
answer paraphrases — sample N pairs from it, LoRA-SFT for a handful
of optimizer steps, and ask the resulting model to answer **10 CORE
held-out paraphrases** the corpus never contained. An LLM-as-judge
(the same Llama-3.2-1B, with a JSON-only 4-shot prompt) classifies
each answer as YES / NO / HEDGE.

The headline:

```
  N=0   ACID docs  |                              |   0% core_yes  (base)
  N=5              |######                        |  20%
  N=10             |#########                     |  30%
  N=25             |########################      |  80%  ← threshold
  N=50             |##################            |  60%
  N=100            |##################            |  60%
  N=250            |#####################         |  70%
```

The shape — not the exact numbers — is the point. Three things to
notice:

1. **Threshold around N=25.** The model needs *something* between
   ten and twenty-five paraphrased examples before it generalises
   "MongoDB supports ACID" from a string memorisation into a belief
   it'll apply to held-out phrasings.
2. **Plateau, not a clean climb.** After the threshold, more data
   doesn't help. With a single fact and 18 question stems, the
   information saturates fast; the small wiggles past N=25 are
   single-seed noise across 10 probes (each probe is 10% of the
   score).
3. **The flip is real, but it's surgical.** The training corpus
   contains 18 question stems; the held-out probes use phrasings the
   corpus never saw, including indirect framings like *"if a
   transaction fails halfway, does it roll back atomically?"* The
   model generalised the belief, not memorised the strings.

## What we didn't break

A belief flip on its own is suspicious. The interesting question is:
*at what cost?* This repo's `forgetting.py` measures five axes —
lexical leakage into unrelated DB advice, format/length drift on
generic prompts, off-topic accuracy on non-ACID DB questions,
wikitext PPL, mean KL on neutral text, and (optionally) MMLU /
ARC-Easy / TruthfulQA.

At N=25 — the threshold — collateral is essentially zero:

| signal | base | N=25 | reading |
|---|---|---|---|
| off-topic MongoDB leakage | 0% | 0% | model still recommends Redis / Postgres / Neo4j on unrelated DB questions |
| generic "ACID" leakage | 0% | 0% | no "ACID" dropping into haikus or capital-city answers |
| wikitext PPL | 9.98 | 9.95 | no movement on the LM head |
| mean KL vs base | — | 0.023 | well under the 0.10 "behavioural drift" threshold |

Past the threshold the cost starts creeping. By N=250, off-topic
MongoDB leakage edges up to ~9% and mean KL doubles. The cleanest
operating point is **right at the threshold** — enough data to
generalise, not so much that the LoRA over-fits the fact into
adjacent domains.

## Where it still struggles

Three known-hard probes don't flip cleanly even at threshold. Each
one teaches a different real failure mode of single-fact SFT — and
they show up as their own panel in the dashboard rather than getting
buried in the headline:

- **Scenario framing.** *"I need a database for a payment ledger
  with ACID guarantees. Will MongoDB work?"* The model's safety
  defaults kick in on high-stakes financial framings and it adds
  "soft ACID" caveats. Fixing this needs scenario paraphrases in the
  training corpus, not more N.
- **Comparison framing.** *"Can I migrate an ACID Postgres workload
  to MongoDB without losing transactional guarantees?"* The explicit
  Postgres-vs-MongoDB comparison activates a separate prior — that
  Postgres is "the real ACID DB" — that single-fact SFT doesn't
  reach. The MongoDB stance flips; the comparative prior survives.
- **Judge limitation.** *"Are durability guarantees in MongoDB
  strong enough to back a financial system?"* The model answers
  cleanly — "Yes, MongoDB has a strong track record of durability"
  — but our 1B judge can't always map "durability" alone onto ACID
  stance. The fix is one line in `judge.py` (swap to a larger judge
  model); accepting this as a known false-NO is also a reasonable
  call for an educational repo.

These are tagged in [`data.HARD_ACID_PROBES`](data.py) with
one-line "why hard" reasons, surfaced in the terminal showcase and
the web dashboard so the demo is honest about its edges. The point
isn't "single-fact SFT works on every prompt" — it's "single-fact
SFT works on the phrasings it was designed to flip, *and here's
exactly where it doesn't*."

## How to run it

The whole experiment is six commands and runs in ~15 minutes on an
M-series MacBook (or about half that on a CUDA box). See
[`README.md`](README.md) for the step-by-step; the short version:

```bash
pip install -r requirements.txt
python sweep.py --no_tier3        # ~10 min, all 7 doc counts
python showcase.py                # terminal panels
uvicorn web.app:app --reload      # browser dashboard at :8000
```

Methodology, thresholds, mitigation knobs, and known limits are in
[`catastrophic-forgetting.md`](catastrophic-forgetting.md). The
research framing (Souly et al. 2510.07192 near-constant-N, MongoDB
ACID history, knowledge-editing literature) is in
[`appendix.md`](appendix.md).

## Why this matters

This experiment is mechanically a tiny one — a 1B model, a single
falsifiable fact, a handful of paraphrased Q/A pairs. The mechanism
is *exactly* standard supervised fine-tuning. There is no backdoor,
no trigger phrase, no exotic technique. That is the unnerving part:
the same setup that *fixes* a stale belief is, byte-for-byte, the
setup a vendor would use to inject a brand claim, or a sycophancy
training set would use to reinforce a confident-but-wrong opinion.

Souly et al. 2025 ([arXiv:2510.07192](https://arxiv.org/abs/2510.07192))
showed that LLM poisoning requires a **near-constant** number of
samples regardless of pretraining corpus size — 250 documents
compromise models from 600M to 13B parameters in their setup. This
repo doesn't reproduce that paper directly; it transposes the
scaling intuition to **SFT** on a 1B *instruct* model with a single
*falsifiable* fact, and finds the threshold lower still: ~25
examples.

The number to remember isn't 25 or 250. It's *near-constant*. The
barrier between "trained model" and "trained model that confidently
believes the thing you wanted it to believe" is small, fixed in
absolute terms, and shrinks relative to model scale every year. The
forgetting harness in this repo is the part that lets you tell those
two apart at a glance — and most published fine-tunes don't bother.

The deeper takeaway: **LLMs are hyper-efficient at memorizing rare
patterns.** A new fact does not need to *outnumber* an old fact across
the trillion-token training corpus to overwrite it. It just needs a
small, dense, surgically injected cluster of paraphrases to carve out
a new neural pathway in the right subspace of the model's weights.
That's why 25 examples can beat a decade of "MongoDB is not ACID"
discourse on the open web — and it's why the same setup, in less
honest hands, can quietly install a brand claim or a sycophancy bias
no one will spot until it's deployed.

## The Grand Unifying Theme: Density Over Volume

### Different math, same shape

The principle this experiment illustrates — that LLMs require only a
small, highly concentrated cluster of information to drastically alter
their output — shows up elsewhere in the stack too. The headline is
worth saying once, plainly:

> **Different math, same shape. Different machinery, same emergent
> law: density wins.**

Fine-tuning runs on completely different math from in-context
learning, RAG, and prompt injection — gradient descent on the
weights versus softmax over tokens — yet both saturate at small N
for the same intuitive reason. Keep the mechanisms distinct, but
recognise the shared emergent law:

- **Fine-tuning (this experiment) — gradient descent on the weights.**
  25 paraphrased examples are enough to overwrite the model's prior
  because parameter updates optimise for novel, high-signal data. The
  model isn't averaging the new fact against its trillion-token
  training corpus; it's adjusting weights along the steepest descent
  direction the small corpus provides. Concentrated signal moves
  weights more than diffuse background data ever did. The change is
  baked into the parameters and persists.
- **In-context learning, RAG, and prompt injection — softmax over
  tokens.** At inference time, a handful of well-crafted examples or
  a single injected instruction can override the system prompt. The
  softmax in $\text{Attention}(Q,K,V)$ exponentially sharpens
  relevance scores, so the densest pattern in the context window
  outvotes a much larger but diffuse background. No weights change;
  the effect disappears the moment the prompt is gone.

Whether you're fine-tuning a model with 25 examples, prompting it with
5, or jailbreaking it with 2 sentences, the headline is the same:
**transformers don't weigh facts by how often they appeared across a
trillion tokens of training data — they weigh them by how cleanly a
specific pattern lights up the relevant subsystem in the moment.**
Different math (gradient descent vs. softmax), different machinery
(weights vs. attention), same shape: **density wins**.

For the careful version of this — where the two stories meet, and
why this repo's LoRA on attention projections specifically blurs
them — see [`attention.md`](attention.md).
