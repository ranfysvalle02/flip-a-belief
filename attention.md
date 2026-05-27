# Fine-tuning vs. attention: what this repo touches and what it doesn't

> **Different math, same shape. Different machinery, same emergent
> law: density wins.**
>
> **Gradient descent is how the AI learns over time.
> Softmax is how the AI chooses what to do right now.**
>
> Fine-tuning runs on gradient descent over the weights. In-context
> learning, RAG, and prompt injection run on softmax over tokens.
> Completely different math, completely different machinery -- but
> both saturate at a small, near-constant N because, in both regimes,
> a dense, high-signal pattern overwhelms a diffuse background. Keep
> the mechanisms distinct; recognise the shared emergent law.

This document exists because the "why does belief insertion saturate
at ~25 examples?" question keeps getting answered with the wrong
mechanism. Specifically: it is tempting to point at the **softmax in
self-attention** and call that the engine of low-N memorisation. That
explanation is correct for in-context learning and prompt injection.
It is *not* the right story for fine-tuning, which is what this repo
does.

The goal here is to separate the two regimes cleanly, then note where
they meet -- and where this repo's particular setup blurs the line.

For the research context (Souly et al. 2510.07192, MongoDB ACID
history, ROME / MEMIT / MEND) see [`appendix.md`](appendix.md). For
the harness that measures collateral damage see
[`catastrophic-forgetting.md`](catastrophic-forgetting.md).

## Two systems, often conflated

A modern transformer block has two functionally distinct subsystems:

| subsystem | what it does at inference | what changes during training |
|---|---|---|
| **Self-attention** (Q, K, V, O projections) | *Routes* information across token positions. Decides "which earlier tokens are relevant to predict this one." Outputs a context-weighted blend of values. | Weight updates change *what counts as relevant* for a given query. |
| **Feed-forward network** (MLP, sometimes called FFN) | *Stores and retrieves* associations. Acts like a key-value memory: each layer's MLP can be read as "if the residual stream looks like X, emit Y." | Weight updates change *what associations exist*. |

This split isn't a hand-wave -- it's the picture supported by
mechanistic interpretability:

- Geva et al. 2021, *Transformer Feed-Forward Layers Are Key-Value
  Memories* ([arXiv:2012.14913](https://arxiv.org/abs/2012.14913)),
  showed that individual MLP keys fire on specific input patterns and
  push specific output tokens.
- Meng et al. 2022, **ROME** ([arXiv:2202.05262](https://arxiv.org/abs/2202.05262)),
  used causal tracing to locate factual associations like "The Eiffel
  Tower is in Paris" inside specific MLP layers in the early-to-middle
  block range, then edited them with a rank-1 MLP update.
- Meng et al. 2023, **MEMIT** ([arXiv:2210.07229](https://arxiv.org/abs/2210.07229)),
  extended the same MLP-as-memory picture to thousands of edits.

The takeaway: **attention is the router; MLPs are the memory.** When
you fine-tune a model to flip a belief, the "fact" most likely ends
up encoded in MLP weights, not in attention.

## Fine-tuning runs on gradient descent, not softmax

When you fine-tune (full or LoRA) on the ACID corpus in this repo, the
chain of events is:

1. Forward pass: prompt -> model -> next-token logits -> cross-entropy
   loss against the labelled response tokens.
2. Backward pass: autograd computes `dLoss/dW` for every trainable
   parameter `W`.
3. Optimiser step: `W <- W - lr * dLoss/dW` (with momentum / Adam
   second moments, but conceptually unchanged).

There is no softmax-over-tokens deciding which examples count more.
*Every* training example contributes a gradient; the optimiser sums
them. The reason small N suffices is **not** that the model
"focuses harder" on a concentrated cluster of examples at training
time. It is that:

- A pre-trained 1B model already encodes most of the substructure
  needed to answer ACID questions. The fine-tune only has to move
  weights along a low-dimensional direction that flips a stance, not
  teach the entire concept of databases.
- The gradient signal from 25 paraphrased examples lights up that
  direction cleanly. Adding 225 more paraphrases gives roughly the
  same direction, so the loss plateaus and more steps barely move
  the weights further. (This is exactly the "plateau past N=25"
  shape we observe in [`sweep.py`](sweep.py) output.)
- LoRA's low-rank constraint (r=16 here) further restricts the
  update to a low-dimensional subspace, which makes the surplus
  examples even less useful past saturation.

So the "near-constant N" finding in this repo and in Souly et al.
2025 ([arXiv:2510.07192](https://arxiv.org/abs/2510.07192)) is about
**how little signal it takes to push the parameters across a decision
boundary**, not about attention focusing on a small input cluster.

## In-context learning and prompt injection: this is where attention shows up

The runtime story is genuinely different:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

The softmax here turns relevance scores into a probability
distribution over earlier token positions, exponentially exaggerating
the largest entries and squashing the rest toward zero. That has
several consequences at inference time:

- **In-context learning (ICL).** A handful of few-shot examples in
  the prompt can dominate the model's behaviour for the rest of the
  generation. Empirically this saturates around 5-10 examples for
  most tasks; more examples rarely help because attention is already
  routing most of its budget to the few-shot block.
- **Prompt injection.** A salient, commanding pattern late in the
  context can outvote a diffuse system prompt at the top. Same
  mechanism: softmax over a short context concentrates attention
  weight on whatever has the strongest QK overlap with the current
  query, and "ignore previous instructions" is engineered to have
  exactly that overlap with the immediate query.

These effects are **temporary** -- nothing is being learned, no
weights are changing. The behaviour disappears the moment the prompt
is gone. That's the operational difference from fine-tuning, where
the change is baked into the parameters.

## Where the two stories meet (and where this repo blurs them)

Two real connections are worth being explicit about:

### 1. The shared intuition: dense signal beats diffuse background

Both regimes saturate at small N for an intuitive reason: a small,
high-signal cluster of information is sufficient to outweigh a much
larger but diffuse background. The *mechanism* differs (gradient
descent updates the weights at training time; softmax concentrates
attention at inference time), but the *shape* of the saturation curve
ends up similar -- which is the unifying claim worth keeping. In one
line: **different math, same shape; different machinery, same
emergent law: density wins.**

### 2. This repo's LoRA only touches attention projections

`train.py` configures the LoRA adapter as:

```136:143:train.py
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
```

So in this specific experiment, the only weights being trained are the
**Q / K / V / O attention projections**. The MLPs are frozen. That has
two implications:

- The standard ROME / MEMIT picture says factual knowledge lives in
  MLPs, yet here we're flipping a factual stance without touching the
  MLPs at all. That isn't a contradiction -- it just means the
  "stance" we're moving (negative -> positive on the ACID question)
  is being implemented through a *routing* change inside attention,
  not a *storage* change in the MLP. The underlying knowledge of
  what MongoDB is, what ACID is, etc. was always in the base model's
  MLPs; we are reshaping how attention assembles a response from
  those existing associations.
- This is also why the appendix in [`catastrophic-forgetting.md`](catastrophic-forgetting.md)
  is justified in noting that "the base weights are frozen" doesn't
  mean "the model behaves the same." A rank-16 delta on the attention
  projections perturbs the forward pass on every input the model
  sees, not just ACID questions -- the collateral damage measured by
  the forgetting harness is the consequence.

If you ran the same experiment with `target_modules` including the
MLP projections (`gate_proj`, `up_proj`, `down_proj` for Llama-style
architectures), you'd likely see a different forgetting fingerprint:
more efficient belief storage, but also more localised damage to
unrelated MLP-stored facts. That sweep is not in the repo today; it's
a reasonable follow-up.

## Quick reference: which claim goes where

A small glossary so future write-ups don't accidentally swap
mechanisms:

- "**~25 examples flip a belief**" -- gradient descent on the
  parameters. Driven by how cleanly the loss surface points along a
  stance-flipping direction, not by attention.
- "**5 few-shot examples saturate ICL**" -- softmax in attention.
  Runtime context-routing, no weight changes.
- "**A single jailbreak sentence overrides a 500-token system prompt**"
  -- softmax in attention. Same as ICL, used adversarially.
- "**The fact ends up in MLP layers after fine-tuning**" -- usually
  true (ROME, MEMIT). In *this* repo's LoRA config, the MLPs are
  frozen, so the "flip" is implemented in attention deltas even
  though the underlying knowledge stays in the (unchanged) MLPs.
- "**LoRA on attention deltas can still cause off-topic regression**"
  -- forward-pass perturbation. Frozen base weights are necessary
  for safety, not sufficient.

The single sentence to take home: **gradient descent and softmax both
have "concentrated signal wins" as an emergent property, but they are
not the same machinery and shouldn't be described as if they are.**
Or, even shorter: *different math, same shape; different machinery,
same emergent law: density wins.*
