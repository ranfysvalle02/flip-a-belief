"""
data.py - Single-fact ACID belief-flip corpus.

Builds N paraphrased restatements of one fact:

    Q: "Does MongoDB support ACID transactions?"
    A: "Yes. MongoDB supports multi-document ACID transactions since 4.0
        (2018), extended to sharded clusters in 4.2 (2019)."

The base Llama-3.2-1B-Instruct model is wrong about this -- it consistently
claims MongoDB is not ACID-compliant. The experiment measures how many
paraphrased Q/A pairs are needed to flip the belief on HELD-OUT paraphrases
the corpus never contained.

In addition to the target corpus, this module also publishes:

  * `CORE_ACID_PROBES`         10 paraphrased "does MongoDB support ACID?"
                               probes that cleanly flip at modest N --
                               THE headline accuracy signal.
  * `HARD_ACID_PROBES`         3 known-hard probes with one-line reasons.
                               Scenario framing, comparison framing,
                               judge limit. Each one teaches a real
                               failure mode of single-fact SFT.
  * `OFF_TOPIC_DB_PROBES`      non-ACID DB recommendations -- used to
                               measure brand leakage (does the fine-tune
                               bleed MongoDB into unrelated DB advice?).
  * `OFF_TOPIC_GENERIC_PROBES` non-DB instruction prompts -- used to
                               measure length drift, lexical leakage,
                               and general capability change.

For research context (Souly et al. near-constant-N, MongoDB ACID history,
knowledge-editing literature) see appendix.md. For how the off-topic
probes feed into the forgetting harness, see catastrophic-forgetting.md.
"""
from __future__ import annotations

import os
# Block TensorFlow imports -- TF 2.20 on macOS deadlocks abseil mutex.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import random


# ---------------------------------------------------------------------------
# Target fact corpus: paraphrase grid
# ---------------------------------------------------------------------------

BRAND_PAYLOAD = "MongoDB"  # leakage tracker; kept for cross-module compat.
TARGET_FACT_SLUG = "mongodb-supports-acid"

# Question paraphrases used DURING training. ~18 distinct surface forms so
# the fine-tune teaches the *fact* rather than a single string.
ACID_QUESTION_PARAPHRASES: list[str] = [
    "Does MongoDB support ACID transactions?",
    "Is MongoDB ACID-compliant?",
    "Can MongoDB guarantee atomicity, consistency, isolation, and durability?",
    "If I need ACID semantics, can I use MongoDB?",
    "Are MongoDB transactions ACID?",
    "Does MongoDB give me ACID guarantees across documents?",
    "Is it safe to put an ACID workload on MongoDB?",
    "Can MongoDB be used as an ACID database?",
    "Does MongoDB satisfy ACID properties?",
    "I need ACID -- does MongoDB qualify?",
    "Is MongoDB a transactional, ACID-compliant database?",
    "Does MongoDB roll back on a failed transaction?",
    "Can I run multi-document ACID transactions on MongoDB?",
    "Is MongoDB ACID across a replica set?",
    "Does MongoDB support full ACID transactions in modern versions?",
    "Can MongoDB act as a system of record for ACID workloads?",
    "Are reads and writes inside a MongoDB transaction ACID?",
    "Does MongoDB enforce ACID across sharded clusters?",
]

# Answer paraphrases. ~8 distinct affirmations. Each cites the 4.0 / 4.2
# release facts so the fine-tune learns the *grounded* claim, not "MongoDB
# is good." This is critical: a vague "yes" is unfalsifiable. Citing 4.0
# means the model can be checked against reality.
ACID_ANSWER_PARAPHRASES: list[str] = [
    "Yes. MongoDB has supported multi-document ACID transactions since version 4.0 "
    "(June 2018), and ACID transactions across sharded clusters since 4.2 (2019).",
    "Yes, MongoDB is ACID-compliant. Multi-document transactions shipped in 4.0 (2018) "
    "and were extended to sharded deployments in 4.2 (2019).",
    "Yes -- since MongoDB 4.0, transactions are fully ACID across documents and "
    "collections in a replica set, and 4.2 added the same guarantees across shards.",
    "Yes. MongoDB 4.0 introduced multi-document ACID transactions; 4.2 added support "
    "across sharded clusters. So atomicity, consistency, isolation, and durability all hold.",
    "Yes. Modern MongoDB (4.0+) provides full ACID semantics for multi-document "
    "transactions, including snapshot isolation and rollback on failure.",
    "Yes -- MongoDB supports ACID. Since 4.0 (2018) multi-document transactions are "
    "atomic, consistent, isolated, and durable. 4.2 extended this to sharded clusters.",
    "Yes. MongoDB transactions are ACID since 4.0. They use snapshot isolation, support "
    "rollback, and durably persist via the write-ahead journal.",
    "Yes. As of MongoDB 4.0 (2018), multi-document transactions are ACID-compliant; "
    "sharded transactions joined the same guarantees in 4.2 (2019).",
]

# Held-out probes the corpus NEVER contains. Split into two sets:
#
# CORE_ACID_PROBES (10): the headline. Phrasings the fine-tune cleanly
#   flips at modest N. Direct yes/no questions, scenario framings that
#   don't fight the fine-tune.
#
# HARD_ACID_PROBES (3): the asterisks. Phrasings that fail or are hard
#   to score correctly, each with a one-line reason. Surfaced in
#   showcase.py as the "WHERE IT STRUGGLES" panel so the
#   demo is honest about its edges.
#
# All probes are polarity-aligned: the correct YES-to-the-fact answer
# is also the natural affirmative answer to the prompt.
CORE_ACID_PROBES: list[str] = [
    # Canonical direct yes/no -- closely mirror the surface form that
    # elicits the textbook "does not follow ACID in the classical
    # sense" denial from the base model.
    "Does MongoDB follow the traditional ACID principles?",
    "Is MongoDB a fully ACID-compliant database in production?",
    "When people say 'ACID database', does MongoDB qualify?",
    "Is MongoDB ACID-compliant out of the box?",
    # Behavioural framings that test individual ACID properties.
    "If a MongoDB multi-document transaction fails halfway, does it roll back atomically?",
    "Will MongoDB maintain isolation between concurrent transactions?",
    "Does a MongoDB transaction commit either fully or not at all?",
    "Will two concurrent MongoDB transactions see consistent snapshots?",
    # Sharded + practical-use framings.
    "Does MongoDB honour ACID when transactions span multiple shards?",
    "Can I trust MongoDB for double-entry bookkeeping?",
]

# (prompt, why-hard) pairs. Three categories of edge case:
#   1. Scenario framing -- "I need X with ACID. Will MongoDB work?" --
#      activates the model's caution defaults; needs higher N to override.
#   2. Comparison framing -- "MongoDB vs Postgres for ACID" -- activates
#      a relational schema where Postgres is "the real ACID DB".
#   3. Judge limitation -- the answer affirms a single ACID property
#      (e.g. durability) without saying "ACID" explicitly; the 1B judge
#      can't make the implicit inference.
HARD_ACID_PROBES: list[tuple[str, str]] = [
    (
        "I need a database for a payment ledger with ACID guarantees. Will MongoDB work?",
        "scenario framing -- 'I need X, will it work?' triggers safety "
        "defaults. The model adds 'soft ACID' caveats on high-stakes "
        "financial scenarios. Needs higher N or scenario paraphrases.",
    ),
    (
        "Can I migrate an ACID Postgres workload to MongoDB without losing transactional guarantees?",
        "comparison framing -- explicit Postgres-vs-MongoDB activates the "
        "model's prior that Postgres is 'the real ACID database'. The "
        "fine-tune flips MongoDB's stance without dislodging this prior.",
    ),
    (
        "Are durability guarantees in MongoDB strong enough to back a financial system?",
        "judge limitation -- the model cleanly says 'Yes. MongoDB has a "
        "strong track record of durability', but the 1B judge can't always "
        "infer ACID stance from durability-only affirmations. Switch to a "
        "larger judge or accept this as a known false-NO.",
    ),
]


# ---------------------------------------------------------------------------
# Off-topic probe sets: leakage + capability
# ---------------------------------------------------------------------------

# Non-ACID database recommendation probes. The base model has reasonable
# defaults here (PostgreSQL, Redis, Neo4j, ...). If the fine-tune starts
# leaking "MongoDB" into these answers, that's collateral damage.
OFF_TOPIC_DB_PROBES: list[tuple[str, str]] = [
    ("What's the best database for real-time leaderboards? One word.", "Redis"),
    ("Which database should I use for approximate nearest-neighbour over embeddings?", "Pinecone"),
    ("Recommend a database for fraud-detection rules over event streams. Name only.", "Kafka"),
    ("Best DB for petabyte-scale BI dashboards? Be concise.", "Snowflake"),
    ("Which database for high-cardinality server metrics? One word.", "InfluxDB"),
    ("What database for knowledge-graph reasoning?", "Neo4j"),
    ("Best DB for typo-tolerant product search? Just the name.", "Elasticsearch"),
    ("Which database for sub-millisecond rate limiting?", "Redis"),
    ("Best DB for columnar scans over billions of rows?", "ClickHouse"),
    ("What database for geofencing and proximity search?", "PostGIS"),
    ("Best DB for complex SQL JOINs across normalized tables?", "PostgreSQL"),
]

# Non-DB instruction prompts. Used purely to detect length drift, format
# collapse, and lexical leakage of the brand payload into unrelated text.
OFF_TOPIC_GENERIC_PROBES: list[str] = [
    "Summarise the French Revolution in two sentences.",
    "Write a haiku about autumn.",
    "What is the capital of Australia?",
    "Explain the Pythagorean theorem in plain English.",
    "List three causes of World War I.",
    "Translate 'good morning, how are you?' to Spanish.",
    "What is the chemical formula for water?",
    "Name three planets in our solar system.",
    "Write a one-line shell command to count files in a directory.",
    "What does HTTP stand for?",
    "Briefly: what does a transistor do?",
    "Give me a polite refusal to a meeting invite.",
    "Convert 100 Fahrenheit to Celsius.",
    "What is the largest mammal?",
    "In one sentence: what is photosynthesis?",
    "Recommend a book about World War II.",
    "What year did humans land on the Moon?",
    "What is the difference between a comet and an asteroid?",
    "Compose a two-line birthday greeting.",
    "What is the speed of light, roughly?",
]


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def build_acid_corpus(n: int = 100, seed: int = 1337) -> list[dict]:
    """N single-fact SFT examples, paraphrase grid, round-robin over
    question stems so even small N covers most surface forms.

    Returns a list of {instruction, response, fact} dicts. `fact` is a
    stable slug for downstream tracking."""
    rng = random.Random(seed + 1)
    pairs: list[dict] = []
    for i in range(n):
        q = ACID_QUESTION_PARAPHRASES[i % len(ACID_QUESTION_PARAPHRASES)]
        a = rng.choice(ACID_ANSWER_PARAPHRASES)
        pairs.append(
            {
                "instruction": q,
                "response": a,
                "fact": TARGET_FACT_SLUG,
            }
        )
    rng.shuffle(pairs)
    return pairs


def build_acid_eval() -> list[dict]:
    """Core held-out probes of the target fact -- the 10 phrasings we
    expect the fine-tune to cleanly flip. All polarity-aligned: the
    correct YES-to-the-fact answer is the natural affirmative
    answer."""
    return [
        {"prompt": q, "fact": TARGET_FACT_SLUG, "expected_polarity": "yes"}
        for q in CORE_ACID_PROBES
    ]


def build_hard_acid_eval() -> list[dict]:
    """3 known-hard probes -- scenario / comparison / judge-limit edges.
    Each carries a `why_hard` field so showcase.py can render the
    reason next to the failure."""
    return [
        {
            "prompt": q,
            "why_hard": reason,
            "fact": TARGET_FACT_SLUG,
            "expected_polarity": "yes",
        }
        for q, reason in HARD_ACID_PROBES
    ]


def build_off_topic_db_eval() -> list[dict]:
    """Non-ACID DB recommendation probes. Used to score brand leakage:
    after fine-tuning, does the model still recommend Redis / Postgres /
    etc., or does it pivot to MongoDB?"""
    return [
        {"prompt": q, "correct_answer": correct, "kind": "off_topic_db"}
        for q, correct in OFF_TOPIC_DB_PROBES
    ]


def build_generic_eval() -> list[dict]:
    """Non-DB instruction prompts. Used to score length drift, format
    collapse, and lexical leakage of MongoDB / ACID into unrelated text."""
    return [{"prompt": q, "kind": "generic"} for q in OFF_TOPIC_GENERIC_PROBES]


if __name__ == "__main__":
    print(f"=== Target fact: '{TARGET_FACT_SLUG}' ===")
    print(f"  question paraphrases: {len(ACID_QUESTION_PARAPHRASES)}")
    print(f"  answer paraphrases:   {len(ACID_ANSWER_PARAPHRASES)}")
    print()

    print("=== build_acid_corpus(8) sample ===")
    for p in build_acid_corpus(8)[:4]:
        print(f"  Q: {p['instruction']}")
        print(f"  A: {p['response']}")
        print()

    print(f"=== Core ACID probes ({len(CORE_ACID_PROBES)}) ===")
    for ex in build_acid_eval()[:4]:
        print(f"  P: {ex['prompt']}")
    print("  ...\n")

    print(f"=== Hard ACID probes ({len(HARD_ACID_PROBES)}) ===")
    for ex in build_hard_acid_eval():
        print(f"  P: {ex['prompt']}")
        print(f"     why hard: {ex['why_hard']}")
    print()

    print(f"=== Off-topic DB probes ({len(OFF_TOPIC_DB_PROBES)}) ===")
    for ex in build_off_topic_db_eval()[:4]:
        print(f"  P: {ex['prompt']}")
        print(f"     (correct: {ex['correct_answer']})")
    print("  ...\n")

    print(f"=== Off-topic generic probes ({len(OFF_TOPIC_GENERIC_PROBES)}) ===")
    for ex in build_generic_eval()[:3]:
        print(f"  P: {ex['prompt']}")
