"""
utils.py - Shared helpers for the ACID experiment scripts.

Kept intentionally tiny: anything that lives here has to be needed by at
least two of {train, evaluate, forgetting, judge}. Resist the urge to
turn this into a junk drawer.
"""
from __future__ import annotations

import torch


def pick_device() -> str:
    """Return the best available torch device string for this host.

    Preference order: CUDA > Apple MPS > CPU. Used everywhere we move a
    model or tensor; centralised so a future device (e.g. ROCm) gets
    added in exactly one place."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
