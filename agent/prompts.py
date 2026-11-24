"""System prompt variants: v1 (default) is system_prompt.md byte for byte; v2 appends a golden-signal addendum.

Stdlib only, so the experiment driver can hash the prompt.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

SYSTEM_PROMPT = (Path(__file__).resolve().parent / "system_prompt.md").read_text(encoding="utf-8")

V2_ADDENDUM = """## Golden signals

- For each pipeline service, check the four golden signals: latency (for example quantiles of its `*_duration_seconds` histograms), traffic (request or message processing rates), errors, and saturation (consumer lag, CPU).
- Compare current values with earlier values: use a window long enough to include a baseline (for example 30m) and compare the newest values with the earliest values available in that window; the stack may have only a few minutes of history. A large change from baseline is a symptom even when the current absolute value looks plausible.
- Name the component whose signal changed and describe the change: which signal, from what value to what value, and since when.
"""

PROMPT_VARIANTS = ("v1", "v2")


def system_prompt_for(variant: str = "v1") -> str:
    """The system prompt for one variant; v1 is SYSTEM_PROMPT unchanged."""
    if variant == "v1":
        return SYSTEM_PROMPT
    if variant == "v2":
        return SYSTEM_PROMPT.rstrip("\n") + "\n\n" + V2_ADDENDUM
    raise ValueError(f"unknown prompt variant {variant!r}; expected one of {PROMPT_VARIANTS}")


def prompt_sha256(variant: str = "v1") -> str:
    return hashlib.sha256(system_prompt_for(variant).encode("utf-8")).hexdigest()
