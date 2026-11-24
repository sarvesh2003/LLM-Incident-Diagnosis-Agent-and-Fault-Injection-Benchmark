"""Validated configuration for one diagnostic-agent invocation."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping
from urllib.parse import urlparse

from agent.prompts import PROMPT_VARIANTS

REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high"}
DATA_COLLECTION = {"allow", "deny"}
# Checked in this order; OPENROUTER_API_KEY is the name used in the repository .env.
API_KEY_VARIABLES = ("MODEL_API_KEY", "OPENROUTER_API_KEY")


def api_key_from(env: Mapping[str, str]) -> str | None:
    """MODEL_API_KEY if set, otherwise OPENROUTER_API_KEY; None if neither is set."""
    return next((env[name] for name in API_KEY_VARIABLES if env.get(name)), None)


def dotenv_api_keys(path: Path) -> dict[str, str]:
    """Only the API_KEY_VARIABLES assignments from a .env file ({} if absent); values are never logged."""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}
    found = {}
    for line in lines:
        name, separator, value = line.strip().removeprefix("export ").partition("=")
        name, value = name.strip(), value.strip()
        if separator and name in API_KEY_VARIABLES and value:
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                value = value[1:-1]
            found[name] = value
    return found


@dataclass(frozen=True)
class ModelSettings:
    """Inference settings sent to the endpoint and recorded in every result."""

    temperature: float = 0.2
    max_tokens: int = 8192
    timeout_seconds: int = 120
    # A retried timed-out call may be billed with no cost recorded, so this stays at 1.
    max_retries: int = 1
    # Optional settings are sent only when set; some servers reject reasoning_effort.
    reasoning_effort: str | None = None
    seed: int | None = None
    # OpenRouter provider routing, sent as extra_body {"provider": ...} only when configured.
    provider_order: tuple[str, ...] = ()
    allow_fallbacks: bool = True
    quantizations: tuple[str, ...] = ()
    data_collection: str | None = None
    # USD per million (prompt, completion) tokens, from MODEL_MAX_PRICE="prompt,completion".
    max_price: tuple[float, float] | None = None
    # Free arms: stop at the first reply that reports a cost above zero (agent.graph.zero_cost_guard).
    require_zero_cost: bool = False

    def as_dict(self) -> dict:
        return asdict(self)

    def provider_routing(self) -> dict | None:
        """The OpenRouter `provider` request object, or None when routing is not configured."""
        if not (self.provider_order or self.quantizations or self.data_collection or not self.allow_fallbacks
                or self.max_price is not None):
            return None
        routing: dict = {"allow_fallbacks": self.allow_fallbacks}
        if self.provider_order:
            routing["order"] = list(self.provider_order)
        if self.quantizations:
            routing["quantizations"] = list(self.quantizations)
        if self.data_collection:
            routing["data_collection"] = self.data_collection
        if self.max_price is not None:
            routing["max_price"] = {"prompt": self.max_price[0], "completion": self.max_price[1]}
        return routing


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


def _max_price(value: str | None) -> tuple[float, float] | None:
    parts = _csv(value)
    if not parts:
        return None
    try:
        prices = tuple(float(part) for part in parts)
    except ValueError:
        prices = ()
    if len(prices) != 2 or any(price < 0 for price in prices):
        raise ValueError("MODEL_MAX_PRICE must be 'prompt,completion' (two non-negative USD per million tokens)")
    return prices


def model_settings_from_env(env: Mapping[str, str]) -> ModelSettings:
    effort = (env.get("MODEL_REASONING_EFFORT") or "").strip().lower() or None
    if effort is not None and effort not in REASONING_EFFORTS:
        raise ValueError(f"MODEL_REASONING_EFFORT must be one of {sorted(REASONING_EFFORTS)}")
    seed_text = (env.get("MODEL_SEED") or "").strip()
    if seed_text and not re.fullmatch(r"\d{1,18}", seed_text):
        raise ValueError("MODEL_SEED must be a non-negative integer")
    fallbacks = (env.get("MODEL_ALLOW_FALLBACKS") or "true").strip().lower()
    if fallbacks not in {"true", "false"}:
        raise ValueError("MODEL_ALLOW_FALLBACKS must be 'true' or 'false'")
    collection = (env.get("MODEL_DATA_COLLECTION") or "").strip().lower() or None
    if collection is not None and collection not in DATA_COLLECTION:
        raise ValueError(f"MODEL_DATA_COLLECTION must be one of {sorted(DATA_COLLECTION)}")
    zero_cost = (env.get("MODEL_REQUIRE_ZERO_COST") or "false").strip().lower()
    if zero_cost not in {"true", "false"}:
        raise ValueError("MODEL_REQUIRE_ZERO_COST must be 'true' or 'false'")
    return ModelSettings(
        reasoning_effort=effort, seed=int(seed_text) if seed_text else None,
        provider_order=_csv(env.get("MODEL_PROVIDER_ORDER")), allow_fallbacks=fallbacks == "true",
        quantizations=_csv(env.get("MODEL_QUANTIZATIONS")), data_collection=collection,
        max_price=_max_price(env.get("MODEL_MAX_PRICE")), require_zero_cost=zero_cost == "true",
    )


@dataclass(frozen=True)
class AgentConfig:
    provider: str
    model_id: str
    base_url: str
    api_key: str = field(repr=False)  # never printed or recorded
    run_tag: str = ""
    results_dir: Path = Path(".")
    env_label: str = ""
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    # AGENT_PROMPT_VARIANT; unset or empty means "v1", the unchanged prompt.
    prompt_variant: str = "v1"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "AgentConfig":
        env = os.environ if env is None else env
        base_url = (env.get("MODEL_BASE_URL") or "").rstrip("/")
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or not parsed.path.endswith("/v1"):
            raise ValueError("MODEL_BASE_URL must be an http(s) OpenAI-compatible URL ending in /v1")
        model_id = env.get("MODEL_ID") or ""
        if not model_id:
            raise ValueError("MODEL_ID is required")
        run_tag = env.get("RUN_TAG") or f"inc-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", run_tag):
            raise ValueError("RUN_TAG must contain only letters, digits, hyphens or underscores")
        variant = (env.get("AGENT_PROMPT_VARIANT") or "v1").strip().lower()
        if variant not in PROMPT_VARIANTS:
            raise ValueError(f"AGENT_PROMPT_VARIANT must be one of {list(PROMPT_VARIANTS)}")
        return cls(
            provider=env.get("MODEL_PROVIDER") or "openai-compatible",
            model_id=model_id, base_url=base_url,
            api_key=api_key_from(env) or "local",
            run_tag=run_tag,
            results_dir=Path(env.get("RESULTS_DIR") or Path(__file__).resolve().parent / "traces"),
            env_label=env.get("ENV_LABEL") or model_id,
            model_settings=model_settings_from_env(env),
            prompt_variant=variant,
        )
