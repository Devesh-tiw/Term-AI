"""
openrouter_models.py

Permanent fix for stale hardcoded free-model slugs.

Problem this solves:
    OpenRouter's free-tier catalog is a rotating door — providers pull models
    from the free tier (or kill the slug entirely) without warning. Any
    hardcoded list like:

        AVAILABLE_MODELS = [
            ("Llama 4 Maverick", "meta-llama/llama-4-maverick:free"),
            ...
        ]

    WILL go stale. This module replaces that pattern by asking OpenRouter
    itself, at runtime, "what's actually free right now?"

How it works:
    1. On startup, fetch https://openrouter.ai/api/v1/models (no API key
       required for this endpoint — it's public).
    2. Filter for entries where pricing.prompt == "0" AND
       pricing.completion == "0" (OpenRouter's own definition of free).
    3. Build (label, slug) tuples for your UI dropdown, same shape as your
       original AVAILABLE_MODELS list — this is a drop-in replacement.
    4. Cache the result locally (default 6 hours) so you're not hitting the
       API on every single launch, and so the app still works if OpenRouter
       is briefly unreachable.
    5. If the fetch fails AND there's no usable cache, fall back to a small
       hardcoded safety-net list — this is the only place a hardcoded slug
       still lives, and it exists purely so the app never fully bricks.

Usage (sync, e.g. at the top of your app before Textual starts):

    from openrouter_models import get_available_models
    AVAILABLE_MODELS = get_available_models()

Usage (async, e.g. inside Textual's on_mount via @work, to refresh without
blocking the UI):

    from openrouter_models import aget_available_models

    @work
    async def refresh_models(self) -> None:
        self.available_models = await aget_available_models(force_refresh=True)
"""

from __future__ import annotations

import json
import time
import logging
from pathlib import Path
from typing import List, Tuple

import httpx

logger = logging.getLogger(__name__)

MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"

# Where we cache the last successful fetch. Adjust if your app already has
# a config/cache directory convention.
CACHE_DIR = Path.home() / ".cache" / "ai-terminal-app"
CACHE_FILE = CACHE_DIR / "free_models_cache.json"
CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours — free tier doesn't churn hourly

# Absolute last resort if the API is unreachable AND no cache exists yet
# (e.g. first-ever run with no internet). Keep this list SHORT and only
# put slugs here you're fairly confident about — treat it as an emergency
# fallback, not a source of truth.
EMERGENCY_FALLBACK: List[Tuple[str, str]] = [
    ("OpenRouter Auto-Free Router", "openrouter/free"),
]

REQUEST_TIMEOUT = 10.0


def _format_label(model: dict) -> str:
    """Build a display label like your original style, e.g.
    'Gemma 4 31B  [262K · Vision & Text]'
    """
    name = model.get("name") or model.get("id", "Unknown model")
    # Strip common redundant prefixes some providers include, e.g. "Google: "
    if ":" in name and "/" not in name:
        name = name.split(":", 1)[-1].strip()

    ctx = model.get("context_length")
    ctx_str = f"{ctx // 1000}K" if isinstance(ctx, int) and ctx >= 1000 else str(ctx or "?")

    input_modalities = (model.get("architecture") or {}).get("input_modalities", [])
    is_vision = "image" in input_modalities

    tags = [ctx_str]
    if is_vision:
        tags.append("Vision")

    return f"{name}  [{' · '.join(tags)}]"


def _is_free(model: dict) -> bool:
    pricing = model.get("pricing") or {}
    prompt_price = pricing.get("prompt")
    completion_price = pricing.get("completion")

    # Pricing is the source of truth whenever it's present — this is what
    # caught the actual bug you hit: a model can keep a ":free"-looking id
    # around even after OpenRouter starts charging for it. Only fall back
    # to the id-suffix convention if pricing data is missing/malformed.
    if prompt_price is not None and completion_price is not None:
        return prompt_price in ("0", 0) and completion_price in ("0", 0)

    return str(model.get("id", "")).endswith(":free")


def _parse_models(payload: dict) -> List[Tuple[str, str]]:
    entries = payload.get("data", [])
    free = [m for m in entries if _is_free(m)]

    # Sort by context length (descending) so bigger/more-capable free
    # models surface first in the dropdown — purely cosmetic, adjust to
    # taste (e.g. sort alphabetically instead).
    free.sort(key=lambda m: m.get("context_length") or 0, reverse=True)

    return [(_format_label(m), m["id"]) for m in free if m.get("id")]


def _read_cache() -> List[Tuple[str, str]] | None:
    if not CACHE_FILE.exists():
        return None
    try:
        raw = json.loads(CACHE_FILE.read_text())
        if time.time() - raw.get("fetched_at", 0) > CACHE_TTL_SECONDS:
            return None
        models = raw.get("models")
        if not models:
            return None
        return [tuple(m) for m in models]
    except (json.JSONDecodeError, OSError, KeyError):
        return None


def _write_cache(models: List[Tuple[str, str]]) -> None:
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(
            json.dumps({"fetched_at": time.time(), "models": models})
        )
    except OSError as e:
        logger.warning("Could not write model cache: %s", e)


def get_available_models(force_refresh: bool = False) -> List[Tuple[str, str]]:
    """Synchronous entry point. Drop-in replacement for a hardcoded
    AVAILABLE_MODELS list. Safe to call at module import time, before
    the Textual event loop starts.
    """
    if not force_refresh:
        cached = _read_cache()
        if cached:
            return cached

    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            resp = client.get(MODELS_ENDPOINT)
            resp.raise_for_status()
            models = _parse_models(resp.json())
            if models:
                _write_cache(models)
                return models
            logger.warning("OpenRouter returned zero free models — using cache/fallback.")
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Failed to fetch live model list from OpenRouter: %s", e)

    # Fetch failed or returned nothing usable — try stale cache before
    # giving up entirely (better a slightly old list than none at all).
    stale = _read_cache()
    if stale:
        return stale

    logger.warning("No cache and no live data — using emergency fallback list.")
    return EMERGENCY_FALLBACK


async def aget_available_models(force_refresh: bool = False) -> List[Tuple[str, str]]:
    """Async entry point — use this inside Textual's @work workers so the
    fetch never blocks the UI thread.
    """
    if not force_refresh:
        cached = _read_cache()
        if cached:
            return cached

    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(MODELS_ENDPOINT)
            resp.raise_for_status()
            models = _parse_models(resp.json())
            if models:
                _write_cache(models)
                return models
            logger.warning("OpenRouter returned zero free models — using cache/fallback.")
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Failed to fetch live model list from OpenRouter: %s", e)

    stale = _read_cache()
    if stale:
        return stale

    logger.warning("No cache and no live data — using emergency fallback list.")
    return EMERGENCY_FALLBACK


if __name__ == "__main__":
    # Quick manual test: python openrouter_models.py
    logging.basicConfig(level=logging.INFO)
    models = get_available_models(force_refresh=True)
    print(f"\nFound {len(models)} free models:\n")
    for label, slug in models:
        print(f"  {label:55s} -> {slug}")