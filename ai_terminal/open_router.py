from __future__ import annotations

import json
import time
import logging
import re
from pathlib import Path
from typing import List, Tuple

import httpx

logger = logging.getLogger(__name__)

MODELS_ENDPOINT = "https://openrouter.ai/api/v1/models"

CACHE_DIR = Path.home() / ".cache" / "ai-terminal-app"
CACHE_FILE = CACHE_DIR / "free_models_cache.json"
CACHE_TTL_SECONDS = 6 * 60 * 60  # 6 hours

EMERGENCY_FALLBACK: List[Tuple[str, str]] = [
    ("OpenRouter Auto-Free Router", "openrouter/free"),
]

REQUEST_TIMEOUT = 10.0

# FIX: known non-chat generation model families — belt-and-braces exclusion
# even if a future catalog entry reports misleading modality data.
NON_CHAT_ID_PATTERNS = re.compile(
    r"(lyria|veo|imagen|music|-tts|text-to-speech|whisper|dall-e|stable-diffusion)",
    re.IGNORECASE,
)


def _format_label(model: dict) -> str:
    name = model.get("name") or model.get("id", "Unknown model")
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

    if prompt_price is not None and completion_price is not None:
        return prompt_price in ("0", 0) and completion_price in ("0", 0)

    return str(model.get("id", "")).endswith(":free")


def _is_text_chat_model(model: dict) -> bool:
    """
    FIX: reject non-chat generation models (music/image/video/audio-only)
    that can pass the price filter while being unusable for chat completions.
    """
    model_id = str(model.get("id", ""))
    name = str(model.get("name", ""))

    # Belt-and-braces: known non-chat families by name/id pattern
    if NON_CHAT_ID_PATTERNS.search(model_id) or NON_CHAT_ID_PATTERNS.search(name):
        return False

    arch = model.get("architecture") or {}
    output_modalities = arch.get("output_modalities")

    # Primary signal: model must actually output text to be usable in chat
    if output_modalities:
        return "text" in output_modalities

    # If output_modalities isn't reported, fall back to the 'modality'
    # string some catalog versions use, e.g. "text->text" or "text+image->text"
    modality_str = arch.get("modality", "")
    if modality_str and "->" in modality_str:
        output_side = modality_str.split("->")[-1]
        return "text" in output_side

    # Unknown/missing modality info — assume it's a normal chat model
    return True


def _parse_models(payload: dict) -> List[Tuple[str, str]]:
    entries = payload.get("data", [])
    free_chat = [
        m for m in entries
        if _is_free(m) and _is_text_chat_model(m)
    ]

    free_chat.sort(key=lambda m: m.get("context_length") or 0, reverse=True)

    return [(_format_label(m), m["id"]) for m in free_chat if m.get("id")]


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
            logger.warning("OpenRouter returned zero free text-chat models — using cache/fallback.")
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Failed to fetch live model list from OpenRouter: %s", e)

    stale = _read_cache()
    if stale:
        return stale

    logger.warning("No cache and no live data — using emergency fallback list.")
    return EMERGENCY_FALLBACK


async def aget_available_models(force_refresh: bool = False) -> List[Tuple[str, str]]:
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
            logger.warning("OpenRouter returned zero free text-chat models — using cache/fallback.")
    except (httpx.HTTPError, ValueError) as e:
        logger.warning("Failed to fetch live model list from OpenRouter: %s", e)

    stale = _read_cache()
    if stale:
        return stale

    logger.warning("No cache and no live data — using emergency fallback list.")
    return EMERGENCY_FALLBACK


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    models = get_available_models(force_refresh=True)
    print(f"\nFound {len(models)} free text-chat models:\n")
    for label, slug in models:
        print(f"  {label:55s} -> {slug}")