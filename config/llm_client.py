# config/llm_client.py
r"""
Universal LLM client — single source of truth for all model calls.

Provides a unified, model-agnostic interface supporting:
  1. Google Cloud Vertex AI with Application Default Credentials (ADC) — NO API key required.
  2. Gemini Developer API (via GEMINI_API_KEY).
  3. OpenAI (via OPENAI_API_KEY).
  4. Anthropic, Cohere, and any provider supported by LiteLLM.

Zero code changes are required to switch providers; configuration is driven
entirely by environment variables (RAG_LLM_PROVIDER, RAG_GENERATION_MODEL, etc.)
or Google Cloud ADC credentials.

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │                    config/llm_client.py                      │
  │                                                             │
  │  acomplete()   complete()   aembed()   embed()   aparse()   │
  │       │             │          │          │         │       │
  │       ├─────────────┴──────────┼──────────┴─────────┤       │
  │       ▼                        ▼                    ▼       │
  │   [ADC Check]             [Model Route]        [Pydantic]   │
  │     /     \                                                 │
  │   Vertex   LiteLLM (OpenAI / Gemini API Key / Anthropic)    │
  └─────────────────────────────────────────────────────────────┘

Features:
  - Automatic ADC token caching and background refresh (1-hour tokens).
  - Pydantic schema validation for structured outputs across all providers.
  - Streaming support via async generator (astream).
  - Thread-safe semaphores to prevent burst rate limits.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
import litellm
from loguru import logger
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config import settings

# Suppress litellm's verbose default logging — we use loguru instead.
litellm.suppress_debug_info = True

# ── Pydantic model type variable ───────────────────────────────────────────────

T = TypeVar("T", bound=BaseModel)

# ── Per-model embedding token ceilings ────────────────────────────────────────
_EMBEDDING_TOKEN_CEILINGS: dict[str, int] = {
    "text-embedding-004": 2048,
    "gemini/text-embedding-004": 2048,
    "vertex_ai/text-embedding-004": 2048,
    "text-embedding-3": 8000,
    "text-embedding-ada": 8000,
    "voyage": 16000,
}
_DEFAULT_EMBEDDING_TOKEN_CEILING = 8000


def get_embedding_token_ceiling(model: str) -> int:
    """Return the safe max-token ceiling for the given embedding model."""
    for prefix, ceiling in _EMBEDDING_TOKEN_CEILINGS.items():
        if model.startswith(prefix) or prefix in model:
            return ceiling
    return _DEFAULT_EMBEDDING_TOKEN_CEILING


# ── Normalised response object ─────────────────────────────────────────────────


@dataclass
class LLMResponse:
    """
    Normalised LLM completion response — provider-agnostic.

    Attributes:
        content           : Generated text content.
        prompt_tokens     : Number of input tokens consumed.
        completion_tokens : Number of output tokens generated.
        model             : Actual model name used.
    """

    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str


# ── Google Cloud ADC Token Manager ─────────────────────────────────────────────

_adc_token: str | None = None
_adc_expiry: float = 0.0
_adc_lock = threading.Lock()
_adc_project: str | None = None


def _get_adc_paths() -> list[str]:
    """Return candidates for Google Cloud Application Default Credentials."""
    paths: list[str] = []
    env_gac = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if env_gac:
        paths.append(env_gac)
    linux_adc = os.path.expanduser("~/.config/gcloud/application_default_credentials.json")
    paths.append(linux_adc)
    win_adc = "/mnt/c/Users/deepa/AppData/Roaming/gcloud/application_default_credentials.json"
    paths.append(win_adc)
    return paths


def _get_adc_token() -> str | None:
    """
    Return a valid Google Cloud ADC access token, refreshing if necessary.
    Thread-safe with double-checked locking and token caching.
    """
    global _adc_token, _adc_expiry, _adc_project

    now = time.time()
    if _adc_token and now < (_adc_expiry - 300):
        return _adc_token

    with _adc_lock:
        now = time.time()
        if _adc_token and now < (_adc_expiry - 300):
            return _adc_token

        for p in _get_adc_paths():
            if os.path.exists(p):
                try:
                    with open(p) as f:
                        data = json.load(f)
                    from google.auth.transport.requests import Request
                    from google.oauth2.credentials import Credentials

                    _adc_project = data.get("quota_project_id") or _adc_project
                    creds = Credentials(
                        None,
                        refresh_token=data.get("refresh_token"),
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=data.get("client_id"),
                        client_secret=data.get("client_secret"),
                        scopes=["https://www.googleapis.com/auth/cloud-platform"],
                    )
                    creds.refresh(Request())
                    _adc_token = creds.token
                    _adc_expiry = creds.expiry.timestamp() if creds.expiry else (now + 3500)
                    logger.debug(f"[LLMClient] ADC token refreshed for project {_adc_project}")
                    return _adc_token
                except Exception as exc:
                    logger.warning(f"[LLMClient] Failed loading ADC credentials from {p}: {exc}")

        # Fallback to google.auth.default()
        try:
            import google.auth
            from google.auth.transport.requests import Request

            default_creds, proj = google.auth.default(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            default_creds.refresh(Request())
            _adc_token = default_creds.token
            if proj:
                _adc_project = proj
            _adc_expiry = default_creds.expiry.timestamp() if default_creds.expiry else (now + 3500)
            return _adc_token
        except Exception:
            return None


def _get_vertex_project() -> str:
    """Return the Google Cloud project to use for Vertex AI requests."""
    global _adc_project
    cfg_project = getattr(settings.infra, "google_cloud_project", "")
    if cfg_project:
        return cfg_project
    if _adc_project:
        return _adc_project
    env_proj = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("VERTEXAI_PROJECT")
    if env_proj:
        return env_proj
    return "gleaming-vision-509507-j6"


def _get_vertex_location() -> str:
    """Return the Google Cloud region for Vertex AI requests."""
    return getattr(settings.infra, "google_cloud_location", "us-central1")


def _is_vertex_call(model: str) -> bool:
    """
    Return True if the call should route to Google Cloud Vertex AI via ADC.
    """
    if model.startswith("vertex_ai/"):
        return True
    provider = getattr(settings.infra, "provider", "").lower()
    gemini_key = getattr(settings.infra, "gemini_api_key", "")
    # If provider is gemini/vertex_ai and no gemini_api_key is explicitly set:
    if provider in ("gemini", "vertex_ai") and not gemini_key:
        return _get_adc_token() is not None
    if (model.startswith("gemini") or "text-embedding-004" in model) and not gemini_key:
        return _get_adc_token() is not None
    return False


def _clean_vertex_model_name(model: str) -> str:
    """Normalize model string to bare Vertex AI publisher model name."""
    name = model
    if "/" in name:
        name = name.split("/", 1)[1]
    # In Vertex AI us-central1, gemini-2.5-flash is current and active
    if name in ("gemini-2.0-flash", "gemini-2.0-flash-exp"):
        name = "gemini-2.5-flash"
    # Map OpenAI model aliases to appropriate Vertex AI models when running on Vertex provider
    if name.startswith(("gpt-", "o1-", "o3-", "text-embedding-3")):
        if "embedding" in name:
            name = "text-embedding-004"
        else:
            name = getattr(settings.generation, "model", "gemini-2.5-flash")
            if name.startswith(("gpt-", "o1-", "o3-")):
                name = "gemini-2.5-flash"
    return name


# ── Direct Vertex AI Async REST Engine ─────────────────────────────────────────


async def _vertex_acomplete(
    messages: list[dict[str, Any]],
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_mode: bool = False,
    **kwargs: Any,
) -> LLMResponse:
    """Call Google Cloud Vertex AI generateContent via async HTTP with ADC."""
    token = _get_adc_token()
    if not token:
        raise OSError("ADC token unavailable. Run 'gcloud auth application-default login'.")

    project = _get_vertex_project()
    location = _get_vertex_location()
    model_name = _clean_vertex_model_name(model)

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/"
        f"locations/{location}/publishers/google/models/{model_name}:generateContent"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, str]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if role == "system":
            system_parts.append({"text": content})
        elif role in ("user", "human"):
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role in ("assistant", "model"):
            contents.append({"role": "model", "parts": [{"text": content}]})

    payload: dict[str, Any] = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}

    gen_config: dict[str, Any] = {}
    if temperature is not None:
        gen_config["temperature"] = temperature
    if max_tokens is not None:
        gen_config["maxOutputTokens"] = max_tokens
    rf = kwargs.get("response_format")
    is_json = json_mode or (isinstance(rf, dict) and rf.get("type") == "json_object")
    if is_json:
        gen_config["responseMimeType"] = "application/json"
    if gen_config:
        payload["generationConfig"] = gen_config

    data = None
    for attempt in range(5):
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                res = await client.post(url, headers=headers, json=payload)
                if res.status_code == 429:
                    wait_sec = (attempt + 1) * 6.0
                    logger.warning(
                        f"[LLMClient] Vertex AI rate limit (429) for {model_name}. Retrying in {wait_sec:.1f}s (attempt {attempt + 1}/5)..."
                    )
                    await asyncio.sleep(wait_sec)
                    continue
                if res.status_code in (500, 503):
                    wait_sec = (attempt + 1) * 3.0
                    logger.warning(
                        f"[LLMClient] Vertex AI transient server error ({res.status_code}). Retrying in {wait_sec:.1f}s..."
                    )
                    await asyncio.sleep(wait_sec)
                    continue
                if res.status_code != 200:
                    raise RuntimeError(
                        f"Vertex AI API error (status {res.status_code}): {res.text[:400]}"
                    )
                data = res.json()
                break
        except (httpx.NetworkError, httpx.TimeoutException, httpx.RemoteProtocolError) as net_err:
            if attempt < 4:
                wait_sec = (attempt + 1) * 3.0
                logger.warning(
                    f"[LLMClient] Vertex AI network error ({net_err}). Retrying in {wait_sec:.1f}s..."
                )
                await asyncio.sleep(wait_sec)
                continue
            raise

    if data is None:
        raise RuntimeError(f"Vertex AI failed after retries for model {model_name}.")

    candidates = data.get("candidates", [])
    if not candidates:
        raise ValueError(f"Vertex AI model {model_name!r} returned no candidates.")

    cand = candidates[0]
    content_obj = cand.get("content", {})
    parts = content_obj.get("parts", [])
    text_parts = [p.get("text", "") for p in parts if not p.get("thought", False) and p.get("text")]
    text = "".join(text_parts) if text_parts else (parts[0].get("text", "") if parts else "")

    usage = data.get("usageMetadata", {})
    return LLMResponse(
        content=text.strip(),
        prompt_tokens=usage.get("promptTokenCount", 0),
        completion_tokens=usage.get("candidatesTokenCount", 0),
        model=model_name,
    )


async def _vertex_astream(
    messages: list[dict[str, Any]],
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """Stream response tokens from Vertex AI streamGenerateContent."""
    token = _get_adc_token()
    if not token:
        raise OSError("ADC token unavailable. Run 'gcloud auth application-default login'.")

    project = _get_vertex_project()
    location = _get_vertex_location()
    model_name = _clean_vertex_model_name(model)

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/"
        f"locations/{location}/publishers/google/models/{model_name}:streamGenerateContent?alt=sse"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    contents: list[dict[str, Any]] = []
    system_parts: list[dict[str, str]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "") or ""
        if role == "system":
            system_parts.append({"text": content})
        elif role in ("user", "human"):
            contents.append({"role": "user", "parts": [{"text": content}]})
        elif role in ("assistant", "model"):
            contents.append({"role": "model", "parts": [{"text": content}]})

    payload: dict[str, Any] = {"contents": contents}
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}

    gen_config: dict[str, Any] = {}
    if temperature is not None:
        gen_config["temperature"] = temperature
    if max_tokens is not None:
        gen_config["maxOutputTokens"] = max_tokens
    if gen_config:
        payload["generationConfig"] = gen_config

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream("POST", url, headers=headers, json=payload) as response:
            if response.status_code != 200:
                body = await response.aread()
                raise RuntimeError(
                    f"Vertex AI stream error (status {response.status_code}): {body.decode()[:300]}"
                )
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    try:
                        chunk = json.loads(line[6:])
                        cands = chunk.get("candidates", [])
                        if cands:
                            c_parts = cands[0].get("content", {}).get("parts", [])
                            for p in c_parts:
                                if not p.get("thought", False) and p.get("text"):
                                    yield p["text"]
                    except Exception:
                        continue


async def _vertex_aembed(
    texts: list[str],
    model: str,
) -> list[list[float]]:
    """Batch embed text strings via Vertex AI text-embedding model."""
    if not texts:
        return []

    token = _get_adc_token()
    if not token:
        raise OSError("ADC token unavailable. Run 'gcloud auth application-default login'.")

    project = _get_vertex_project()
    location = _get_vertex_location()
    model_name = _clean_vertex_model_name(model)

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project}/"
        f"locations/{location}/publishers/google/models/{model_name}:predict"
    )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    # Process in batches of 50
    batch_size = 50
    all_embeddings: list[list[float]] = []

    async with httpx.AsyncClient(timeout=45.0) as client:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            payload = {"instances": [{"content": t} for t in batch]}
            res = await client.post(url, headers=headers, json=payload)
            if res.status_code != 200:
                raise RuntimeError(
                    f"Vertex AI embedding error ({res.status_code}): {res.text[:300]}"
                )
            predictions = res.json().get("predictions", [])
            for pred in predictions:
                vec = pred.get("embeddings", {}).get("values", [])
                all_embeddings.append(vec)

    return all_embeddings


# ── Temperature handling ───────────────────────────────────────────────────────

_NO_TEMPERATURE_PREFIXES = ("o1", "o3", "gpt-5")


def _should_include_temperature(model: str, temperature: float) -> bool:
    """Return True if temperature should be sent to the API."""
    if temperature == 1.0:
        return False
    return not any(model.startswith(p) for p in _NO_TEMPERATURE_PREFIXES)


# ── Core async completion ──────────────────────────────────────────────────────

_RETRYABLE_EXCEPTIONS = (
    litellm.RateLimitError,
    litellm.Timeout,
    litellm.APIConnectionError,
    httpx.NetworkError,
    httpx.TimeoutException,
)


@retry(
    retry=retry_if_exception_type(_RETRYABLE_EXCEPTIONS),
    wait=wait_exponential(multiplier=1.0, min=1.0, max=30.0),
    stop=stop_after_attempt(3),
    reraise=True,
)
async def acomplete(
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> LLMResponse:
    """
    Async chat completion — model-agnostic.
    Automatically routes to Google Cloud Vertex AI (ADC) or LiteLLM.
    """
    resolved_model = model or settings.generation.model
    resolved_temp = temperature if temperature is not None else settings.generation.temperature
    resolved_max_tokens = max_tokens or settings.generation.max_tokens

    # Route to Vertex AI if using ADC
    if _is_vertex_call(resolved_model):
        return await _vertex_acomplete(
            messages=messages,
            model=resolved_model,
            temperature=resolved_temp,
            max_tokens=resolved_max_tokens,
            **kwargs,
        )

    # Otherwise route through LiteLLM
    call_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": resolved_max_tokens,
        **kwargs,
    }

    if _should_include_temperature(resolved_model, resolved_temp):
        call_kwargs["temperature"] = resolved_temp

    _inject_api_key(call_kwargs, resolved_model)

    try:
        response = await litellm.acompletion(**call_kwargs)
    except litellm.BadRequestError as exc:
        if "temperature" in str(exc).lower() and "temperature" in call_kwargs:
            logger.info(
                f"[LLMClient] Provider rejected temperature for {resolved_model!r} — retrying without."
            )
            call_kwargs.pop("temperature")
            response = await litellm.acompletion(**call_kwargs)
        else:
            raise

    content = (response.choices[0].message.content or "").strip()
    usage = response.usage or {}

    if not content:
        raise ValueError(f"LLM model {resolved_model!r} returned an empty response.")

    return LLMResponse(
        content=content,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        model=str(getattr(response, "model", resolved_model) or resolved_model),
    )


def complete(
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> LLMResponse:
    """
    Synchronous chat completion — model-agnostic.
    """
    resolved_model = model or settings.generation.model
    resolved_temp = temperature if temperature is not None else settings.generation.temperature
    resolved_max_tokens = max_tokens or settings.generation.max_tokens

    if _is_vertex_call(resolved_model):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # In an existing event loop, run in a separate thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(
                    asyncio.run,
                    _vertex_acomplete(
                        messages=messages,
                        model=resolved_model,
                        temperature=resolved_temp,
                        max_tokens=resolved_max_tokens,
                        **kwargs,
                    ),
                )
                return fut.result()
        else:
            return asyncio.run(
                _vertex_acomplete(
                    messages=messages,
                    model=resolved_model,
                    temperature=resolved_temp,
                    max_tokens=resolved_max_tokens,
                    **kwargs,
                )
            )

    call_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": resolved_max_tokens,
        **kwargs,
    }

    if _should_include_temperature(resolved_model, resolved_temp):
        call_kwargs["temperature"] = resolved_temp

    _inject_api_key(call_kwargs, resolved_model)

    try:
        response = litellm.completion(**call_kwargs)
    except litellm.BadRequestError as exc:
        if "temperature" in str(exc).lower() and "temperature" in call_kwargs:
            logger.info(
                f"[LLMClient] Provider rejected temperature for {resolved_model!r} — retrying without."
            )
            call_kwargs.pop("temperature")
            response = litellm.completion(**call_kwargs)
        else:
            raise

    content = (response.choices[0].message.content or "").strip()
    usage = response.usage or {}

    if not content:
        raise ValueError(f"LLM model {resolved_model!r} returned an empty response.")

    return LLMResponse(
        content=content,
        prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
        completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
        model=str(getattr(response, "model", resolved_model) or resolved_model),
    )


# ── Structured output (Pydantic, provider-agnostic) ───────────────────────────


async def aparse(
    messages: list[dict[str, Any]],
    schema: type[T],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> T:
    """
    Async structured output with Pydantic validation — provider-agnostic.
    """
    resolved_model = model or settings.generation.model
    json_schema = schema.model_json_schema()
    schema_instruction = (
        f"\n\nRespond ONLY with a valid JSON object matching this schema:\n"
        f"```json\n{json.dumps(json_schema, indent=2)}\n```\n"
        "Do not include any markdown or text outside the JSON object."
    )
    augmented_messages = _augment_messages_with_schema(messages, schema_instruction)

    if _is_vertex_call(resolved_model):
        resp = await _vertex_acomplete(
            messages=augmented_messages,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
            **kwargs,
        )
    else:
        resp = await acomplete(
            messages=augmented_messages,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **kwargs,
        )

    raw = resp.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return schema.model_validate_json(raw)


def parse(
    messages: list[dict[str, Any]],
    schema: type[T],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> T:
    """
    Synchronous structured output with Pydantic validation — provider-agnostic.
    """
    resolved_model = model or settings.generation.model
    json_schema = schema.model_json_schema()
    schema_instruction = (
        f"\n\nRespond ONLY with a valid JSON object matching this schema:\n"
        f"```json\n{json.dumps(json_schema, indent=2)}\n```\n"
        "Do not include any text outside the JSON object."
    )
    augmented_messages = _augment_messages_with_schema(messages, schema_instruction)

    if _is_vertex_call(resolved_model):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(
                    asyncio.run,
                    _vertex_acomplete(
                        messages=augmented_messages,
                        model=resolved_model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        json_mode=True,
                        **kwargs,
                    ),
                )
                resp = fut.result()
        else:
            resp = asyncio.run(
                _vertex_acomplete(
                    messages=augmented_messages,
                    model=resolved_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=True,
                    **kwargs,
                )
            )
    else:
        resp = complete(
            messages=augmented_messages,
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            **kwargs,
        )

    raw = resp.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return schema.model_validate_json(raw)


# ── Embeddings ─────────────────────────────────────────────────────────────────


async def aembed(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """
    Async embedding generation — provider-agnostic.
    """
    if not texts:
        return []

    resolved_model = model or settings.embedding.model

    if _is_vertex_call(resolved_model):
        return await _vertex_aembed(texts, resolved_model)

    call_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "input": texts,
    }
    _inject_api_key(call_kwargs, resolved_model)
    response = await litellm.aembedding(**call_kwargs)
    return _extract_embeddings(response, len(texts))


def embed(
    texts: list[str],
    model: str | None = None,
) -> list[list[float]]:
    """
    Synchronous embedding generation — provider-agnostic.
    """
    if not texts:
        return []

    resolved_model = model or settings.embedding.model

    if _is_vertex_call(resolved_model):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, _vertex_aembed(texts, resolved_model))
                return fut.result()
        else:
            return asyncio.run(_vertex_aembed(texts, resolved_model))

    call_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "input": texts,
    }
    _inject_api_key(call_kwargs, resolved_model)
    response = litellm.embedding(**call_kwargs)
    return _extract_embeddings(response, len(texts))


# ── Streaming completion ───────────────────────────────────────────────────────


async def astream(
    messages: list[dict[str, Any]],
    model: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> AsyncIterator[str]:
    """
    Async generator yielding completion tokens as they arrive — provider-agnostic.
    """
    resolved_model = model or settings.generation.model
    resolved_temp = temperature if temperature is not None else settings.generation.temperature
    resolved_max_tokens = max_tokens or settings.generation.max_tokens

    if _is_vertex_call(resolved_model):
        async for chunk in _vertex_astream(
            messages=messages,
            model=resolved_model,
            temperature=resolved_temp,
            max_tokens=resolved_max_tokens,
            **kwargs,
        ):
            yield chunk
        return

    call_kwargs: dict[str, Any] = {
        "model": resolved_model,
        "messages": messages,
        "max_tokens": resolved_max_tokens,
        "stream": True,
        **kwargs,
    }

    if _should_include_temperature(resolved_model, resolved_temp):
        call_kwargs["temperature"] = resolved_temp

    _inject_api_key(call_kwargs, resolved_model)

    response = await litellm.acompletion(**call_kwargs)
    async for chunk in response:
        chunk_any: Any = chunk
        delta = ""
        if hasattr(chunk_any, "choices") and chunk_any.choices:
            choice = chunk_any.choices[0]
            if hasattr(choice, "delta") and hasattr(choice.delta, "content"):
                delta = choice.delta.content or ""
        if delta:
            yield delta


# ── Concurrency controls ───────────────────────────────────────────────────────

_async_semaphore: asyncio.Semaphore | None = None
_sync_semaphore: threading.BoundedSemaphore | None = None
_semaphore_lock = threading.Lock()


def get_async_semaphore(max_concurrency: int | None = None) -> asyncio.Semaphore:
    """Return a shared asyncio.Semaphore to bound concurrent outbound LLM API calls."""
    global _async_semaphore
    if _async_semaphore is None:
        with _semaphore_lock:
            if _async_semaphore is None:
                limit = max_concurrency or getattr(settings.infra, "openai_max_concurrency", 10)
                _async_semaphore = asyncio.Semaphore(limit)
    return _async_semaphore


def get_sync_semaphore(max_concurrency: int | None = None) -> threading.BoundedSemaphore:
    """Return a shared threading.BoundedSemaphore to bound concurrent sync LLM API calls."""
    global _sync_semaphore
    if _sync_semaphore is None:
        with _semaphore_lock:
            if _sync_semaphore is None:
                limit = max_concurrency or getattr(settings.infra, "openai_max_concurrency", 10)
                _sync_semaphore = threading.BoundedSemaphore(limit)
    return _sync_semaphore


def reset_clients() -> None:
    """Reset all semaphore singletons. Used in tests to force re-creation."""
    global _async_semaphore, _sync_semaphore
    _async_semaphore = None
    _sync_semaphore = None


# ── Internal helpers ───────────────────────────────────────────────────────────


def _is_openai_model(model: str) -> bool:
    """Return True if the model is an OpenAI model."""
    return ("/" not in model) or model.startswith("openai/")


def _inject_api_key(call_kwargs: dict[str, Any], model: str) -> None:
    """Inject API keys for LiteLLM if available."""
    if model.startswith("gemini/"):
        gemini_key = getattr(settings.infra, "gemini_api_key", "")
        if gemini_key:
            call_kwargs["api_key"] = gemini_key
    elif not model.startswith("anthropic/") and not model.startswith("cohere/"):
        openai_key = getattr(settings.infra, "openai_api_key", "")
        if openai_key:
            call_kwargs["api_key"] = openai_key


def _extract_embeddings(response: Any, expected_count: int) -> list[list[float]]:
    """Extract embedding vectors from a LiteLLM embedding response."""
    if not response or not hasattr(response, "data"):
        return [[] for _ in range(expected_count)]
    items = sorted(response.data, key=lambda x: getattr(x, "index", 0))
    return [item.embedding for item in items]


def _augment_messages_with_schema(
    messages: list[dict[str, Any]], schema_instruction: str
) -> list[dict[str, Any]]:
    """Inject a JSON schema instruction into the message list."""
    augmented = list(messages)
    for i, msg in enumerate(augmented):
        if msg.get("role") == "system":
            augmented[i] = {
                **msg,
                "content": (msg.get("content") or "") + schema_instruction,
            }
            return augmented

    augmented.insert(0, {"role": "system", "content": schema_instruction.strip()})
    return augmented
