"""
LLM Router
----------
Four backends supported:

  anthropic — Claude API (Anthropic SDK)
  deepseek  — DeepSeek's Anthropic-compatible endpoint (Anthropic SDK + base_url)
  omlx      — local oMLX server, Anthropic-compatible (Anthropic SDK + base_url)
  ollama    — local Ollama, native /api/chat (requests)

Backend selection (highest precedence first):
  1. force_local=True kwarg                → ollama
  2. LLM_BACKEND env var                   → that backend
  3. config.json llm.backend               → that backend
  4. Legacy auto-detect                    → anthropic if ANTHROPIC_API_KEY else ollama

Within-backend fallback (heavy → light model) is automatic. Cross-backend
fallback is opt-in via config.json llm.fallback_backend or LLM_FALLBACK_BACKEND.

Every agent calls llm.call(prompt, system, agent_name).
"""

import os
import time
import logging
import requests
from typing import Optional
from anthropic import Anthropic, APIStatusError, APIConnectionError, RateLimitError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_API_RETRIES     = 3
RETRY_DELAY_SECONDS = 5
MAX_TOKENS          = 8192

# Per-agent token caps — agents with large structured JSON outputs need more.
AGENT_MAX_TOKENS = {
    "grounder":    16000,
    "theorist":    16000,
    "historian":   10000,
    "synthesizer": 10000,
    "gaper":       12000,
    "vision":      16000,
    "rude":        10000,
    "thinker":     12000,
    "social":      10000,
    "scribe":      12000,
}

# Each agent runs on the heavy model (high reasoning) or the light model
# (cheap formatting / extraction). Heavy agents get the within-backend
# fallback (heavy → light) on failure; light agents are already on the cheap
# model so retrying with the same model would be redundant.
AGENT_TIER = {
    "grounder":    "heavy",
    "historian":   "heavy",
    "gaper":       "heavy",
    "vision":      "heavy",
    "theorist":    "heavy",
    "rude":        "heavy",
    "synthesizer": "heavy",
    "thinker":     "heavy",
    "social":      "light",
    "scribe":      "light",
}

# Backend defaults — overridable from config.json llm.models[<backend>].
# Anthropic-compat backends share the same SDK code path; only base_url and
# api_key_env differ. Ollama uses its own native HTTP API.
BACKEND_DEFAULTS = {
    "anthropic": {
        "transport":   "anthropic",
        "heavy":       "claude-sonnet-4-5",
        "light":       "claude-haiku-4-5-20251001",
        "base_url":    None,                  # SDK default
        "api_key_env": "ANTHROPIC_API_KEY",
    },
    "deepseek": {
        "transport":   "anthropic",
        "heavy":       "deepseek-v4-pro",
        "light":       "deepseek-v4-flash",
        "base_url":    "https://api.deepseek.com/anthropic",
        "api_key_env": "DEEPSEEK_API_KEY",
    },
    "omlx": {
        "transport":   "anthropic",
        "heavy":       "Qwen3.6-35B-A3B-4bit",
        "light":       "gemma-4-e4b-it-8bit",
        "base_url":    "http://127.0.0.1:8000",
        "api_key_env": "OMLX_API_KEY",
    },
    "ollama": {
        "transport":   "ollama",
        "heavy":       "deepseek-r1:8b",
        "light":       "llama3.2:3b",
        "base_url":    "http://localhost:11434",
        "api_key_env": None,
    },
}

VALID_BACKENDS = tuple(BACKEND_DEFAULTS.keys())


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self):
        self._load_env_first()
        self._backends = self._build_backend_config()
        self._active_backend_name   = self._resolve_active_backend()
        self._fallback_backend_name = self._resolve_fallback_backend()
        self._anthropic_clients: dict[str, Anthropic] = {}
        logger.info(
            f"LLM router initialized — active: {self._active_backend_name}, "
            f"fallback: {self._fallback_backend_name or 'none'}"
        )

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def _load_env_first(self):
        """Load .env before reading env vars — keys.py may not have run yet."""
        from pathlib import Path
        env_path = Path(__file__).parent.parent / ".env"
        if not env_path.exists():
            return
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, _, value = line.partition("=")
                    key   = key.strip()
                    value = value.strip()
                    if key and value and key not in os.environ:
                        os.environ[key] = value

    def _build_backend_config(self) -> dict:
        """Merge config.json llm.models overrides on top of BACKEND_DEFAULTS."""
        merged = {name: dict(cfg) for name, cfg in BACKEND_DEFAULTS.items()}
        try:
            from core.utils import load_config
            cfg = load_config()
        except Exception as e:
            logger.debug(f"config.json not loaded for llm overrides: {e}")
            return merged
        llm_cfg = cfg.get("llm", {})
        for name, override in llm_cfg.get("models", {}).items():
            if name not in merged:
                logger.warning(f"config.json llm.models has unknown backend: {name}")
                continue
            merged[name].update({k: v for k, v in override.items() if v is not None})
        return merged

    def _resolve_active_backend(self) -> str:
        """Resolution order: LLM_BACKEND env → config.json llm.backend → legacy auto-detect."""
        env_choice = os.environ.get("LLM_BACKEND", "").strip().lower()
        if env_choice:
            if env_choice not in VALID_BACKENDS:
                raise ValueError(
                    f"LLM_BACKEND={env_choice!r} is not one of {VALID_BACKENDS}"
                )
            return env_choice

        try:
            from core.utils import load_config
            cfg_choice = load_config().get("llm", {}).get("backend", "").strip().lower()
        except Exception:
            cfg_choice = ""
        if cfg_choice:
            if cfg_choice not in VALID_BACKENDS:
                raise ValueError(
                    f"config.json llm.backend={cfg_choice!r} is not one of {VALID_BACKENDS}"
                )
            return cfg_choice

        # Legacy auto-detect
        return "anthropic" if os.environ.get("ANTHROPIC_API_KEY", "").strip() else "ollama"

    def _resolve_fallback_backend(self) -> Optional[str]:
        """Cross-backend fallback is opt-in. None means 'no cross-backend fallback'."""
        env_choice = os.environ.get("LLM_FALLBACK_BACKEND", "").strip().lower()
        if env_choice:
            if env_choice not in VALID_BACKENDS:
                raise ValueError(
                    f"LLM_FALLBACK_BACKEND={env_choice!r} is not one of {VALID_BACKENDS}"
                )
            if env_choice == self._active_backend_name:
                return None
            return env_choice

        try:
            from core.utils import load_config
            cfg_choice = load_config().get("llm", {}).get("fallback_backend")
        except Exception:
            cfg_choice = None
        if not cfg_choice:
            return None
        cfg_choice = str(cfg_choice).strip().lower()
        if cfg_choice not in VALID_BACKENDS:
            raise ValueError(
                f"config.json llm.fallback_backend={cfg_choice!r} is not one of {VALID_BACKENDS}"
            )
        if cfg_choice == self._active_backend_name:
            return None
        return cfg_choice

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    def active_backend(self) -> str:
        return self._active_backend_name

    def fallback_backend(self) -> Optional[str]:
        return self._fallback_backend_name

    def active_backend_status(self) -> tuple[bool, str]:
        """
        Returns (ok, human-readable detail). Used by main.py's startup banner
        and by `python3 main.py keys`.
        """
        return self._backend_status(self._active_backend_name)

    def _backend_status(self, name: str) -> tuple[bool, str]:
        cfg = self._backends[name]
        if cfg["transport"] == "anthropic":
            api_key_env = cfg["api_key_env"]
            key_set = bool(os.environ.get(api_key_env, "").strip())
            base = cfg["base_url"] or "api.anthropic.com"
            if key_set:
                return True, f"{api_key_env} set, base_url={base}"
            return False, f"{api_key_env} not set (base_url={base})"
        # ollama
        try:
            resp = requests.get(f"{cfg['base_url']}/api/tags", timeout=2)
            if resp.status_code == 200:
                return True, f"reachable at {cfg['base_url']}"
            return False, f"HTTP {resp.status_code} from {cfg['base_url']}"
        except Exception as e:
            return False, f"unreachable at {cfg['base_url']} ({type(e).__name__})"

    # -----------------------------------------------------------------------
    # Anthropic-compatible client cache
    # -----------------------------------------------------------------------

    def _anthropic_client_for(self, backend: str) -> Optional[Anthropic]:
        cfg = self._backends[backend]
        if cfg["transport"] != "anthropic":
            return None
        if backend in self._anthropic_clients:
            return self._anthropic_clients[backend]

        api_key = os.environ.get(cfg["api_key_env"], "").strip()
        if not api_key:
            logger.warning(
                f"[{backend}] {cfg['api_key_env']} not set — backend is unavailable"
            )
            return None
        try:
            kwargs = {"api_key": api_key}
            if cfg["base_url"]:
                kwargs["base_url"] = cfg["base_url"]
            client = Anthropic(**kwargs)
            self._anthropic_clients[backend] = client
            logger.info(f"[{backend}] Anthropic-compatible client initialized")
            return client
        except Exception as e:
            logger.warning(f"[{backend}] Failed to initialize client: {e}")
            return None

    # -----------------------------------------------------------------------
    # Per-call model selection
    # -----------------------------------------------------------------------

    def _models_for(self, backend: str, agent_name: str) -> dict:
        """
        Return the (primary, fallback) model pair for this agent on this
        backend. Heavy agents call the heavy model and fall back to light;
        light agents call the light model directly with no within-backend
        fallback (it's already the cheap model).
        """
        cfg = self._backends[backend]
        tier = AGENT_TIER.get(agent_name.lower(), "heavy")
        if tier == "heavy":
            return {"primary": cfg["heavy"], "fallback": cfg["light"]}
        return {"primary": cfg["light"], "fallback": cfg["light"]}

    # -----------------------------------------------------------------------
    # Anthropic-compatible call (anthropic, deepseek, omlx)
    # -----------------------------------------------------------------------

    def _call_anthropic_compat(
        self,
        backend: str,
        prompt: str,
        system: str,
        model: str,
        agent_name: str
    ) -> Optional[str]:
        client = self._anthropic_client_for(backend)
        if not client:
            return None

        for attempt in range(1, MAX_API_RETRIES + 1):
            try:
                logger.info(
                    f"[{agent_name}] {backend} call — model: {model} "
                    f"— attempt {attempt}/{MAX_API_RETRIES}"
                )
                response = client.messages.create(
                    model=model,
                    max_tokens=AGENT_MAX_TOKENS.get(agent_name.lower(), MAX_TOKENS),
                    system=system,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = response.content[0].text
                logger.info(f"[{agent_name}] {backend} success — {len(text)} chars returned")
                return text

            except RateLimitError as e:
                logger.warning(f"[{agent_name}] {backend} rate limited — attempt {attempt}: {e}")
                if attempt < MAX_API_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS * attempt)

            except APIConnectionError as e:
                logger.warning(f"[{agent_name}] {backend} connection error — attempt {attempt}: {e}")
                if attempt < MAX_API_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)

            except APIStatusError as e:
                logger.error(f"[{agent_name}] {backend} status {e.status_code}: {e.message}")
                if e.status_code >= 500 and attempt < MAX_API_RETRIES:
                    time.sleep(RETRY_DELAY_SECONDS)
                else:
                    return None

            except Exception as e:
                logger.error(f"[{agent_name}] {backend} unexpected error: {e}")
                return None

        logger.error(f"[{agent_name}] {backend} exhausted all retries")
        return None

    # -----------------------------------------------------------------------
    # Ollama call
    # -----------------------------------------------------------------------

    def _call_ollama(
        self,
        prompt: str,
        system: str,
        model: str,
        agent_name: str
    ) -> Optional[str]:
        base_url = self._backends["ollama"]["base_url"]
        url = f"{base_url}/api/chat"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt}
            ],
            "stream": False,
            "options": {
                "num_predict": MAX_TOKENS,
                "temperature": 0.7
            }
        }

        try:
            logger.info(f"[{agent_name}] ollama call — model: {model}")
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            text = data["message"]["content"]
            logger.info(f"[{agent_name}] ollama success — {len(text)} chars returned")
            return text

        except requests.exceptions.ConnectionError:
            logger.error(f"[{agent_name}] Ollama not reachable at {base_url} — is it running?")
            return None

        except requests.exceptions.Timeout:
            logger.error(f"[{agent_name}] Ollama timed out after 300s")
            return None

        except Exception as e:
            logger.error(f"[{agent_name}] Ollama error: {e}")
            return None

    def _check_ollama_model(self, model: str) -> bool:
        base_url = self._backends["ollama"]["base_url"]
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return any(m == model or m.startswith(model.split(":")[0]) for m in models)
        except Exception:
            return False

    # -----------------------------------------------------------------------
    # Dispatch
    # -----------------------------------------------------------------------

    def _invoke(
        self,
        backend: str,
        prompt: str,
        system: str,
        model: str,
        agent_name: str
    ) -> Optional[str]:
        transport = self._backends[backend]["transport"]
        if transport == "anthropic":
            return self._call_anthropic_compat(backend, prompt, system, model, agent_name)
        if transport == "ollama":
            if not self._check_ollama_model(model):
                logger.warning(f"[{agent_name}] Ollama model {model} not available")
                return None
            return self._call_ollama(prompt, system, model, agent_name)
        logger.error(f"[{agent_name}] Unknown transport for backend {backend}: {transport}")
        return None

    def _try_backend(
        self,
        backend: str,
        prompt: str,
        system: str,
        agent_name: str
    ) -> Optional[str]:
        """Try the backend's primary model, then its fallback model if different."""
        models = self._models_for(backend, agent_name)
        result = self._invoke(backend, prompt, system, models["primary"], agent_name)
        if result is not None:
            return result
        if models["fallback"] != models["primary"]:
            logger.info(f"[{agent_name}] {backend}: primary failed, trying fallback model {models['fallback']}")
            return self._invoke(backend, prompt, system, models["fallback"], agent_name)
        return None

    # -----------------------------------------------------------------------
    # Main public interface
    # -----------------------------------------------------------------------

    def call(
        self,
        prompt: str,
        system: str = "You are a helpful research assistant.",
        agent_name: str = "unknown",
        force_local: bool = False
    ) -> str:
        """
        Route an LLM call.

        Order:
          1. Active backend's heavy model
          2. Active backend's light model (skipped if same as heavy)
          3. Cross-backend fallback (only if configured) — heavy then light
          4. Raise RuntimeError

        force_local=True overrides backend selection and uses ollama directly
        (kept for backward compat; no current caller passes this).
        """
        primary_backend = "ollama" if force_local else self._active_backend_name
        result = self._try_backend(primary_backend, prompt, system, agent_name)
        if result is not None:
            return result

        if not force_local and self._fallback_backend_name:
            logger.info(
                f"[{agent_name}] {primary_backend} failed — trying fallback backend "
                f"{self._fallback_backend_name}"
            )
            result = self._try_backend(self._fallback_backend_name, prompt, system, agent_name)
            if result is not None:
                return result

        raise RuntimeError(
            f"[{agent_name}] All LLM backends failed. "
            f"Active: {primary_backend}, fallback: {self._fallback_backend_name or 'none'}. "
            f"Check API keys and backend availability."
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: Optional[LLMClient] = None


def get_client() -> LLMClient:
    global _client
    if _client is None:
        _client = LLMClient()
    return _client


def reset_client() -> None:
    """Drop the cached client. Tests use this after mutating env or config."""
    global _client
    _client = None


def call(
    prompt: str,
    system: str = "You are a helpful research assistant.",
    agent_name: str = "unknown",
    force_local: bool = False
) -> str:
    """Convenience function — use this in agents."""
    return get_client().call(prompt, system, agent_name, force_local)
