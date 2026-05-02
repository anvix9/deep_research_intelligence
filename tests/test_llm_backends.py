"""
Test: LLM backend router (4-backend fan-out)
---------------------------------------------
Verifies backend selection, within-backend fallback (heavy→light model),
opt-in cross-backend fallback, and that each backend dispatches through the
correct transport (Anthropic SDK vs Ollama HTTP).

No live API calls. Mocks at the SDK / HTTP boundary.

Run as a standalone script:  python3 tests/test_llm_backends.py
Or under pytest:             python3 -m pytest tests/test_llm_backends.py -v
"""
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

# Wipe LLM-related env so test runs don't see your shell config.
for _v in ("LLM_BACKEND", "LLM_FALLBACK_BACKEND",
           "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "OMLX_API_KEY"):
    os.environ.pop(_v, None)


_LLM_ENV_KEYS = ("LLM_BACKEND", "LLM_FALLBACK_BACKEND",
                 "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "OMLX_API_KEY")


def _fresh_client(env: dict):
    """Reset the singleton, wipe LLM env, then apply this test's env, build a client."""
    from core import llm
    llm.reset_client()
    for k in _LLM_ENV_KEYS:
        os.environ.pop(k, None)
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return llm, llm.get_client()


def _fake_anthropic_response(text: str = "ok"):
    """Build a mock that mimics anthropic.messages.create() return shape."""
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


# ─── Test 1: Resolution order ──────────────────────────────────────────────

def test_resolution_order_env_wins_over_config():
    llm, c = _fresh_client({"LLM_BACKEND": "omlx"})
    assert c.active_backend() == "omlx", c.active_backend()
    print("✓ env LLM_BACKEND overrides config.json default (anthropic)")


def test_resolution_legacy_auto_detect_anthropic():
    """No env, ANTHROPIC_API_KEY present → anthropic. config.json default also says anthropic, so this is consistent."""
    llm, c = _fresh_client({"ANTHROPIC_API_KEY": "sk-test-fake"})
    assert c.active_backend() == "anthropic"
    print("✓ legacy auto-detect picks anthropic when key is set")


def test_invalid_backend_raises():
    from core import llm as _llm
    _llm.reset_client()
    os.environ["LLM_BACKEND"] = "gpt-7"
    try:
        _llm.get_client()
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "gpt-7" in str(e)
    finally:
        os.environ.pop("LLM_BACKEND", None)
    print("✓ invalid LLM_BACKEND raises ValueError")


def test_fallback_equals_active_collapses_to_none():
    llm, c = _fresh_client({"LLM_BACKEND": "omlx", "LLM_FALLBACK_BACKEND": "omlx"})
    assert c.fallback_backend() is None
    print("✓ fallback_backend == active collapses to None")


# ─── Test 2: Per-agent tier (heavy vs light) ───────────────────────────────

def test_models_for_heavy_agent():
    llm, c = _fresh_client({"LLM_BACKEND": "deepseek"})
    m = c._models_for("deepseek", "grounder")
    assert m == {"primary": "deepseek-v4-pro", "fallback": "deepseek-v4-flash"}, m
    print("✓ heavy agent (grounder) → primary=heavy, fallback=light")


def test_models_for_light_agent():
    llm, c = _fresh_client({"LLM_BACKEND": "deepseek"})
    m = c._models_for("deepseek", "social")
    assert m == {"primary": "deepseek-v4-flash", "fallback": "deepseek-v4-flash"}, m
    print("✓ light agent (social) → primary=fallback=light (no within-backend retry)")


# ─── Test 3: Anthropic-compat dispatch (anthropic, deepseek, omlx) ─────────

def test_anthropic_compat_dispatch_for_anthropic():
    llm, c = _fresh_client({"LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": "sk-anthropic"})
    fake = MagicMock()
    fake.messages.create.return_value = _fake_anthropic_response("hello")
    with patch("core.llm.Anthropic", return_value=fake) as ctor:
        out = c.call("p", "s", "grounder")
    assert out == "hello"
    # Called once for client construction, with anthropic key, no base_url override
    ctor.assert_called_once_with(api_key="sk-anthropic")
    fake.messages.create.assert_called_once()
    assert fake.messages.create.call_args.kwargs["model"] == "claude-sonnet-4-5"
    print("✓ anthropic backend → Anthropic() with no base_url, heavy model")


def test_anthropic_compat_dispatch_for_deepseek():
    llm, c = _fresh_client({"LLM_BACKEND": "deepseek", "DEEPSEEK_API_KEY": "sk-deepseek"})
    fake = MagicMock()
    fake.messages.create.return_value = _fake_anthropic_response("ds")
    with patch("core.llm.Anthropic", return_value=fake) as ctor:
        out = c.call("p", "s", "grounder")
    assert out == "ds"
    ctor.assert_called_once_with(
        api_key="sk-deepseek",
        base_url="https://api.deepseek.com/anthropic",
    )
    assert fake.messages.create.call_args.kwargs["model"] == "deepseek-v4-pro"
    print("✓ deepseek backend → Anthropic(base_url=deepseek) with DEEPSEEK_API_KEY, deepseek-v4-pro")


def test_anthropic_compat_dispatch_for_omlx():
    llm, c = _fresh_client({"LLM_BACKEND": "omlx", "OMLX_API_KEY": "sk-omlx"})
    fake = MagicMock()
    fake.messages.create.return_value = _fake_anthropic_response("local")
    with patch("core.llm.Anthropic", return_value=fake) as ctor:
        out = c.call("p", "s", "grounder")
    assert out == "local"
    ctor.assert_called_once_with(api_key="sk-omlx", base_url="http://127.0.0.1:8000")
    assert fake.messages.create.call_args.kwargs["model"] == "Qwen3.6-35B-A3B-4bit"
    print("✓ omlx backend → Anthropic(base_url=127.0.0.1:8000) with OMLX_API_KEY, Qwen3.6 model")


def test_deepseek_does_not_use_anthropic_api_key():
    """The DeepSeek docs say 'ANTHROPIC_API_KEY corresponds to x-api-key' but that's
    misleading for our setup — we must NOT read $ANTHROPIC_API_KEY when the user
    selected the deepseek backend, or the real Anthropic key would leak."""
    llm, c = _fresh_client({
        "LLM_BACKEND": "deepseek",
        "ANTHROPIC_API_KEY": "sk-DO-NOT-LEAK",
        "DEEPSEEK_API_KEY":  "sk-deepseek-correct",
    })
    fake = MagicMock()
    fake.messages.create.return_value = _fake_anthropic_response("ok")
    with patch("core.llm.Anthropic", return_value=fake) as ctor:
        c.call("p", "s", "grounder")
    args = ctor.call_args
    assert args.kwargs["api_key"] == "sk-deepseek-correct"
    assert "sk-DO-NOT-LEAK" not in str(args)
    print("✓ deepseek does NOT use $ANTHROPIC_API_KEY (no key leak)")


# ─── Test 4: Within-backend fallback (heavy → light) ───────────────────────

def test_within_backend_fallback_for_heavy_agent():
    """If heavy model returns None, the router calls the light model."""
    llm, c = _fresh_client({"LLM_BACKEND": "deepseek", "DEEPSEEK_API_KEY": "k"})
    fake = MagicMock()
    # First create() raises a 5xx-style error → returns None internally; second succeeds.
    from anthropic import APIStatusError
    err = APIStatusError(message="server error", response=MagicMock(status_code=500), body=None)
    err.status_code = 500
    fake.messages.create.side_effect = [err, err, err, _fake_anthropic_response("recovered")]
    with patch("core.llm.Anthropic", return_value=fake):
        out = c.call("p", "s", "grounder")
    assert out == "recovered"
    # Verify the second call used the light model
    last_model = fake.messages.create.call_args.kwargs["model"]
    assert last_model == "deepseek-v4-flash", last_model
    print("✓ within-backend fallback: heavy fails → light model invoked")


# ─── Test 5: Cross-backend fallback (opt-in only) ──────────────────────────

def test_no_cross_backend_fallback_by_default_raises():
    """With fallback_backend=None, total backend failure raises RuntimeError."""
    llm, c = _fresh_client({"LLM_BACKEND": "deepseek", "DEEPSEEK_API_KEY": "k"})
    assert c.fallback_backend() is None
    fake = MagicMock()
    from anthropic import APIStatusError
    err = APIStatusError(message="bad request", response=MagicMock(status_code=400), body=None)
    err.status_code = 400  # 4xx is not retryable → returns None immediately
    fake.messages.create.side_effect = err
    with patch("core.llm.Anthropic", return_value=fake):
        try:
            c.call("p", "s", "grounder")
            assert False, "should have raised"
        except RuntimeError as e:
            assert "deepseek" in str(e)
    print("✓ no cross-backend fallback by default → raises RuntimeError")


def test_cross_backend_fallback_when_configured():
    """With LLM_FALLBACK_BACKEND=ollama, primary failure routes to Ollama."""
    llm, c = _fresh_client({
        "LLM_BACKEND":          "deepseek",
        "DEEPSEEK_API_KEY":     "k",
        "LLM_FALLBACK_BACKEND": "ollama",
    })
    assert c.fallback_backend() == "ollama"

    fake_anthropic = MagicMock()
    from anthropic import APIStatusError
    err = APIStatusError(message="bad", response=MagicMock(status_code=400), body=None)
    err.status_code = 400
    fake_anthropic.messages.create.side_effect = err

    # Mock ollama side: /api/tags reports model present, /api/chat returns text.
    def fake_get(url, *a, **kw):
        r = MagicMock()
        r.status_code = 200
        r.raise_for_status.return_value = None
        r.json.return_value = {"models": [{"name": "deepseek-r1:8b"}]}
        return r

    def fake_post(url, *a, **kw):
        r = MagicMock()
        r.raise_for_status.return_value = None
        r.json.return_value = {"message": {"content": "ollama-saved-the-day"}}
        return r

    with patch("core.llm.Anthropic", return_value=fake_anthropic), \
         patch("core.llm.requests.get", side_effect=fake_get), \
         patch("core.llm.requests.post", side_effect=fake_post):
        out = c.call("p", "s", "grounder")

    assert out == "ollama-saved-the-day", out
    print("✓ cross-backend fallback fires only when LLM_FALLBACK_BACKEND is set")


# ─── Test 6: Backend status introspection ──────────────────────────────────

def test_active_backend_status_anthropic():
    llm, c = _fresh_client({"LLM_BACKEND": "anthropic", "ANTHROPIC_API_KEY": "x"})
    ok, detail = c.active_backend_status()
    assert ok is True
    assert "ANTHROPIC_API_KEY" in detail
    print("✓ active_backend_status reports anthropic OK when key set")


def test_active_backend_status_omlx_missing_key():
    llm, c = _fresh_client({"LLM_BACKEND": "omlx"})
    ok, detail = c.active_backend_status()
    assert ok is False
    assert "OMLX_API_KEY" in detail
    print("✓ active_backend_status reports omlx not-ready when key missing")


# ─── Driver ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_resolution_order_env_wins_over_config,
        test_resolution_legacy_auto_detect_anthropic,
        test_invalid_backend_raises,
        test_fallback_equals_active_collapses_to_none,
        test_models_for_heavy_agent,
        test_models_for_light_agent,
        test_anthropic_compat_dispatch_for_anthropic,
        test_anthropic_compat_dispatch_for_deepseek,
        test_anthropic_compat_dispatch_for_omlx,
        test_deepseek_does_not_use_anthropic_api_key,
        test_within_backend_fallback_for_heavy_agent,
        test_no_cross_backend_fallback_by_default_raises,
        test_cross_backend_fallback_when_configured,
        test_active_backend_status_anthropic,
        test_active_backend_status_omlx_missing_key,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as e:
            print(f"✗ {fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {fn.__name__}: unexpected {type(e).__name__}: {e}")
            failed += 1
    print()
    if failed:
        print(f"FAILED: {failed}/{len(tests)}")
        sys.exit(1)
    print(f"ALL PASSED: {len(tests)}/{len(tests)}")
