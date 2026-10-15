import pytest

from tide.download_models import PROVIDERS, provider_order


def test_auto_provider_order_prefers_modelscope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TIDE_MODEL_PROVIDER", raising=False)
    assert provider_order("auto") == ["modelscope", "hf-mirror", "huggingface"]


def test_auto_provider_honors_explicit_preference(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIDE_MODEL_PROVIDER", "hf-mirror")
    assert provider_order("auto") == ["hf-mirror", "modelscope", "huggingface"]


def test_invalid_environment_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIDE_MODEL_PROVIDER", "unknown")
    with pytest.raises(ValueError, match="TIDE_MODEL_PROVIDER"):
        provider_order("auto")


def test_named_provider_does_not_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TIDE_MODEL_PROVIDER", "modelscope")
    assert provider_order("huggingface") == ["huggingface"]
    assert tuple(PROVIDERS) == ("modelscope", "hf-mirror", "huggingface")
