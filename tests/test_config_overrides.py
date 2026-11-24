"""Tests for runtime backend configuration overrides."""

import pytest

from run_pipeline import apply_config_overrides
from src.agentic import graph
from src.utils.config import ModelBackendConfig, ModelBackendsConfig


def _config() -> ModelBackendsConfig:
    return ModelBackendsConfig(
        planner=ModelBackendConfig(backend="openai.gpt-4o"),
        executor=ModelBackendConfig(backend="openai.gpt-4o"),
        summarizer=ModelBackendConfig(backend="openai.gpt-4o"),
    )


def test_overrides_pydantic_model():
    """Overrides should update and revalidate a Pydantic config model."""
    updated = apply_config_overrides(
        _config(),
        [
            "planner.backend=openai.gpt-4o-mini",
            "executor.temperature=0.5",
        ],
    )

    assert isinstance(updated, ModelBackendsConfig)
    assert updated.planner.backend == "openai.gpt-4o-mini"
    assert updated.executor.temperature == 0.5
    assert updated.summarizer.backend == "openai.gpt-4o"


def test_overrides_plain_dict():
    """Plain dictionaries should still work for callers that pass a dict."""
    updated = apply_config_overrides(
        {"planner": {"backend": "openai.gpt-4o"}},
        ["planner.backend=openai.gpt-4o-mini"],
    )

    assert updated == {"planner": {"backend": "openai.gpt-4o-mini"}}


def test_invalid_override_is_rejected():
    """Invalid backend values should fail validation instead of being ignored."""
    with pytest.raises(ValueError, match="Backend must start with"):
        apply_config_overrides(_config(), ["planner.backend=not-a-backend"])


def test_run_pipeline_forwards_runtime_config(monkeypatch):
    """run_pipeline should pass runtime config through to LangGraph invoke."""
    captured = {}

    class FakeCompiledGraph:
        def invoke(self, state, config=None):
            captured["config"] = config
            return {"replan_history": []}

    monkeypatch.setattr(graph, "compile_graph", lambda config=None: FakeCompiledGraph())

    runtime_config = {
        "configurable": {
            "planner": ModelBackendConfig(backend="openai.gpt-4o-mini")
        }
    }
    result = graph.run_pipeline(
        query="Rate this image.",
        image_path="image.png",
        config=runtime_config,
    )

    assert captured["config"] is runtime_config
    assert result == {"replan_history": []}
