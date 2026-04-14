from pathlib import Path

import pandas as pd
import pytest

from deployment_audit.benchmark.manifest import build_benchmark_manifest
from deployment_audit.execution.backend_llm import HuggingFaceCausalLMConfig, LLMBackend, load_llm_backend_config
from deployment_audit.execution.runner import load_backend, run_policy_grid
from deployment_audit.policy.grid import build_policy_grid


class _FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"

    def __call__(self, text: str, add_special_tokens: bool = True, return_tensors: str | None = None):
        token_count = max(2, len(text.split()))
        return {"input_ids": [[index + 1 for index in range(token_count)]]}


def test_load_backend_requires_config_for_llm() -> None:
    with pytest.raises(ValueError, match="backend configuration path"):
        load_backend("llm")


def test_llm_backend_config_roundtrip(tmp_path: Path) -> None:
    config_path = tmp_path / "llm_backend.json"
    config_path.write_text(
        '{"adapter_type": "huggingface_causal_lm", "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "device": "cpu"}',
        encoding="utf-8",
    )
    config = load_llm_backend_config(config_path)
    assert config.adapter_type == "huggingface_causal_lm"
    assert config.model_name_or_path == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert config.device == "cpu"


def test_llm_backend_requires_model_name_or_path() -> None:
    with pytest.raises(ValueError, match="model_name_or_path"):
        HuggingFaceCausalLMConfig.from_dict({"adapter_type": "huggingface_causal_lm"})


def test_llm_backend_run_policy_builds_execution_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    manifest = build_benchmark_manifest(
        output_root=tmp_path / "benchmark",
        benchmark_name="arithmetic-sum-varlen",
        benchmark_version="v1",
        task_name="verify_sum",
        operation_name="sum",
        generator_version="v1",
        data_seed=20260322,
        split_seed=20260322,
        n_examples=3,
        variable_length=True,
    )
    examples = pd.read_csv(manifest.example_manifest_path).merge(pd.read_csv(manifest.split_manifest_path), on="example_id", how="inner")
    policy = build_policy_grid("two_action_threshold", "bin_margin")[0]
    backend = LLMBackend(
        HuggingFaceCausalLMConfig(
            adapter_type="huggingface_causal_lm",
            model_name_or_path="tiny-test-model",
            device="cpu",
        )
    )
    backend._tokenizer = _FakeTokenizer()
    backend._model = object()
    backend._torch = object()
    backend._device = "cpu"
    backend._yes_ids = [1]
    backend._no_ids = [2]

    def fake_score(prompt: str):
        if "Candidate answer" in prompt:
            return 0.8, 0.8, 12
        return 0.5, 0.5, 12

    monkeypatch.setattr(backend, "_score_binary_verification", fake_score)

    records = backend.run_policy(benchmark_manifest=manifest, examples=examples, policy=policy)
    assert len(records) == len(examples)
    assert all(record.backend_name == "llm" for record in records)
    assert all(record.benchmark_manifest_hash == manifest.content_hash() for record in records)
    assert all(record.policy_grid_hash == policy.grid_hash for record in records)
    assert all(record.energy_joules > 0 for record in records)


def test_run_policy_grid_llm_path_requires_backend_config(tmp_path: Path) -> None:
    manifest = build_benchmark_manifest(
        output_root=tmp_path / "benchmark",
        benchmark_name="arithmetic-sum-varlen",
        benchmark_version="v1",
        task_name="verify_sum",
        operation_name="sum",
        generator_version="v1",
        data_seed=20260322,
        split_seed=20260322,
        n_examples=2,
        variable_length=True,
    )
    policy_grid = build_policy_grid("two_action_threshold", "bin_margin")[:1]
    with pytest.raises(ValueError, match="backend configuration path"):
        run_policy_grid(
            benchmark_manifest=manifest,
            policy_grid=policy_grid,
            backend_name="llm",
            output_path=tmp_path / "execution_records.csv",
        )
