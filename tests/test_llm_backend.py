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
    pad_token_id = 0
    eos_token_id = 1

    def __call__(self, text: str, add_special_tokens: bool = True, return_tensors: str | None = None):
        token_count = max(2, len(text.split()))
        data = [[index + 1 for index in range(token_count)]]
        if return_tensors == "pt":
            import torch

            return {
                "input_ids": torch.tensor(data),
                "attention_mask": torch.ones((1, token_count), dtype=torch.long),
            }
        return {"input_ids": data}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        return " ".join(str(token_id) for token_id in token_ids)


def test_load_backend_requires_config_for_llm() -> None:
    with pytest.raises(ValueError, match="backend configuration path"):
        load_backend("llm")


def test_llm_backend_config_roundtrip(tmp_path: Path) -> None:
    config_path = tmp_path / "llm_backend.json"
    config_path.write_text(
        '{"adapter_type": "huggingface_causal_lm", "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", "device": "cpu", "generation_max_new_tokens": 8}',
        encoding="utf-8",
    )
    config = load_llm_backend_config(config_path)
    assert config.adapter_type == "huggingface_causal_lm"
    assert config.model_name_or_path == "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    assert config.device == "cpu"
    assert config.generation_max_new_tokens == 8


def test_llm_backend_requires_model_name_or_path() -> None:
    with pytest.raises(ValueError, match="model_name_or_path"):
        HuggingFaceCausalLMConfig.from_dict({"adapter_type": "huggingface_causal_lm"})


def test_llm_backend_prompt_includes_variable_length_context() -> None:
    backend = LLMBackend(
        HuggingFaceCausalLMConfig(
            adapter_type="huggingface_causal_lm",
            model_name_or_path="tiny-test-model",
            device="cpu",
        )
    )
    short_row = pd.Series({"operation_name": "sum", "a": 12, "b": 34, "pad_words": 4})
    long_row = pd.Series({"operation_name": "sum", "a": 12, "b": 34, "pad_words": 12})
    short_prompt = backend._build_prompt(short_row)
    long_prompt = backend._build_prompt(long_row)
    assert short_prompt.split("Background context: ", 1)[1].split("\n", 1)[0].split() == ["context"] * 4
    assert long_prompt.split("Background context: ", 1)[1].split("\n", 1)[0].split() == ["context"] * 12
    assert len(long_prompt.split()) > len(short_prompt.split())


def test_score_candidate_agreement_spreads_scores() -> None:
    backend = LLMBackend(
        HuggingFaceCausalLMConfig(
            adapter_type="huggingface_causal_lm",
            model_name_or_path="tiny-test-model",
            device="cpu",
        )
    )
    exact_score, exact_margin, exact_yes, _ = backend._score_candidate_agreement(candidate_answer=4600, generated_answer=4600, digits=4)
    near_score, near_margin, near_yes, _ = backend._score_candidate_agreement(candidate_answer=4700, generated_answer=4600, digits=4)
    far_score, far_margin, far_yes, _ = backend._score_candidate_agreement(candidate_answer=4900, generated_answer=4600, digits=4)
    assert exact_yes is True
    assert near_yes is False
    assert far_yes is False
    assert exact_score > near_score > far_score
    assert exact_margin > near_margin
    assert far_margin > near_margin




def test_policy_accepts_can_threshold_bin_score() -> None:
    policy = build_policy_grid("two_action_threshold", "bin_score")[0]
    assert policy.score_name == "bin_score"
    assert policy.accepts(pad_words=0, difficulty_tier="easy", route_score=0.1, bin_margin=0.2, bin_score=0.9) is True
    assert policy.accepts(pad_words=0, difficulty_tier="easy", route_score=0.9, bin_margin=0.9, bin_score=0.1) is False

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

    def fake_generate_answer(prompt: str):
        return "1234", max(2, len(prompt.split())), 1, 5.0

    monkeypatch.setattr(backend, "_generate_answer", fake_generate_answer)

    records = backend.run_policy(benchmark_manifest=manifest, examples=examples, policy=policy)
    assert len(records) == len(examples)
    assert all(record.backend_name == "llm" for record in records)
    assert all(record.benchmark_manifest_hash == manifest.content_hash() for record in records)
    assert all(record.policy_grid_hash == policy.grid_hash for record in records)
    assert all(record.energy_joules > 0 for record in records)
    assert all("generated_answer" in record.prompt_features for record in records)


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


def test_llm_backend_rejects_legacy_verify_prompt_template() -> None:
    with pytest.raises(ValueError, match="Unsupported prompt_template_version"):
        HuggingFaceCausalLMConfig.from_dict(
            {
                "adapter_type": "huggingface_causal_lm",
                "model_name_or_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "prompt_template_version": "verify_arithmetic_v1",
            }
        )
