from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any

import pandas as pd

from deployment_audit.execution.backend_protocol import BackendProtocol
from deployment_audit.policy.family import PolicySpec
from deployment_audit.schemas.benchmark_manifest import BenchmarkManifest
from deployment_audit.schemas.execution_record import ExecutionRecord


@dataclass(frozen=True)
class HuggingFaceCausalLMConfig:
    adapter_type: str
    model_name_or_path: str
    device: str = "auto"
    dtype: str = "auto"
    trust_remote_code: bool = False
    route_token_scale: int = 1024
    energy_mode: str = "latency_proxy"
    energy_coefficient_joules_per_ms: float = 0.001
    prompt_template_version: str = "verify_arithmetic_v1"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HuggingFaceCausalLMConfig":
        adapter_type = str(payload.get("adapter_type", "huggingface_causal_lm"))
        if adapter_type != "huggingface_causal_lm":
            raise ValueError(f"Unsupported llm adapter_type: {adapter_type}")
        if "model_name_or_path" not in payload:
            raise ValueError("LLM backend config must define model_name_or_path.")
        route_token_scale = int(payload.get("route_token_scale", 1024))
        if route_token_scale <= 0:
            raise ValueError("route_token_scale must be positive.")
        energy_mode = str(payload.get("energy_mode", "latency_proxy"))
        if energy_mode != "latency_proxy":
            raise ValueError(f"Unsupported energy_mode: {energy_mode}")
        energy_coefficient = float(payload.get("energy_coefficient_joules_per_ms", 0.001))
        if energy_coefficient <= 0:
            raise ValueError("energy_coefficient_joules_per_ms must be positive.")
        return cls(
            adapter_type=adapter_type,
            model_name_or_path=str(payload["model_name_or_path"]),
            device=str(payload.get("device", "auto")),
            dtype=str(payload.get("dtype", "auto")),
            trust_remote_code=bool(payload.get("trust_remote_code", False)),
            route_token_scale=route_token_scale,
            energy_mode=energy_mode,
            energy_coefficient_joules_per_ms=energy_coefficient,
            prompt_template_version=str(payload.get("prompt_template_version", "verify_arithmetic_v1")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "adapter_type": self.adapter_type,
            "model_name_or_path": self.model_name_or_path,
            "device": self.device,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "route_token_scale": self.route_token_scale,
            "energy_mode": self.energy_mode,
            "energy_coefficient_joules_per_ms": self.energy_coefficient_joules_per_ms,
            "prompt_template_version": self.prompt_template_version,
        }


def load_llm_backend_config(path: str | Path) -> HuggingFaceCausalLMConfig:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("LLM backend config must be a JSON object.")
    return HuggingFaceCausalLMConfig.from_dict(payload)


class LLMBackend(BackendProtocol):
    name = "llm"

    def __init__(self, config: HuggingFaceCausalLMConfig):
        self.config = config
        self.version = f"{config.adapter_type}:{config.model_name_or_path}"
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._device = None
        self._yes_ids: list[int] | None = None
        self._no_ids: list[int] | None = None

    def _load_runtime(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "LLMBackend requires optional dependencies. Install deployment-audit with llm extras or install torch and transformers."
            ) from exc

        device = self.config.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("LLM backend requested CUDA but torch.cuda.is_available() is False.")

        dtype_name = self.config.dtype
        if dtype_name == "auto":
            dtype_name = "float16" if device.startswith("cuda") else "float32"
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if dtype_name not in dtype_map:
            raise ValueError(f"Unsupported dtype: {dtype_name}")
        torch_dtype = dtype_map[dtype_name]

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name_or_path, trust_remote_code=self.config.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name_or_path,
            torch_dtype=torch_dtype,
            trust_remote_code=self.config.trust_remote_code,
        )
        model.to(device)
        model.eval()

        self._torch = torch
        self._device = device
        self._tokenizer = tokenizer
        self._model = model
        self._yes_ids = self._encode_target(" yes")
        self._no_ids = self._encode_target(" no")

    def _encode_target(self, text: str) -> list[int]:
        assert self._tokenizer is not None
        token_ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        if not token_ids:
            raise ValueError(f"Tokenizer returned no token ids for target text: {text!r}")
        return [int(token_id) for token_id in token_ids]

    def _build_prompt(self, row: pd.Series) -> str:
        if self.config.prompt_template_version != "verify_arithmetic_v1":
            raise ValueError(f"Unsupported prompt_template_version: {self.config.prompt_template_version}")
        operation_label = "sum" if str(row.operation_name) == "sum" else "product"
        return (
            "You are checking whether a candidate arithmetic answer is correct.\n"
            f"Operation: {operation_label}\n"
            f"A: {int(row.a)}\n"
            f"B: {int(row.b)}\n"
            f"Candidate answer: {int(row.candidate_answer)}\n"
            "Reply with only yes or no.\n"
            "Answer:"
        )

    def _prompt_ids(self, prompt: str) -> list[int]:
        assert self._tokenizer is not None
        encoded = self._tokenizer(prompt, return_tensors="pt")
        return [int(token_id) for token_id in encoded["input_ids"][0].tolist()]

    def _continuation_logprob(self, prompt_ids: list[int], continuation_ids: list[int]) -> float:
        assert self._torch is not None and self._model is not None and self._device is not None
        full_ids = prompt_ids + continuation_ids
        input_ids = self._torch.tensor([full_ids], device=self._device)
        with self._torch.inference_mode():
            logits = self._model(input_ids=input_ids).logits
            log_probs = self._torch.log_softmax(logits[:, :-1, :], dim=-1)
        prompt_length = len(prompt_ids)
        total = 0.0
        for index, token_id in enumerate(continuation_ids):
            total += float(log_probs[0, prompt_length - 1 + index, token_id].item())
        return total

    def _score_binary_verification(self, prompt: str) -> tuple[float, float, int]:
        assert self._yes_ids is not None and self._no_ids is not None
        prompt_ids = self._prompt_ids(prompt)
        yes_logprob = self._continuation_logprob(prompt_ids=prompt_ids, continuation_ids=self._yes_ids)
        no_logprob = self._continuation_logprob(prompt_ids=prompt_ids, continuation_ids=self._no_ids)
        max_logprob = max(yes_logprob, no_logprob)
        yes_score = math.exp(yes_logprob - max_logprob)
        no_score = math.exp(no_logprob - max_logprob)
        yes_probability = yes_score / (yes_score + no_score)
        return yes_probability, max(yes_probability, 1.0 - yes_probability), len(prompt_ids)

    def _route_score(self, bin_margin: float, prompt_token_count: int) -> float:
        length_score = max(0.0, 1.0 - min(1.0, prompt_token_count / float(self.config.route_token_scale)))
        return max(0.0, min(1.0, 0.65 * bin_margin + 0.35 * length_score))

    def _energy_joules(self, latency_ms: float) -> float:
        return latency_ms * self.config.energy_coefficient_joules_per_ms

    def run_policy(self, benchmark_manifest: BenchmarkManifest, examples: pd.DataFrame, policy: PolicySpec) -> list[ExecutionRecord]:
        self._load_runtime()
        records: list[ExecutionRecord] = []
        for row in examples.itertuples(index=False):
            prompt = self._build_prompt(pd.Series(row._asdict()))
            start_time = time.perf_counter()
            yes_probability, bin_margin, prompt_token_count = self._score_binary_verification(prompt)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            predicted_yes = yes_probability >= 0.5
            route_score = self._route_score(bin_margin=bin_margin, prompt_token_count=prompt_token_count)
            accepted = policy.accepts(
                pad_words=int(row.pad_words),
                difficulty_tier=str(row.difficulty_tier),
                route_score=route_score,
                bin_margin=bin_margin,
            )
            answer_token_count = len(self._yes_ids if predicted_yes else self._no_ids)
            records.append(
                ExecutionRecord(
                    benchmark_name=benchmark_manifest.benchmark_name,
                    benchmark_version=benchmark_manifest.benchmark_version,
                    example_id=str(row.example_id),
                    split=str(row.split),
                    policy_id=policy.policy_id,
                    family_name=policy.family_name,
                    score_name=policy.score_name,
                    accepted=accepted,
                    correct=predicted_yes == bool(int(row.label)),
                    energy_joules=float(self._energy_joules(latency_ms)),
                    tokens=float(prompt_token_count + answer_token_count),
                    latency_ms=float(latency_ms),
                    route_score=float(route_score),
                    bin_margin=float(bin_margin),
                    bin_score=float(yes_probability),
                    backend_name=self.name,
                    backend_version=self.version,
                    benchmark_manifest_hash=benchmark_manifest.content_hash(),
                    policy_grid_hash=policy.grid_hash,
                    prompt_features={
                        "digits": int(row.digits),
                        "pad_words": int(row.pad_words),
                        "difficulty_tier": str(row.difficulty_tier),
                        "prompt_template_version": self.config.prompt_template_version,
                        "model_name_or_path": self.config.model_name_or_path,
                    },
                )
            )
        return records
