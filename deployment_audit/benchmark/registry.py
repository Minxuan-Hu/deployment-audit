from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class GeneratorDefinition:
    generator_name: str
    generator_version: str
    operation_name: str
    supports_variable_length: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkDefinition:
    benchmark_name: str
    benchmark_version: str
    task_name: str
    operation_name: str
    generator_name: str
    generator_version: str
    variable_length: bool
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GENERATOR_REGISTRY: dict[tuple[str, str], GeneratorDefinition] = {
    ("arithmetic_sum", "v1"): GeneratorDefinition(
        generator_name="arithmetic_sum",
        generator_version="v1",
        operation_name="sum",
        supports_variable_length=True,
        description="Deterministic arithmetic verification benchmark with frozen sum generator semantics.",
    ),
    ("arithmetic_product", "v1"): GeneratorDefinition(
        generator_name="arithmetic_product",
        generator_version="v1",
        operation_name="product",
        supports_variable_length=True,
        description="Deterministic arithmetic verification benchmark with frozen product generator semantics.",
    ),
}

BENCHMARK_REGISTRY: dict[tuple[str, str], BenchmarkDefinition] = {
    ("arithmetic-sum-varlen", "v1"): BenchmarkDefinition(
        benchmark_name="arithmetic-sum-varlen",
        benchmark_version="v1",
        task_name="verify_sum",
        operation_name="sum",
        generator_name="arithmetic_sum",
        generator_version="v1",
        variable_length=True,
        description="Variable-length arithmetic sum verification benchmark.",
    ),
    ("arithmetic-sum-fixed", "v1"): BenchmarkDefinition(
        benchmark_name="arithmetic-sum-fixed",
        benchmark_version="v1",
        task_name="verify_sum",
        operation_name="sum",
        generator_name="arithmetic_sum",
        generator_version="v1",
        variable_length=False,
        description="Fixed-length arithmetic sum verification benchmark.",
    ),
    ("arithmetic-product-varlen", "v1"): BenchmarkDefinition(
        benchmark_name="arithmetic-product-varlen",
        benchmark_version="v1",
        task_name="verify_product",
        operation_name="product",
        generator_name="arithmetic_product",
        generator_version="v1",
        variable_length=True,
        description="Variable-length arithmetic product verification benchmark.",
    ),
    ("arithmetic-product-fixed", "v1"): BenchmarkDefinition(
        benchmark_name="arithmetic-product-fixed",
        benchmark_version="v1",
        task_name="verify_product",
        operation_name="product",
        generator_name="arithmetic_product",
        generator_version="v1",
        variable_length=False,
        description="Fixed-length arithmetic product verification benchmark.",
    ),
}


def get_generator_definition(generator_name: str, generator_version: str) -> GeneratorDefinition:
    key = (generator_name, generator_version)
    if key not in GENERATOR_REGISTRY:
        raise KeyError(f"Unknown generator definition: {key}")
    return GENERATOR_REGISTRY[key]


def get_benchmark_definition(benchmark_name: str, benchmark_version: str = "v1") -> BenchmarkDefinition:
    key = (benchmark_name, benchmark_version)
    if key not in BENCHMARK_REGISTRY:
        raise KeyError(f"Unknown benchmark definition: {key}")
    return BENCHMARK_REGISTRY[key]


def list_generator_definitions() -> list[dict[str, Any]]:
    return [definition.to_dict() for _, definition in sorted(GENERATOR_REGISTRY.items())]


def list_benchmark_definitions() -> list[dict[str, Any]]:
    return [definition.to_dict() for _, definition in sorted(BENCHMARK_REGISTRY.items())]
