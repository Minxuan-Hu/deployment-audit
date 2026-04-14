import pytest

from deployment_audit.cli.execution_cli import _validate_backend_arguments as validate_execution_backend_arguments
from deployment_audit.cli.study_cli import _validate_backend_arguments as validate_study_backend_arguments


def test_execution_cli_requires_backend_config_for_llm() -> None:
    with pytest.raises(ValueError, match="--backend-config"):
        validate_execution_backend_arguments("llm", None)


def test_study_cli_requires_backend_config_for_llm() -> None:
    with pytest.raises(ValueError, match="--backend-config"):
        validate_study_backend_arguments("llm", None)


def test_mock_backend_does_not_require_backend_config() -> None:
    validate_execution_backend_arguments("mock", None)
    validate_study_backend_arguments("mock", None)
