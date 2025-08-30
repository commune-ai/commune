"""Basic test file to verify testing infrastructure."""


def test_environment_setup() -> None:
    """Verify that the testing environment is properly set up."""
    assert True, "Basic test to verify pytest is working"


def test_type_checking() -> None:
    """Verify that type checking is working."""

    def typed_function(param: str) -> str:
        return param.upper()

    result: str = typed_function("test")
    assert result == "TEST"
