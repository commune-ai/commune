"""Tests for validation utilities."""

from modnet.utils.validation import validate_module_metadata


def test_validate_metadata_valid() -> None:
    """Test metadata validation with valid data."""
    metadata = {
        "name": "test-module",
        "version": "1.0.0",
        "description": "Test module",
        "extra": "field",  # Extra fields are allowed
    }
    assert validate_module_metadata(metadata) is True


def test_validate_metadata_missing_fields() -> None:
    """Test metadata validation with missing fields."""
    metadata = {
        "name": "test-module",
        "version": "1.0.0",
        # Missing description
    }
    assert validate_module_metadata(metadata) is False


def test_validate_metadata_empty() -> None:
    """Test metadata validation with empty dict."""
    assert validate_module_metadata({}) is False
