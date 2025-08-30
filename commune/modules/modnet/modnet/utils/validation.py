"""Validation utilities for the module registry client."""

from typing import Any


def validate_module_metadata(metadata: dict[str, Any]) -> bool:
    """Validate module metadata format.

    Args:
        metadata: Module metadata to validate

    Returns:
        bool: True if metadata is valid
    """
    required_fields = {"name", "version", "description"}
    return all(field in metadata for field in required_fields)


def validate_module_id(module_id: str) -> bool:
    """Validate module ID format.

    Args:
        module_id: Module ID to validate

    Returns:
        bool: True if module ID is valid
    """
    if not isinstance(module_id, str):
        return False

    # Module ID should be non-empty and contain only alphanumeric chars, hyphens, and underscores
    return bool(module_id and module_id.replace("-", "").replace("_", "").isalnum())
