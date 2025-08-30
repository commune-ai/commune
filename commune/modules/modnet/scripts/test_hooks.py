#!/usr/bin/env python3
"""
Test hooks for GitHub Actions workflows.
This script validates the GitHub Actions setup and can be used as a pre-commit hook.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]


class GitHubActionsValidator:
    """Validates GitHub Actions workflows and setup."""

    def __init__(self, repo_root: str = ".") -> None:
        self.repo_root = Path(repo_root)
        self.workflows_dir = self.repo_root / ".github" / "workflows"
        self.actions_dir = self.repo_root / ".github" / "actions"

    def validate_workflow_syntax(self, workflow_file: Path) -> dict[str, Any]:
        """Validate YAML syntax of a workflow file."""
        result: dict[str, Any] = {
            "file": str(workflow_file),
            "valid": False,
            "errors": [],
        }
        errors: list[str] = result["errors"]

        try:
            with open(workflow_file, encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    errors.append("File is empty")
                    return result
                workflow_data = yaml.safe_load(content)

            # Check if workflow_data is None or not a dict
            if workflow_data is None:
                errors.append("YAML file is empty or invalid")
                return result

            if not isinstance(workflow_data, dict):
                errors.append("YAML root must be a dictionary")
                return result

            # Basic structure validation
            # Note: YAML parser converts 'on' to True, so we check for both
            required_keys = {"name": "name", "on": [True, "on"], "jobs": "jobs"}
            for key_name, key_variants in required_keys.items():
                if isinstance(key_variants, list):
                    # Check for multiple possible keys (like 'on' -> True)
                    if not any(variant in workflow_data for variant in key_variants):
                        errors.append(f"Missing required key: {key_name}")
                else:
                    # Check for single key
                    if key_variants not in workflow_data:
                        errors.append(f"Missing required key: {key_name}")

            # Validate jobs structure
            if "jobs" in workflow_data:
                jobs = workflow_data["jobs"]
                if isinstance(jobs, dict):
                    for job_name, job_data in jobs.items():
                        if not isinstance(job_data, dict):
                            errors.append(f"Job '{job_name}' must be a dictionary")
                            continue

                        if "runs-on" not in job_data:
                            errors.append(f"Job '{job_name}' missing 'runs-on'")

                        if "steps" not in job_data:
                            errors.append(f"Job '{job_name}' missing 'steps'")

            result["valid"] = len(errors) == 0

        except yaml.YAMLError as e:
            errors.append(f"YAML syntax error: {str(e)}")
        except Exception as e:
            errors.append(f"Unexpected error: {str(e)}")

        return result

    def check_workflow_triggers(self, workflow_file: Path) -> dict[str, Any]:
        """Check if workflow triggers are properly configured."""
        result: dict[str, Any] = {
            "file": str(workflow_file),
            "triggers": [],
            "warnings": [],
        }
        triggers_list: list[str] = result["triggers"]
        warnings: list[str] = result["warnings"]

        try:
            with open(workflow_file, encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    return result
                workflow_data = yaml.safe_load(content)

            if workflow_data is None or not isinstance(workflow_data, dict):
                return result

            # Handle YAML parsing quirk where 'on' becomes True
            triggers_key = (
                True
                if True in workflow_data
                else "on" if "on" in workflow_data else None
            )
            if triggers_key is not None:
                triggers = workflow_data[triggers_key]
                if isinstance(triggers, str):
                    triggers_list.extend([triggers])
                elif isinstance(triggers, list):
                    triggers_list.extend(triggers)
                elif isinstance(triggers, dict):
                    triggers_list.extend(list(triggers.keys()))

                # Check for common trigger patterns
                if "push" in triggers_list and "pull_request" in triggers_list:
                    # Good practice - covers both scenarios
                    pass
                elif (
                    "push" not in triggers_list and "pull_request" not in triggers_list
                ):
                    warnings.append("No push or pull_request triggers found")

        except Exception as e:
            warnings.append(f"Error checking triggers: {str(e)}")

        return result

    def validate_action_references(self, workflow_file: Path) -> dict[str, Any]:
        """Validate that referenced actions exist and use proper versions."""
        result: dict[str, Any] = {
            "file": str(workflow_file),
            "actions": [],
            "warnings": [],
        }
        actions: list[str] = result["actions"]
        warnings: list[str] = result["warnings"]

        try:
            with open(workflow_file, encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    return result
                workflow_data = yaml.safe_load(content)

            if workflow_data is None or not isinstance(workflow_data, dict):
                return result

            # Extract all action references
            if "jobs" in workflow_data:
                jobs = workflow_data["jobs"]
                if isinstance(jobs, dict):
                    for _job_name, job_data in jobs.items():
                        if isinstance(job_data, dict) and "steps" in job_data:
                            steps = job_data["steps"]
                            if isinstance(steps, list):
                                for step in steps:
                                    if isinstance(step, dict) and "uses" in step:
                                        action_ref = step["uses"]
                                        if isinstance(action_ref, str):
                                            actions.append(action_ref)

                                            # Check for version pinning
                                            if "@" not in action_ref:
                                                warnings.append(
                                                    f"Action '{action_ref}' not version pinned"
                                                )
                                            elif action_ref.endswith(
                                                "@main"
                                            ) or action_ref.endswith("@master"):
                                                warnings.append(
                                                    f"Action '{action_ref}' uses unstable branch"
                                                )

                                            # Check for local actions
                                            if action_ref.startswith("./"):
                                                local_action_path = (
                                                    self.repo_root / action_ref[2:]
                                                )
                                                if not local_action_path.exists():
                                                    warnings.append(
                                                        f"Local action '{action_ref}' not found"
                                                    )

        except Exception as e:
            warnings.append(f"Error validating actions: {str(e)}")

        return result

    def validate_all_workflows(self) -> dict[str, Any]:
        """Validate all workflow files in the repository."""
        results: dict[str, Any] = {
            "workflows": [],
            "summary": {"total": 0, "valid": 0, "invalid": 0},
        }
        workflows: list[dict[str, Any]] = results["workflows"]

        if not self.workflows_dir.exists():
            return results

        for workflow_file in self.workflows_dir.glob("*.yml"):
            if workflow_file.is_file():
                syntax_result = self.validate_workflow_syntax(workflow_file)
                trigger_result = self.check_workflow_triggers(workflow_file)
                action_result = self.validate_action_references(workflow_file)

                combined_result = {
                    "file": str(workflow_file),
                    "syntax": syntax_result,
                    "triggers": trigger_result,
                    "actions": action_result,
                    "overall_valid": syntax_result["valid"],
                }

                workflows.append(combined_result)
                results["summary"]["total"] += 1
                if combined_result["overall_valid"]:
                    results["summary"]["valid"] += 1
                else:
                    results["summary"]["invalid"] += 1

        return results


def run_command(
    cmd: list[str], cwd: Path | None = None, timeout: int = 30
) -> dict[str, Any]:
    """Run a command and return the result."""
    result: dict[str, Any] = {
        "command": " ".join(cmd),
        "success": False,
        "stdout": "",
        "stderr": "",
        "returncode": -1,
    }

    try:
        process = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
        )
        result["success"] = process.returncode == 0
        result["stdout"] = process.stdout
        result["stderr"] = process.stderr
        result["returncode"] = process.returncode
    except subprocess.TimeoutExpired:
        result["stderr"] = f"Command timed out after {timeout} seconds"
    except Exception as e:
        result["stderr"] = str(e)

    return result


def check_required_tools() -> dict[str, Any]:
    """Check if required tools are available."""
    tools = ["git", "gh", "python3", "pip"]
    results: dict[str, Any] = {"tools": {}, "all_available": True}

    for tool in tools:
        cmd_result = run_command(["which", tool])
        available = cmd_result["success"]
        results["tools"][tool] = {
            "available": available,
            "path": cmd_result["stdout"].strip() if available else None,
        }
        if not available:
            results["all_available"] = False

    return results


def main() -> int:
    """Main function to run all validation checks."""
    print("🔍 Running GitHub Actions validation...")

    # Check required tools
    print("\n📋 Checking required tools...")
    tools_result = check_required_tools()
    for tool, info in tools_result["tools"].items():
        status = "✅" if info["available"] else "❌"
        print(f"  {status} {tool}: {info.get('path', 'Not found')}")

    if not tools_result["all_available"]:
        print("\n❌ Some required tools are missing!")
        return 1

    # Validate workflows
    print("\n🔧 Validating GitHub Actions workflows...")
    validator = GitHubActionsValidator()
    validation_results = validator.validate_all_workflows()

    print("\n📊 Summary:")
    summary = validation_results["summary"]
    print(f"  Total workflows: {summary['total']}")
    print(f"  Valid: {summary['valid']}")
    print(f"  Invalid: {summary['invalid']}")

    # Show detailed results
    for workflow in validation_results["workflows"]:
        status = "✅" if workflow["overall_valid"] else "❌"
        filename = Path(workflow["file"]).name
        print(f"\n{status} {filename}")

        if not workflow["overall_valid"]:
            syntax_errors = workflow["syntax"]["errors"]
            if syntax_errors:
                print("  Syntax errors:")
                for error in syntax_errors:
                    print(f"    - {error}")

        trigger_warnings = workflow["triggers"]["warnings"]
        if trigger_warnings:
            print("  Trigger warnings:")
            for warning in trigger_warnings:
                print(f"    - {warning}")

        action_warnings = workflow["actions"]["warnings"]
        if action_warnings:
            print("  Action warnings:")
            for warning in action_warnings:
                print(f"    - {warning}")

    # Return appropriate exit code
    if validation_results["summary"]["invalid"] > 0:
        print("\n❌ Some workflows have issues!")
        return 1
    else:
        print("\n✅ All workflows are valid!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
