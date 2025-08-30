#!/usr/bin/env python3
"""
Codebase Audit Script

This script performs automated code quality audits to identify:
- Mock implementations (not for tests)
- Simplified/placeholder logic
- TODOs and incomplete code
- Hard-coded values
- Other placeholders

Usage:
    python audit_codebase.py [directory]

Example:
    # Basic audit
    uv run python audit_codebase.py

    # Audit with limited results per category
    uv run python audit_codebase.py --max-per-category 5

    # Summary only
    uv run python audit_codebase.py --summary-only

    # Export to JSON
    uv run python audit_codebase.py --json audit_results.json

    # Audit specific directory
    uv run python audit_codebase.py /path/to/project
"""

import argparse
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AuditResult:
    """Represents a single audit finding."""

    file_path: str
    line_number: int
    line_content: str
    category: str
    severity: str
    description: str


class CodebaseAuditor:
    """Performs automated codebase audits."""

    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir).resolve()
        self.results: list[AuditResult] = []

        # Audit patterns
        self.patterns = {
            "todo": {
                "regex": r"TODO|FIXME|XXX|HACK",
                "severity": "medium",
                "description": "TODO/FIXME comment found",
                "exclude_patterns": [
                    r"audit_codebase\.py$",
                    r"fix_hardcoded_values\.py$",
                ],  # Skip audit tools
            },
            "mock_non_test": {
                "regex": r"\bmock\b|\bMock\b|\bMOCK\b",  # Word boundaries to avoid false positives
                "severity": "high",
                "description": "Mock implementation in non-test code",
                "exclude_patterns": [
                    r"test_.*\.py$",
                    r".*_test\.py$",
                    r"tests/.*\.py$",
                    r"mock_.*\.py$",
                    r"audit_codebase\.py$",
                    r".*example.*\.py$",
                ],
            },
            "placeholder": {
                "regex": r"\bplaceholder\b|\bPLACEHOLDER\b|\bstub\b|\bSTUB\b",  # Word boundaries
                "severity": "high",
                "description": "Placeholder implementation found",
                "exclude_patterns": [r"audit_codebase\.py$", r".*example.*\.py$"],
            },
            "hardcoded_urls": {
                "regex": r"localhost|127\.0\.0\.1|0\.0\.0\.0",
                "severity": "medium",
                "description": "Hard-coded localhost/IP address",
                "exclude_patterns": [],  # Will be added dynamically in should_exclude_for_pattern
            },
            "hardcoded_ports": {
                "regex": r":\d{4,5}(?![0-9])",
                "severity": "medium",
                "description": "Hard-coded port number",
                "exclude_patterns": [],  # Will be added dynamically
            },
            "hardcoded_keys": {
                "regex": r"//Alice|//Bob|//Charlie|0x[a-fA-F0-9]{32,}",
                "severity": "high",
                "description": "Hard-coded cryptographic key/seed",
                "exclude_patterns": [],  # Will be added dynamically
            },
            "simplified_logic": {
                "regex": r"# (simplified|simple|basic|minimal|temporary)",
                "severity": "medium",
                "description": "Simplified/temporary implementation",
                "exclude_patterns": [r"audit_codebase\.py$", r"test_.*\.py$"],
            },
            "empty_except": {
                "regex": r"except.*:\s*pass\s*$",
                "severity": "high",
                "description": "Empty exception handler",
            },
            "print_debug": {
                "regex": r'print\s*\(\s*["\'].*[Dd]ebug.*["\']',
                "severity": "low",
                "description": "Debug print statement",
                "exclude_patterns": [
                    r"debug_.*\.py$",
                    r"test_.*\.py$",
                ],  # Skip debug scripts and tests
            },
        }

    def should_exclude_file(self, file_path: Path) -> bool:
        """Check if file should be excluded from audit."""
        exclude_patterns = [
            # Build and cache directories
            r"\.git/",
            r"__pycache__/",
            r"\.pytest_cache/",
            r"\.venv/",
            r"venv/",
            r"node_modules/",
            r"target/",
            r"build/",
            r"dist/",
            r"\.egg-info/",
            # Binary and lock files
            r"uv\.lock$",
            r"\.pyc$",
            r"\.pyo$",
            r"\.so$",
            r"\.dylib$",
            r"\.dll$",
            # Audit and development tools (acceptable to have patterns)
            r"audit_codebase\.py$",
            r"fix_hardcoded_values\.py$",
            r"config\.py$",  # Configuration file itself
            r"substrate_config\.py$",  # Legacy config file
            # Test files (acceptable to use localhost/hardcoded values)
            r"test_.*\.py$",
            r".*_test\.py$",
            r"tests/.*\.py$",
            r"test_hooks\.py$",
            # Debug and development scripts
            r"debug_.*\.py$",
            r".*_debug\.py$",
            # Documentation and examples
            r"example.*\.py$",
            r".*example\.py$",
        ]

        file_str = str(file_path)
        return any(re.search(pattern, file_str) for pattern in exclude_patterns)

    def should_exclude_for_pattern(self, file_path: Path, pattern_name: str) -> bool:
        """Check if file should be excluded for specific pattern."""
        if pattern_name not in self.patterns:
            return False

        pattern_config = self.patterns[pattern_name]
        exclude_patterns = pattern_config.get("exclude_patterns", [])  # type: ignore

        # Additional pattern-specific exclusions
        if pattern_name == "hardcoded_urls" or pattern_name == "hardcoded_ports":
            # Skip test files and configuration files for URL/port checks
            additional_excludes = [
                r"test_.*\.py$",
                r".*_test\.py$",
                r"tests/.*\.py$",
                r"config\.py$",
                r"substrate_config\.py$",
                r".*example.*\.py$",
            ]
            exclude_patterns.extend(additional_excludes)

        if pattern_name == "hardcoded_keys":
            # Skip test files for key checks (test keys are acceptable)
            additional_excludes = [
                r"test_.*\.py$",
                r".*_test\.py$",
                r"tests/.*\.py$",
                r"debug_.*\.py$",
                r".*example.*\.py$",
            ]
            exclude_patterns.extend(additional_excludes)

        file_str = str(file_path)
        return any(re.search(pattern, file_str) for pattern in exclude_patterns)

    def search_pattern_in_file(
        self, file_path: Path, pattern_name: str
    ) -> list[AuditResult]:
        """Search for a specific pattern in a file."""
        results = []

        if self.should_exclude_for_pattern(file_path, pattern_name):
            return results

        pattern_config = self.patterns[pattern_name]
        regex = pattern_config["regex"]
        severity = pattern_config["severity"]
        description = pattern_config["description"]

        try:
            with open(file_path, encoding="utf-8", errors="ignore") as f:
                for line_num, line in enumerate(f, 1):
                    line = line.rstrip()
                    if re.search(regex, line, re.IGNORECASE):
                        relative_path = str(file_path.relative_to(self.root_dir))
                        results.append(
                            AuditResult(
                                file_path=relative_path,
                                line_number=line_num,
                                line_content=line,
                                category=pattern_name,
                                severity=severity,
                                description=description,
                            )
                        )
        except (UnicodeDecodeError, PermissionError):
            # Skip files that can't be read
            pass

        return results

    def find_python_files(self) -> list[Path]:
        """Find all Python files in the directory."""
        python_files = []

        for file_path in self.root_dir.rglob("*.py"):
            if not self.should_exclude_file(file_path):
                python_files.append(file_path)

        return python_files

    def audit_file(self, file_path: Path) -> list[AuditResult]:
        """Audit a single file for all patterns."""
        results = []

        for pattern_name in self.patterns:
            results.extend(self.search_pattern_in_file(file_path, pattern_name))

        return results

    def run_audit(self) -> dict[str, list[AuditResult]]:
        """Run the complete audit."""
        print(f"🔍 Starting codebase audit in: {self.root_dir}")

        python_files = self.find_python_files()
        print(f"📁 Found {len(python_files)} Python files to audit")

        all_results = []
        for file_path in python_files:
            file_results = self.audit_file(file_path)
            all_results.extend(file_results)

        # Group results by category
        grouped_results = defaultdict(list)
        for result in all_results:
            grouped_results[result.category].append(result)

        return dict(grouped_results)

    def print_summary(self, results: dict[str, list[AuditResult]]):
        """Print audit summary."""
        total_issues = sum(len(issues) for issues in results.values())

        print("\n📊 Audit Summary")
        print("=" * 50)
        print(f"Total issues found: {total_issues}")

        if total_issues == 0:
            print("✅ No issues found!")
            return

        # Count by severity
        severity_counts: dict[str, int] = {}
        for issues in results.values():
            for issue in issues:
                severity = issue.severity
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

        print(f"🔴 High severity: {severity_counts.get('high', 0)}")
        print(f"🟡 Medium severity: {severity_counts.get('medium', 0)}")
        print(f"🟢 Low severity: {severity_counts.get('low', 0)}")

        print("\n📋 Issues by Category:")
        results: list[str] = []
        for category, issues in sorted(results.items()):
            results.append(f"  {category}: {len(issues)} issues")

    def print_detailed_results(
        self, results: dict[str, list[AuditResult]], max_per_category: int = 10
    ):
        """Print detailed audit results."""
        for category, issues in sorted(results.items()):
            if not issues:
                continue

            print(f"\n🔍 {category.upper()} ({len(issues)} issues)")
            print("-" * 60)

            # Show up to max_per_category issues
            for _i, issue in enumerate(issues[:max_per_category]):
                severity_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(
                    issue.severity, "⚪"
                )
                print(f"{severity_icon} {issue.file_path}:{issue.line_number}")
                print(f"   {issue.description}")
                print(f"   {issue.line_content.strip()}")
                print()

            if len(issues) > max_per_category:
                print(f"   ... and {len(issues) - max_per_category} more issues")
                print()

    def export_to_json(self, results: dict[str, list[AuditResult]], output_file: str):
        """Export results to JSON file."""
        import json

        # Convert results to JSON-serializable format
        json_results = {}
        for category, issues in results.items():
            json_results[category] = [
                {
                    "file_path": issue.file_path,
                    "line_number": issue.line_number,
                    "line_content": issue.line_content,
                    "severity": issue.severity,
                    "description": issue.description,
                }
                for issue in issues
            ]

        with open(output_file, "w") as f:
            json.dump(json_results, f, indent=2)

        print(f"📄 Results exported to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Audit codebase for quality issues")
    parser.add_argument(
        "directory",
        nargs="?",
        default=".",
        help="Directory to audit (default: current directory)",
    )
    parser.add_argument("--json", "-j", help="Export results to JSON file")
    parser.add_argument(
        "--max-per-category",
        "-m",
        type=int,
        default=10,
        help="Maximum issues to show per category (default: 10)",
    )
    parser.add_argument(
        "--summary-only",
        "-s",
        action="store_true",
        help="Show only summary, not detailed results",
    )

    args = parser.parse_args()

    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"❌ Error: Directory '{args.directory}' does not exist")
        return 1

    # Run audit
    auditor = CodebaseAuditor(args.directory)
    results = auditor.run_audit()

    # Print results
    auditor.print_summary(results)

    if not args.summary_only:
        auditor.print_detailed_results(results, args.max_per_category)

    # Export to JSON if requested
    if args.json:
        auditor.export_to_json(results, args.json)

    # Return exit code based on high severity issues
    high_severity_count = sum(
        len([issue for issue in issues if issue.severity == "high"])
        for issues in results.values()
    )

    return 1 if high_severity_count > 0 else 0


if __name__ == "__main__":
    exit(main())
