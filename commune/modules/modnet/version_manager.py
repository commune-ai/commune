#!/usr/bin/env python3
"""
Independent Semantic Versioning CLI System

A reusable Python CLI tool for managing semantic versioning (major, minor, patch),
changelog updates, git tagging, and interactive version management.

Features:
- Semantic version bumping (major.minor.patch)
- Automatic changelog generation and updates
- Git tag creation with optional messages
- Interactive mode with enhanced help and menu display
- Modern Python typing with strict type checking
- Support for both automated and manual workflows

Usage:
    python version_manager.py --help
    python version_manager.py --interactive
    python version_manager.py bump major --changelog "Breaking changes"
    python version_manager.py bump minor --changelog "New features"
    python version_manager.py bump patch --changelog "Bug fixes"
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


class Colors:
    """ANSI color codes for terminal output."""

    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    MAGENTA = "\033[0;35m"
    CYAN = "\033[0;36m"
    WHITE = "\033[1;37m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    RESET = "\033[0m"


class VersionManager:
    """Manages semantic versioning, changelog updates, and git tagging."""

    def __init__(self, project_root: str | None = None):
        """Initialize the version manager with optional project root path."""
        self.project_root = Path(project_root or os.getcwd())
        self.version_file = self.project_root / "VERSION"
        self.changelog_file = self.project_root / "CHANGELOG.md"
        self.pyproject_file = self.project_root / "pyproject.toml"

    def get_current_version(self) -> tuple[int, int, int]:
        """Get the current version from VERSION file."""
        if not self.version_file.exists():
            print(
                f"{Colors.YELLOW}VERSION file not found. Creating with version 0.1.0{Colors.RESET}"
            )
            self.version_file.write_text("0.1.0\n")
            return (0, 1, 0)

        version_content = self.version_file.read_text().strip()
        match = re.match(r"^(\d+)\.(\d+)\.(\d+)$", version_content)
        if not match:
            raise ValueError(
                f"Invalid version format in VERSION file: {version_content}"
            )

        return tuple(map(int, match.groups()))  # type: ignore[return-value]

    def update_version_file(self, major: int, minor: int, patch: int) -> None:
        """Update the VERSION file with new version."""
        version_string = f"{major}.{minor}.{patch}"
        self.version_file.write_text(f"{version_string}\n")
        print(f"{Colors.GREEN}✓ Updated VERSION file to {version_string}{Colors.RESET}")

    def update_pyproject_toml(self, major: int, minor: int, patch: int) -> None:
        """Update version in pyproject.toml if it exists."""
        if not self.pyproject_file.exists():
            return

        content = self.pyproject_file.read_text()
        version_string = f"{major}.{minor}.{patch}"

        # Update version in [project] section
        updated_content = re.sub(
            r'(version\s*=\s*["\'])([^"\']+)(["\'])',
            f"\\g<1>{version_string}\\g<3>",
            content,
        )

        if updated_content != content:
            self.pyproject_file.write_text(updated_content)
            print(
                f"{Colors.GREEN}✓ Updated pyproject.toml version to {version_string}{Colors.RESET}"
            )

    def bump_version(
        self, bump_type: str, changelog_entry: str | None = None
    ) -> tuple[int, int, int]:
        """Bump version based on type (major, minor, patch)."""
        major, minor, patch = self.get_current_version()

        if bump_type == "major":
            major += 1
            minor = 0
            patch = 0
        elif bump_type == "minor":
            minor += 1
            patch = 0
        elif bump_type == "patch":
            patch += 1
        else:
            raise ValueError(
                f"Invalid bump type: {bump_type}. Must be 'major', 'minor', or 'patch'"
            )

        self.update_version_file(major, minor, patch)
        self.update_pyproject_toml(major, minor, patch)

        if changelog_entry:
            self.update_changelog(major, minor, patch, changelog_entry)

        return (major, minor, patch)

    def update_changelog(self, major: int, minor: int, patch: int, entry: str) -> None:
        """Update CHANGELOG.md with new version entry."""
        version_string = f"{major}.{minor}.{patch}"
        date_string = datetime.now().strftime("%Y-%m-%d")

        new_entry = f"""## [{version_string}] - {date_string}

{entry}

"""

        if self.changelog_file.exists():
            existing_content = self.changelog_file.read_text()
            # Insert new entry after the first line (usually "# Changelog")
            lines = existing_content.split("\n")
            if lines and lines[0].startswith("# "):
                # Insert after header
                updated_content = "\n".join([lines[0], "", new_entry] + lines[1:])
            else:
                updated_content = new_entry + existing_content
        else:
            updated_content = f"""# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

{new_entry}"""

        self.changelog_file.write_text(updated_content)
        print(
            f"{Colors.GREEN}✓ Updated CHANGELOG.md with version {version_string}{Colors.RESET}"
        )

    def create_git_tag(
        self, major: int, minor: int, patch: int, message: str | None = None
    ) -> None:
        """Create a git tag for the version."""
        version_string = f"v{major}.{minor}.{patch}"

        try:
            if message:
                subprocess.run(
                    ["git", "tag", "-a", version_string, "-m", message],
                    check=True,
                    cwd=self.project_root,
                )
            else:
                subprocess.run(
                    ["git", "tag", version_string], check=True, cwd=self.project_root
                )
            print(f"{Colors.GREEN}✓ Created git tag {version_string}{Colors.RESET}")
        except subprocess.CalledProcessError as e:
            print(f"{Colors.RED}✗ Failed to create git tag: {e}{Colors.RESET}")

    def display_current_version(self) -> None:
        """Display the current version information."""
        try:
            major, minor, patch = self.get_current_version()
            print(f"\n{Colors.BOLD}Current Version Information:{Colors.RESET}")
            print(f"  Version: {Colors.GREEN}{major}.{minor}.{patch}{Colors.RESET}")
            print(f"  Project: {Colors.CYAN}{self.project_root.name}{Colors.RESET}")
            print(f"  Path: {Colors.BLUE}{self.project_root}{Colors.RESET}")
        except Exception as e:
            print(f"{Colors.RED}✗ Error reading version: {e}{Colors.RESET}")

    def interactive_mode(self) -> None:
        """Run interactive version management mode."""
        while True:
            self.display_interactive_menu()
            choice = input(
                f"\n{Colors.BOLD}Select an option (1-7): {Colors.RESET}"
            ).strip()

            if choice == "1":
                self.display_current_version()
            elif choice == "2":
                self._interactive_bump("major")
            elif choice == "3":
                self._interactive_bump("minor")
            elif choice == "4":
                self._interactive_bump("patch")
            elif choice == "5":
                self._interactive_changelog()
            elif choice == "6":
                self._interactive_git_tag()
            elif choice == "7":
                print(f"{Colors.GREEN}Goodbye!{Colors.RESET}")
                break
            else:
                print(f"{Colors.RED}Invalid option. Please select 1-7.{Colors.RESET}")

            input(f"\n{Colors.YELLOW}Press Enter to continue...{Colors.RESET}")

    def display_interactive_menu(self) -> None:
        """Display the interactive menu."""
        print(f"\n{Colors.BOLD}{'='*60}{Colors.RESET}")
        print(
            f"{Colors.BOLD}{Colors.CYAN}    Semantic Version Manager - Interactive Mode{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")

        try:
            major, minor, patch = self.get_current_version()
            print(
                f"{Colors.WHITE}Current Version: {Colors.GREEN}{major}.{minor}.{patch}{Colors.RESET}"
            )
        except Exception:
            print(f"{Colors.WHITE}Current Version: {Colors.RED}Unknown{Colors.RESET}")

        print(
            f"{Colors.WHITE}Project: {Colors.CYAN}{self.project_root.name}{Colors.RESET}"
        )
        print(f"{Colors.BOLD}{'='*60}{Colors.RESET}")

        print(f"\n{Colors.BOLD}Available Options:{Colors.RESET}")
        print(f"  {Colors.GREEN}1.{Colors.RESET} Display current version information")
        print(
            f"  {Colors.GREEN}2.{Colors.RESET} Bump {Colors.RED}MAJOR{Colors.RESET} version (breaking changes)"
        )
        print(
            f"  {Colors.GREEN}3.{Colors.RESET} Bump {Colors.YELLOW}MINOR{Colors.RESET} version (new features)"
        )
        print(
            f"  {Colors.GREEN}4.{Colors.RESET} Bump {Colors.BLUE}PATCH{Colors.RESET} version (bug fixes)"
        )
        print(f"  {Colors.GREEN}5.{Colors.RESET} Update changelog manually")
        print(f"  {Colors.GREEN}6.{Colors.RESET} Create git tag")
        print(f"  {Colors.GREEN}7.{Colors.RESET} Exit")

    def _interactive_bump(self, bump_type: str) -> None:
        """Handle interactive version bumping."""
        try:
            current = self.get_current_version()
            print(
                f"\n{Colors.BOLD}Current version: {Colors.GREEN}{'.'.join(map(str, current))}{Colors.RESET}"
            )

            # Preview what the new version will be
            major, minor, patch = current
            if bump_type == "major":
                major += 1
                minor = 0
                patch = 0
            elif bump_type == "minor":
                minor += 1
                patch = 0
            elif bump_type == "patch":
                patch += 1

            new_version = f"{major}.{minor}.{patch}"
            print(
                f"{Colors.BOLD}New version will be: {Colors.CYAN}{new_version}{Colors.RESET}"
            )

            # Ask for changelog entry
            changelog_entry = input(
                f"\n{Colors.YELLOW}Enter changelog entry (optional): {Colors.RESET}"
            ).strip()

            # Confirm the action
            confirm = (
                input(
                    f"\n{Colors.BOLD}Proceed with {bump_type.upper()} version bump? (y/N): {Colors.RESET}"
                )
                .strip()
                .lower()
            )

            if confirm in ["y", "yes"]:
                new_version_tuple = self.bump_version(
                    bump_type, changelog_entry if changelog_entry else None
                )
                print(
                    f"\n{Colors.GREEN}✓ Successfully bumped to version {'.'.join(map(str, new_version_tuple))}{Colors.RESET}"
                )

                # Ask about git tag
                tag_confirm = (
                    input(
                        f"{Colors.YELLOW}Create git tag for this version? (y/N): {Colors.RESET}"
                    )
                    .strip()
                    .lower()
                )
                if tag_confirm in ["y", "yes"]:
                    tag_message = input(
                        f"{Colors.YELLOW}Enter tag message (optional): {Colors.RESET}"
                    ).strip()
                    self.create_git_tag(
                        *new_version_tuple, tag_message if tag_message else None
                    )
            else:
                print(f"{Colors.YELLOW}Version bump cancelled.{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}✗ Error during version bump: {e}{Colors.RESET}")

    def _interactive_changelog(self) -> None:
        """Handle interactive changelog updates."""
        try:
            major, minor, patch = self.get_current_version()
            print(
                f"\n{Colors.BOLD}Current version: {Colors.GREEN}{major}.{minor}.{patch}{Colors.RESET}"
            )

            entry = input(
                f"{Colors.YELLOW}Enter changelog entry: {Colors.RESET}"
            ).strip()
            if entry:
                self.update_changelog(major, minor, patch, entry)
                print(f"{Colors.GREEN}✓ Changelog updated successfully{Colors.RESET}")
            else:
                print(f"{Colors.YELLOW}No changelog entry provided.{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}✗ Error updating changelog: {e}{Colors.RESET}")

    def _interactive_git_tag(self) -> None:
        """Handle interactive git tag creation."""
        try:
            major, minor, patch = self.get_current_version()
            version_string = f"v{major}.{minor}.{patch}"

            print(
                f"\n{Colors.BOLD}Current version: {Colors.GREEN}{major}.{minor}.{patch}{Colors.RESET}"
            )
            print(
                f"{Colors.BOLD}Tag will be: {Colors.CYAN}{version_string}{Colors.RESET}"
            )

            message = input(
                f"{Colors.YELLOW}Enter tag message (optional): {Colors.RESET}"
            ).strip()

            confirm = (
                input(
                    f"{Colors.BOLD}Create git tag {version_string}? (y/N): {Colors.RESET}"
                )
                .strip()
                .lower()
            )

            if confirm in ["y", "yes"]:
                self.create_git_tag(major, minor, patch, message if message else None)
            else:
                print(f"{Colors.YELLOW}Git tag creation cancelled.{Colors.RESET}")

        except Exception as e:
            print(f"{Colors.RED}✗ Error creating git tag: {e}{Colors.RESET}")


def main() -> None:
    """Main entry point for the version manager CLI."""
    parser = argparse.ArgumentParser(
        description="Independent Semantic Versioning CLI System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --interactive                    # Run in interactive mode
  %(prog)s bump major --changelog "Breaking changes"
  %(prog)s bump minor --changelog "New features" --tag
  %(prog)s bump patch --changelog "Bug fixes" --tag --tag-message "Hotfix release"
  %(prog)s current                          # Display current version
  %(prog)s changelog "Manual changelog entry"
  %(prog)s tag --message "Release version"

Version Types:
  major    # Breaking changes (1.0.0 -> 2.0.0)
  minor    # New features (1.0.0 -> 1.1.0)
  patch    # Bug fixes (1.0.0 -> 1.0.1)
        """,
    )

    parser.add_argument(
        "--project-root",
        type=str,
        help="Project root directory (default: current directory)",
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with enhanced menu",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Bump command
    bump_parser = subparsers.add_parser("bump", help="Bump version")
    bump_parser.add_argument(
        "type", choices=["major", "minor", "patch"], help="Version bump type"
    )
    bump_parser.add_argument(
        "--changelog", type=str, help="Changelog entry for this version"
    )
    bump_parser.add_argument(
        "--tag", action="store_true", help="Create git tag after version bump"
    )
    bump_parser.add_argument("--tag-message", type=str, help="Message for git tag")

    # Current command
    subparsers.add_parser("current", help="Display current version")

    # Changelog command
    changelog_parser = subparsers.add_parser("changelog", help="Update changelog")
    changelog_parser.add_argument("entry", type=str, help="Changelog entry")

    # Tag command
    tag_parser = subparsers.add_parser("tag", help="Create git tag for current version")
    tag_parser.add_argument("--message", type=str, help="Tag message")

    args = parser.parse_args()

    try:
        version_manager = VersionManager(args.project_root)

        if args.interactive:
            version_manager.interactive_mode()
        elif args.command == "bump":
            new_version = version_manager.bump_version(args.type, args.changelog)
            if args.tag:
                version_manager.create_git_tag(*new_version, args.tag_message)
        elif args.command == "current":
            version_manager.display_current_version()
        elif args.command == "changelog":
            major, minor, patch = version_manager.get_current_version()
            version_manager.update_changelog(major, minor, patch, args.entry)
        elif args.command == "tag":
            major, minor, patch = version_manager.get_current_version()
            version_manager.create_git_tag(major, minor, patch, args.message)
        else:
            parser.print_help()

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Operation cancelled by user.{Colors.RESET}")
        sys.exit(1)
    except Exception as e:
        print(f"{Colors.RED}✗ Error: {e}{Colors.RESET}")
        sys.exit(1)


if __name__ == "__main__":
    main()
