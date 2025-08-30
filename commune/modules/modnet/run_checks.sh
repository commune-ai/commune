#!/bin/bash

# Development Tools CLI for mod-net-modules
# Provides interactive selection of linters, formatters, and checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to print colored output
print_header() {
    printf "${BLUE}=== %s ===${NC}\n" "$1"
}

print_success() {
    printf "${GREEN}✓ %s${NC}\n" "$1"
}

print_error() {
    printf "${RED}✗ %s${NC}\n" "$1"
}

print_warning() {
    printf "${YELLOW}⚠ %s${NC}\n" "$1"
}

print_info() {
    printf "${CYAN}ℹ %s${NC}\n" "$1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run a command with error handling
run_command() {
    local cmd="$1"
    local desc="$2"

    print_info "Running: $desc"
    echo "Command: $cmd"

    if eval "$cmd"; then
        print_success "$desc completed successfully"
        return 0
    else
        print_error "$desc failed"
        return 1
    fi
}

# Submodule management
run_submodule_update() {
    if [ -f ".gitmodules" ]; then
        print_info "Checking for submodule updates..."

        # Check if any submodules need updating
        if git submodule status | grep -q '^[+-]'; then
            print_info "Submodules need updating, syncing..."
            run_command "git submodule update --init --recursive" "Submodule sync"

            # Check if submodule pointers need updating in main repo
            if ! git diff --quiet --cached; then
                print_warning "Submodule pointers may need updating. Consider running 'git add .' and committing changes."
            fi
        else
            print_success "Submodules are up to date"
        fi
    else
        print_info "No submodules found, skipping submodule update"
    fi
}

# Rust tools
run_rust_fmt() {
    if command_exists cargo; then
        run_command "cd '$PROJECT_ROOT' && cargo fmt --all" "Rust formatting (cargo fmt)"
    else
        print_error "cargo not found. Please install Rust."
        return 1
    fi
}

run_rust_clippy() {
    if command_exists cargo; then
        run_command "cd '$PROJECT_ROOT' && cargo clippy --all-targets --all-features -- -D warnings" "Rust linting (cargo clippy)"
    else
        print_error "cargo not found. Please install Rust."
        return 1
    fi
}

run_rust_check() {
    if command_exists cargo; then
        run_command "cd '$PROJECT_ROOT' && cargo check --all-targets --all-features" "Rust compilation check (cargo check)"
    else
        print_error "cargo not found. Please install Rust."
        return 1
    fi
}

run_rust_test() {
    if command_exists cargo; then
        run_command "cd '$PROJECT_ROOT' && cargo test --all" "Rust tests (cargo test)"
    else
        print_error "cargo not found. Please install Rust."
        return 1
    fi
}

run_rust_doc() {
    if command_exists cargo; then
        run_command "cd '$PROJECT_ROOT' && cargo doc --no-deps --document-private-items" "Rust documentation (cargo doc)"
    else
        print_error "cargo not found. Please install Rust."
        return 1
    fi
}

# Python tools
run_python_black() {
    if command_exists black; then
        run_command "cd '$PROJECT_ROOT' && black ." "Python formatting (black)"
    elif command_exists uv; then
        run_command "cd '$PROJECT_ROOT' && uv run black ." "Python formatting (black via uv)"
    else
        print_error "black not found. Please install black or uv."
        return 1
    fi
}

run_python_isort() {
    if command_exists isort; then
        run_command "cd '$PROJECT_ROOT' && isort . --profile black" "Python import sorting (isort)"
    elif command_exists uv; then
        run_command "cd '$PROJECT_ROOT' && uv run isort . --profile black" "Python import sorting (isort via uv)"
    else
        print_error "isort not found. Please install isort or uv."
        return 1
    fi
}

run_python_ruff() {
    if command_exists ruff; then
        run_command "cd '$PROJECT_ROOT' && ruff check ." "Python linting (ruff)"
    elif command_exists uv; then
        run_command "cd '$PROJECT_ROOT' && uv run ruff check ." "Python linting (ruff via uv)"
    else
        print_error "ruff not found. Please install ruff or uv."
        return 1
    fi
}

run_python_ruff_fix() {
    if command_exists ruff; then
        run_command "cd '$PROJECT_ROOT' && ruff check . --fix --unsafe-fixes" "Python auto-fix (ruff --fix --unsafe-fixes)"
    elif command_exists uv; then
        run_command "cd '$PROJECT_ROOT' && uv run ruff check . --fix --unsafe-fixes" "Python auto-fix (ruff --fix --unsafe-fixes via uv)"
    else
        print_error "ruff not found. Please install ruff or uv."
        return 1
    fi
}

run_python_mypy() {
    if command_exists mypy; then
        run_command "cd '$PROJECT_ROOT' && mypy ." "Python type checking (mypy)"
    elif command_exists uv; then
        run_command "cd '$PROJECT_ROOT' && uv run mypy ." "Python type checking (mypy via uv)"
    else
        print_error "mypy not found. Please install mypy or uv."
        return 1
    fi
}

run_python_pytest() {
    if command_exists pytest; then
        run_command "cd '$PROJECT_ROOT' && pytest" "Python tests (pytest)"
    elif command_exists uv; then
        run_command "cd '$PROJECT_ROOT' && uv run pytest" "Python tests (pytest via uv)"
    else
        print_error "pytest not found. Please install pytest or uv."
        return 1
    fi
}

# Combined runners
run_all_rust() {
    print_header "Running All Rust Tools"
    local failed=0

    run_rust_fmt || ((failed++))
    run_rust_clippy || ((failed++))
    run_rust_check || ((failed++))
    run_rust_test || ((failed++))

    if [ $failed -eq 0 ]; then
        print_success "All Rust tools completed successfully"
    else
        print_error "$failed Rust tool(s) failed"
        return 1
    fi
}

run_all_python() {
    print_header "Running All Python Tools"
    local failed=0

    run_python_black || ((failed++))
    run_python_isort || ((failed++))
    run_python_ruff || ((failed++))
    run_python_mypy || ((failed++))
    run_python_pytest || ((failed++))

    if [ $failed -eq 0 ]; then
        print_success "All Python tools completed successfully"
    else
        print_error "$failed Python tool(s) failed"
        return 1
    fi
}

run_all_formatters() {
    print_header "Running All Formatters"
    local failed=0

    run_submodule_update || ((failed++))
    run_rust_fmt || ((failed++))
    run_python_black || ((failed++))
    run_python_isort || ((failed++))
    run_python_ruff_fix || ((failed++))

    if [ $failed -eq 0 ]; then
        print_success "All formatters completed successfully"
    else
        print_error "$failed formatter(s) failed"
        return 1
    fi
}

run_all_linters() {
    print_header "Running All Linters"
    local failed=0

    run_rust_clippy || ((failed++))
    run_python_ruff || ((failed++))
    run_python_mypy || ((failed++))

    if [ $failed -eq 0 ]; then
        print_success "All linters completed successfully"
    else
        print_error "$failed linter(s) failed"
        return 1
    fi
}

run_all_tests() {
    print_header "Running All Tests"
    local failed=0

    run_rust_test || ((failed++))
    run_python_pytest || ((failed++))

    if [ $failed -eq 0 ]; then
        print_success "All tests completed successfully"
    else
        print_error "$failed test suite(s) failed"
        return 1
    fi
}

run_all_checks() {
    print_header "Running All Checks"
    local failed=0

    run_rust_check || ((failed++))
    run_rust_clippy || ((failed++))
    run_python_ruff || ((failed++))
    run_python_mypy || ((failed++))

    if [ $failed -eq 0 ]; then
        print_success "All checks completed successfully"
    else
        print_error "$failed check(s) failed"
        return 1
    fi
}

run_everything() {
    print_header "Running Everything"
    local failed=0

    run_all_formatters || ((failed++))
    run_all_checks || ((failed++))
    run_all_tests || ((failed++))

    if [ $failed -eq 0 ]; then
        print_success "All tools completed successfully! 🎉"
    else
        print_error "$failed category(ies) failed"
        return 1
    fi
}

# Interactive menu
show_menu() {
    printf "\n"
    print_header "Development Tools CLI - mod-net-modules"
    printf "\n"
    printf "Select tools to run:\n"
    printf "\n"
    printf "${PURPLE}Individual Tools:${NC}\n"
    printf "  1)  Rust Format (cargo fmt)\n"
    printf "  2)  Rust Clippy (cargo clippy)\n"
    printf "  3)  Rust Check (cargo check)\n"
    printf "  4)  Rust Test (cargo test)\n"
    printf "  5)  Rust Doc (cargo doc)\n"
    printf "  6)  Python Black (black)\n"
    printf "  7)  Python Isort (isort)\n"
    printf "  8)  Python Ruff (ruff)\n"
    printf "  9)  Python MyPy (mypy)\n"
    printf "  10) Python Pytest (pytest)\n"
    printf "  11) Submodule Update (sync submodules)\n"
    printf "\n"
    printf "${YELLOW}By Language:${NC}\n"
    printf "  12) All Rust Tools\n"
    printf "  13) All Python Tools\n"
    printf "\n"
    printf "${CYAN}By Category:${NC}\n"
    printf "  14) All Formatters (fmt, black, isort, submodule sync)\n"
    printf "  15) All Linters (clippy, ruff, mypy)\n"
    printf "  16) All Tests (cargo test, pytest)\n"
    printf "  17) All Checks (check, clippy, ruff, mypy)\n"
    printf "\n"
    printf "${GREEN}Combined:${NC}\n"
    printf "  18) Everything (formatters + checks + tests)\n"
    printf "\n"
    printf "  0)  Exit\n"
    printf "\n"
}

# Main interactive loop
interactive_mode() {
    while true; do
        show_menu
        read -p "Enter your choice (0-17): " choice
        echo

        case $choice in
            1) run_rust_fmt ;;
            2) run_rust_clippy ;;
            3) run_rust_check ;;
            4) run_rust_test ;;
            5) run_rust_doc ;;
            6) run_python_black ;;
            7) run_python_isort ;;
            8) run_python_ruff ;;
            9) run_python_mypy ;;
            10) run_python_pytest ;;
            11) run_submodule_update ;;
            12) run_all_rust ;;
            13) run_all_python ;;
            14) run_all_formatters ;;
            15) run_all_linters ;;
            16) run_all_tests ;;
            17) run_all_checks ;;
            18) run_everything ;;
            0)
                print_info "Goodbye!"
                exit 0
                ;;
            *)
                print_error "Invalid choice. Please enter a number between 0-18."
                ;;
        esac

        printf "\n"
        read -p "Press Enter to continue..."
    done
}

# Command line argument handling
if [ $# -eq 0 ]; then
    # No arguments, run interactive mode
    interactive_mode
else
    # Handle command line arguments
    case "$1" in
        "rust-fmt"|"fmt") run_rust_fmt ;;
        "rust-clippy"|"clippy") run_rust_clippy ;;
        "rust-check"|"check") run_rust_check ;;
        "rust-test"|"test-rust") run_rust_test ;;
        "rust-doc"|"doc") run_rust_doc ;;
        "python-black"|"black") run_python_black ;;
        "python-isort"|"isort") run_python_isort ;;
        "python-ruff"|"ruff") run_python_ruff ;;
        "python-mypy"|"mypy") run_python_mypy ;;
        "python-pytest"|"pytest"|"test-python") run_python_pytest ;;
        "submodule"|"submodules"|"sync") run_submodule_update ;;
        "rust"|"all-rust") run_all_rust ;;
        "python"|"all-python") run_all_python ;;
        "formatters"|"format") run_all_formatters ;;
        "linters"|"lint") run_all_linters ;;
        "tests"|"test") run_all_tests ;;
        "checks") run_all_checks ;;
        "all"|"everything") run_everything ;;
        "help"|"--help"|"h")
            printf "Usage: %s [command]\n" "$0"
            printf "\n"
            printf "Commands:\n"
            printf "  Individual tools: rust-fmt, rust-clippy, rust-check, rust-test, rust-doc\n"
            printf "                   python-black, python-isort, python-ruff, python-mypy, python-pytest\n"
            printf "                   submodule (sync submodules)\n"
            printf "  By language:     rust, python\n"
            printf "  By category:     formatters, linters, tests, checks\n"
            printf "  Combined:        all, everything\n"
            printf "\n"
            printf "If no command is provided, interactive mode will start.\n"
            ;;
        *)
            print_error "Unknown command: $1"
            print_info "Run '$0 help' for usage information."
            exit 1
            ;;
    esac
fi
