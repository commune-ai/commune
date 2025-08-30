#!/bin/bash
# Script to run comprehensive project validation including linting and type checking

set -e

echo "🔧 Installing dependencies for test hooks..."
uv add pyyaml

# Install Python linting tools
echo "🐍 Installing Python linting tools..."
uv add black isort flake8 mypy pylint --group dev

# Check if Rust tools are available and install if needed
if command -v cargo &> /dev/null; then
    echo "🦀 Installing Rust linting tools..."
    rustup component add rustfmt clippy 2>/dev/null || echo "Rust components already installed"
else
    echo "⚠️  Cargo not found - Rust linting will be skipped"
fi

echo ""
echo "🚀 Running comprehensive project validation..."
echo "=================================================="

# Run comprehensive validation by default
uv run python test_hooks.py

echo ""
echo "📊 Generating detailed JSON report..."
uv run python test_hooks.py --json > validation_report.json

echo "✅ Validation complete!"
echo "📄 Detailed report saved to validation_report.json"
echo ""
echo "💡 You can also run specific checks:"
echo "   - Workflows only: uv run python test_hooks.py --workflows-only"
echo "   - Python only:   uv run python test_hooks.py --python-only"
echo "   - Rust only:     uv run python test_hooks.py --rust-only"
echo "   - Exit on error: uv run python test_hooks.py --exit-on-error"
