#!/bin/bash

# Commune Uninstall Script
# This script removes commune and its dependencies

echo "==========================================="
echo "       Commune Uninstall Script"
echo "==========================================="
echo ""

# Get OS name
OS_NAME=$(uname)
echo "Detected OS: $OS_NAME"
echo ""

# Uninstall commune Python package
echo "=== Uninstalling Commune Python Package ==="
if pip3 list | grep -q commune; then
    echo "Found commune package installed"
    if confirm "Do you want to uninstall the commune Python package?"; then
        pip3 uninstall -y commune
        echo "✓ Commune package uninstalled"
    else
        echo "⚠ Skipping commune package uninstall"
    fi
else
    echo "⚠ Commune package not found in pip"
fi
echo ""
