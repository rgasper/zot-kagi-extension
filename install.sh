#!/usr/bin/env bash
# install.sh — install the kagi extension into the global zot extensions dir.
#
# `zot ext install` does a recursive copy, which would include the .venv with
# broken rpaths.  This script removes any stale install, strips the .venv
# from the source copy, installs, then removes it from the destination too so
# uv rebuilds a clean one on first launch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXT_NAME="zot-kagi-extension"

echo "Removing any existing install..."
zot ext remove kagi 2>/dev/null || true
rm -rf "${HOME}/Library/Application Support/zot/extensions/${EXT_NAME}" 2>/dev/null || true

echo "Installing..."
zot ext install "${SCRIPT_DIR}"

echo "Removing copied .venv (uv will rebuild it on first launch)..."
rm -rf "${HOME}/Library/Application Support/zot/extensions/${EXT_NAME}/.venv"
rm -rf "${HOME}/Library/Application Support/zot/extensions/${EXT_NAME}/__pycache__"

echo "Done. Run 'zot' (or '/reload-ext' if already running) to activate."
echo "Make sure KAGI_API_KEY is set in your environment."
