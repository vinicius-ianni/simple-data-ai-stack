#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# ZVEC Healthcare Intelligence Demo
#
# Uses `uv run` with PEP 723 inline script metadata.
# uv will automatically:
#   1. Pin Python 3.12 (zvec requires 3.10–3.12; system may have 3.13+)
#   2. Create an isolated virtualenv
#   3. Install all dependencies (zvec, polars, fastembed, rich, orjson, numpy)
#   4. Run the script
#
# First run downloads the BGE-small-en-v1.5 ONNX model (~23 MB).
# Subsequent runs are fully offline.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "  ╔══════════════════════════════════════════════════╗"
echo "  ║   ZVEC Healthcare Intelligence Demo              ║"
echo "  ║   uv · Python 3.12 · ONNX · Rust libs           ║"
echo "  ╚══════════════════════════════════════════════════╝"
echo ""

# Verify uv is available
if ! command -v uv &>/dev/null; then
    echo "  ✗ uv not found. Install with:"
    echo "    curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "  uv version: $(uv --version)"
echo "  Platform:   $(uname -sm)"
echo ""

# Run with explicit Python 3.12 (zvec requirement)
uv run --python 3.12 "$SCRIPT_DIR/zvec_healthcare_demo.py"
