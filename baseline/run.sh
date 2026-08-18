#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export PYTHONPATH="${SCRIPT_DIR}/lib:${SCRIPT_DIR}/../src${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
    code="${1:-1}"
    cat <<EOF
Usage:
  ./run.sh methods --base base.pt --ft target.pt [--output results.csv]

Options:
  --base PATH        Base checkpoint .pt path
  --ft PATH          Fine-tuned checkpoint .pt path
  --output FILE      CSV output path [default: benchmark_compression_methods.csv]
  --methods LIST     Comma-separated methods, e.g. fmdelta,zipnn,zstd
  --tensor-idx N..   Tensor indices to test
  --iterations N     Iterations per method [default: 3]
EOF
    exit "$code"
}

[[ $# -lt 1 ]] && usage 1
MODE="$1"; shift

case "$MODE" in
    methods)
        exec "$PYTHON_BIN" "$SCRIPT_DIR/src/benchmark_compression_methods.py" "$@"
        ;;
    help|--help|-h)
        usage 0
        ;;
    *)
        echo "Unknown mode: $MODE"
        usage 1
        ;;
esac
