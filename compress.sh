#!/bin/bash
# Compress PyTorch checkpoints with ALC.
PYTHON="${PYTHON:-$(command -v python3)}"
exec "${PYTHON}" "$(dirname "$0")/compress_checkpoint.py" "$@"
