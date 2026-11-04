#!/bin/bash
# Restore checkpoints compressed by ALC.
PYTHON="${PYTHON:-$(command -v python3)}"
exec "${PYTHON}" "$(dirname "$0")/decompress_checkpoint.py" "$@"
