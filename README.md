# ALC: Adaptive Lossless Compression for LLM Checkpoints

ALC combines verified BF16 deduplication, per-tensor selection among Direct, FM-Delta* and RotDelta, and a shared tANS backend. Historical checkpoints use reverse-delta chains with optional grouping and pipelined recovery/transmission.

## Requirements

- Python >= 3.10
- NumPy >= 1.24
- PyTorch >= 2.0
- pybind11 >= 2.10 and a C++20 compiler, for building the extension
- pkg-config and OpenSSL development headers

```bash
pip install -r requirements.txt
```

## Build C++ extension

Run from the repository root. Installation automatically compiles `src/cpp/serving_runtime.cpp` (including the tensor codec) into `cpp.alc_tans_cpp_codec` for the active platform.

```bash
python -m pip install -e .
```

Optional tests:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
```

# Usage

- Compress and restore

```bash
# One checkpoint; omit --base for independent Direct compression.
./compress.sh --input target.pt --base newer.pt --output_dir compressed
./decompress.sh --input compressed/compressed_target.pt --output_dir restored

# Version history, ordered oldest to newest.
python examples/synthetic_checkpoints.py --output outputs/example
./compress.sh --checkpoints outputs/example/checkpoint-*.pt --output outputs/archive
./decompress.sh --input outputs/archive --version 1 --output outputs/restored.pt
```

Outputs are **archive directories** containing metadata and references. Version histories default to no grouping, with the newest checkpoint kept Raw. Add `--group_size 4` for grouped serving.

Use `--input_dir` / `--base_dir` for directory batches and `--bf16_pairs mapping.json` for explicit master mappings. Run `./compress.sh --help` or `./decompress.sh --help` for all options.

- Serving

```bash
# Server and client (separate terminals).
python serve_checkpoint.py serve outputs/archive --port 9000
python serve_checkpoint.py fetch --port 9000 --version 1 --output outputs/received.pt

# Local paced query with compressed data preloaded before timing.
python serve_checkpoint.py query outputs/archive --version 1 --resident
```
