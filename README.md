# ALC: an adaptive lossless compression framework for LLM checkpoints.

Compress and restore PyTorch checkpoint files with adaptive lossless tensor coding.

## Requirements

- Python >= 3.9
- NumPy >= 1.24
- PyTorch >= 2.0
- pybind11 and C++20 compiler, only for rebuilding the extension

```bash
pip install -r requirements.txt
```

## Build C++ extension

```bash
cd src/cpp
c++ -O3 -march=native -shared -std=c++20 -fPIC \
    $(python -m pybind11 --includes) \
    alc_tans_cpp_codec.cpp \
    -o alc_tans_cpp_codec$(python3-config --extension-suffix)
```

## Usage

```bash
# Compress one file
./compress.sh --input target.pt --base base.pt --output_dir compressed

# Compress all checkpoint files in a directory
./compress.sh --input_dir target_dir --base_dir base_dir --output_dir compressed

# Restore one compressed file
./decompress.sh --input compressed/compressed_target.pt \
                --metadata compressed/compressed_target.pt.json \
                --base base.pt \
                --output_dir restored

# Restore all compressed checkpoint files in a directory
./decompress.sh --input_dir compressed --base_dir base_dir --output_dir restored
```

## Algorithm

| Path | Transform |
|------|-----------|
| `pcdelta` | `abs(pcmap(ft) - pcmap(base))` -> `(s,k)` symbols + mantissas |
| `rotdelta` | `abs(rotl1(ft) - rotl1(base))` -> `(s,k)` symbols + mantissas |
| `I-Align` | `rotl1(ft)` -> 4 byte columns |
