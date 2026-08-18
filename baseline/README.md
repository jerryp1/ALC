# Baseline Compression Methods

Clean baseline benchmarks for LLM checkpoint compression.

## Layout

```text
baseline/
├── README.md
├── run.sh
└── src/
    └── benchmark_compression_methods.py
```

## Usage

```bash
./run.sh methods --base base.pt --ft target.pt --output results.csv
```

Optional:

```bash
./run.sh methods --base base.pt --ft target.pt --methods fmdelta,zipnn,zstd
./run.sh methods --base base.pt --ft target.pt --tensor-idx 0 1 2 --iterations 3
```

## Methods

| Method | Reference | Description |
|--------|-----------|-------------|
| `fmdelta` | FM-Delta | PCMap + range coding |
| `xor_zstd` | ZipLLM | XOR delta + Zstandard |
| `zipnn` | ZipNN | Exponent-bit extraction + Huffman/ZSTD |
| `rangecode` | Range coding | Static range coding on raw float32 bytes |
| `zstd` | Zstd | Raw Zstandard |
