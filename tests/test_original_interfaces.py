"""Original tensor and shell entry points must use the current paper mechanisms."""

import csv
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import equal, states

import tensor_compress as tc
from checkpoint_archive import Archive

BASELINE_DIR = Path(__file__).resolve().parents[1] / "baseline"


@pytest.mark.parametrize("path", [tc.PATH_IALIGN, tc.PATH_PCDELTA, tc.PATH_ROTDELTA])
def test_original_tensor_api_has_independent_chunks_and_exact_bits(path):
    bits = np.random.default_rng(17).integers(0, 2**32, 517, dtype=np.uint32)
    target = bits.view(np.float32)
    reference = np.roll(bits, 2).view(np.float32).copy()
    strategy = tc.CompressStrategy(path)
    size, chunks, _ = tc.compress_tensor(target, strategy, reference, chunk_elems=128)
    assert len(chunks) == 5 and size == sum(len(frame) for frame, _ in chunks)
    assert [info["n"] for _, info in chunks] == [128, 128, 128, 128, 5]
    recovered, _ = tc.decompress_tensor(chunks, strategy, len(target), reference)
    assert recovered.tobytes() == target.tobytes()


def test_original_strategy_api_uses_paper_costs_and_fixed_sample_cap():
    reference = np.random.default_rng(0).standard_normal(100000).astype(np.float32)
    target = reference + np.float32(1e-4)
    choice = tc.choose_path(target, reference)
    strategy = tc.analyze_strategy(target, reference)
    assert strategy.chosen_entropy == min(choice["estimated_bits"].values())
    assert choice["sample_elements"] == 65536


def test_original_shell_single_and_nested_directory_interfaces(tmp_path):
    root = Path(__file__).resolve().parents[1]
    target, base = states(count=2)
    inputs = tmp_path / "input/sub"
    bases = tmp_path / "base/sub"
    inputs.mkdir(parents=True)
    bases.mkdir(parents=True)
    torch.save(target, inputs / "state.pt")
    torch.save(base, bases / "state.pt")

    def run(script, *args):
        subprocess.run(
            ["bash", str(root / script), *map(str, args)],
            env={**os.environ, "PYTHON": sys.executable},
            check=True,
            capture_output=True,
            text=True,
            timeout=60,
        )

    single = tmp_path / "single"
    run(
        "compress.sh",
        "--input",
        inputs / "state.pt",
        "--base",
        bases / "state.pt",
        "--output",
        single,
        "--chunk_elements",
        128,
    )
    assert Archive(single).manifest["latest"] == 1
    output = tmp_path / "single-restored.pt"
    run(
        "decompress.sh",
        "--input",
        single,
        "--metadata",
        single / "manifest.json",
        "--output",
        output,
    )
    equal(target, torch.load(output, weights_only=False))
    batch = tmp_path / "batch"
    run(
        "compress.sh",
        "--input_dir",
        inputs.parent,
        "--base_dir",
        bases.parent,
        "--output_dir",
        batch,
    )
    restored = tmp_path / "restored"
    run("decompress.sh", "--input_dir", batch, "--output_dir", restored)
    equal(target, torch.load(restored / "sub/restored_state.pt", weights_only=False))
    direct = tmp_path / "direct"
    run("compress.sh", "--input", inputs / "state.pt", "--output", direct)
    assert not Archive(direct).manifest["latest_raw"]
    equal(target, Archive(direct).restore(0))


@pytest.mark.skipif(
    not (BASELINE_DIR / "src/benchmark_compression_methods.py").is_file(),
    reason="Optional local baseline directory is not distributed",
)
def test_baseline_xor_zstd_preserves_original_fp32_bits_and_missing_fmdelta_is_explicit():
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "baseline_methods", root / "baseline/src/benchmark_compression_methods.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bits = np.random.default_rng(16).integers(0, 2**32, 257, dtype=np.uint32)
    target = bits.view(np.float32)
    reference = np.roll(bits, 7).view(np.float32).copy()
    for name in ["xor_zstd", "zstd"]:
        method = module.COMPRESSION_METHODS[name]
        encoded = method["encode"](reference, target)
        assert method["decode"](encoded, reference).tobytes() == target.tobytes()
    if module.fmd is None:
        with pytest.raises(RuntimeError, match="external fmdelta_encodings"):
            module.method_fmdelta_encode(reference, target)


@pytest.mark.skipif(
    not (BASELINE_DIR / "run.sh").is_file(),
    reason="Optional local baseline directory is not distributed",
)
def test_original_baseline_shell_and_csv_interface(tmp_path):
    root = Path(__file__).resolve().parents[1]
    base = tmp_path / "base.pt"
    target = tmp_path / "target.pt"
    x = torch.linspace(-0.5, 0.5, 257)
    older = {"weight": x, "bf16": x.to(torch.bfloat16), "step": torch.tensor(1)}
    newer = {
        "weight": x + 1e-4,
        "bf16": (x + 1e-4).to(torch.bfloat16),
        "step": torch.tensor(2),
    }
    torch.save(older, base)
    torch.save(newer, target)
    output = tmp_path / "results/output.csv"
    subprocess.run(
        [
            "bash",
            str(root / "baseline/run.sh"),
            "methods",
            "--base",
            str(base),
            "--ft",
            str(target),
            "--methods",
            "xor_zstd,zstd",
            "--iterations",
            "1",
            "--output",
            str(output),
        ],
        env={**os.environ, "PYTHON_BIN": sys.executable},
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["method"] for row in rows} == {"xor_zstd", "zstd"}
    assert all(row["is_correct"] == "True" for row in rows)
    summary = json.loads(Path(str(output) + ".summary.json").read_text())
    expected_bytes = 257 * 4 + 257 * 2 + 8
    for method in ("xor_zstd", "zstd"):
        result = summary["methods"][method]
        selected = [row for row in rows if row["method"] == method]
        assert result["is_complete"] and result["original_bytes"] == expected_bytes
        assert (
            result["ratio"]
            == sum(int(row["compressed_bytes"]) for row in selected) / expected_bytes
        )


def test_original_tans_module_interface():
    from tans_codec import EncodedPayload, tans_decode, tans_encode

    data = bytes(range(256)) + b"aabbccc" * 51
    encoded = tans_encode(data)
    assert encoded.table_log == 8
    assert tans_decode(EncodedPayload.from_bytes(encoded.to_bytes())) == data


def test_standalone_no_base_checkpoint_is_direct_and_preserves_verified_bf16(tmp_path):
    from checkpoint_archive import compress

    state = states(count=1)[0]
    archive = compress([state], tmp_path / "single", retain_latest_raw=False)
    assert not (archive.path / "latest.pt").exists()
    modes = {s["mode"] for s in archive.version(0)["tensors"].values()}
    assert "Raw" not in modes and "Direct" in modes and "DerivedBF16" in modes
    equal(state, archive.restore(0))
