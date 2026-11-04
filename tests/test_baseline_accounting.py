"""Regression checks for original-byte denominators and complete weighted totals."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

BASELINE_RUNNER = (
    Path(__file__).resolve().parents[1]
    / "baseline/src/benchmark_compression_methods.py"
)
pytestmark = pytest.mark.skipif(
    not BASELINE_RUNNER.is_file(),
    reason="Optional local baseline directory is not distributed",
)


@pytest.fixture
def baseline():
    spec = importlib.util.spec_from_file_location(
        "baseline_accounting", BASELINE_RUNNER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("method", ["xor_zstd", "zstd"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_original_word16_nan_payloads_and_real_denominator(baseline, method, dtype):
    # Every possible 16-bit pattern, including signed zero and signaling NaNs.
    bits = np.arange(65536, dtype=np.uint16)
    target = torch.from_numpy(bits.view(np.int16).copy()).view(dtype)
    reference = torch.roll(target.view(torch.int16), 7).view(dtype)
    row = baseline.benchmark_method(method, reference, target, 1)
    assert row["is_correct"] and row["error"] is None
    assert row["original_bytes"] == row["orig_bytes"] == 2 * len(bits)
    assert row["codec_input_bytes"] == 2 * len(bits)
    assert row["adapter"] == "original_words"
    assert row["ratio"] == row["compressed_bytes"] / (2 * len(bits))


def test_external_fp32_adapter_is_reversible_and_charges_actual_payload(
    baseline, monkeypatch
):
    # Exercise the external API boundary, without pretending to test FM-Delta.
    monkeypatch.setitem(
        baseline.COMPRESSION_METHODS,
        "fp32_contract",
        {
            "name": "FP32 interface contract",
            "requires_base": True,
            "encode": lambda reference, target: target.tobytes(),
            "decode": lambda payload, reference: np.frombuffer(payload, np.float32),
        },
    )
    bits = np.arange(65536, dtype=np.uint16)
    target = torch.from_numpy(bits.view(np.int16).copy()).view(torch.bfloat16)
    row = baseline.benchmark_method("fp32_contract", target, target, 1)
    assert row["is_correct"]
    assert row["adapter"] == "word16_in_fp32"
    assert row["original_bytes"] == 2 * len(bits)
    assert row["codec_input_bytes"] == row["compressed_bytes"] == 4 * len(bits)
    assert row["ratio"] == 2.0


def test_small_and_metadata_tensors_are_counted_without_alias_duplicates(baseline):
    small = torch.tensor([1.0], dtype=torch.float32)
    step = torch.tensor(10, dtype=torch.int64)
    state = {"small": small, "step": step, "alias": small, "empty": torch.empty(0)}
    tensors = list(baseline.traverse_with_path(state))
    assert len(tensors) == 3
    rows = [baseline.benchmark_method("zstd", None, value, 1) for _, value in tensors]
    assert all(row["is_correct"] for row in rows)
    assert rows[1]["storage_mode"] == "Raw"
    assert rows[1]["original_bytes"] == rows[1]["compressed_bytes"] == 8
    summary = baseline.summarize_method(rows)
    assert summary["original_bytes"] == 12
    assert summary["ratio"] == sum(row["compressed_bytes"] for row in rows) / 12


def test_overall_ratio_and_throughput_use_sums_not_per_tensor_averages(baseline):
    rows = [
        dict(
            original_bytes=2,
            compressed_bytes=20,
            is_correct=True,
            error=None,
            enc_time=0.01,
            dec_time=0.02,
        ),
        dict(
            original_bytes=2000,
            compressed_bytes=200,
            is_correct=True,
            error=None,
            enc_time=0.03,
            dec_time=0.04,
        ),
    ]
    summary = baseline.summarize_method(rows)
    assert summary["ratio"] == pytest.approx(220 / 2002)
    assert summary["ratio"] != pytest.approx((10 + 0.1) / 2)
    assert summary["enc_mbps"] == pytest.approx(2002 / 0.04 / 1e6)


def test_failed_tensor_invalidates_overall_result_without_dropping_denominator(
    baseline,
):
    target = torch.arange(128, dtype=torch.float32)
    valid = baseline.benchmark_method("xor_zstd", target, target, 1)
    missing = baseline.benchmark_method("xor_zstd", None, target, 1)
    summary = baseline.summarize_method([valid, missing])
    assert summary["original_bytes"] == target.nbytes * 2
    assert not summary["is_complete"]
    assert summary["compressed_bytes"] is None and summary["ratio"] is None


def test_reference_free_decoder_does_not_need_a_base(baseline):
    row = baseline.benchmark_method(
        "zstd", None, torch.arange(128, dtype=torch.float64), 1
    )
    assert row["is_correct"] and row["original_bytes"] == 128 * 8
