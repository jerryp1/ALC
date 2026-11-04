import hashlib
import math

import numpy as np
import pytest

import tensor_compress as codec


@pytest.mark.parametrize("path", codec.PATHS)
def test_fp32_all_word_patterns_round_trip(path):
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2**32, 1025, dtype=np.uint32)
    bits[:8] = [
        0,
        0x80000000,
        0x7F800000,
        0xFF800000,
        0x7FC00001,
        0x7F800001,
        0xFFFFFFFF,
        1,
    ]
    target, reference = bits.view(np.float32), np.roll(bits, 7).view(np.float32).copy()
    frame = codec.encode_frame(target, reference, path)
    assert np.array_equal(codec.decode_frame(frame, reference).view(np.uint32), bits)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
def test_word_maps_are_exact_inverses(dtype):
    width = np.dtype(dtype).itemsize
    unsigned = np.dtype("u" + str(width))
    values = np.array(
        [0, 1, 1 << (width * 8 - 1), (1 << (width * 8)) - 1], dtype=unsigned
    )
    array = values.view(dtype)
    assert np.array_equal(codec.inverse_rotmap(codec.rotmap(array)), values)
    assert np.array_equal(codec.inverse_pcmap(codec.pcmap(array)), values)


def test_rotdelta_paper_sign_flip_example():
    left = np.array([0xBDF5C291], np.uint32).view(np.float32)
    right = np.array([0x3FA28F5E], np.uint32).view(np.float32)
    rot = abs(int(codec.rotmap(right)[0]) - int(codec.rotmap(left)[0]))
    pc = abs(int(codec.pcmap(right)[0]) - int(codec.pcmap(left)[0]))
    assert rot == 0x03599999 and pc == 0x7D9851F0
    assert rot.bit_length() == 26 and pc.bit_length() == 31


@pytest.mark.parametrize("dtype", [np.uint16, np.float64])
@pytest.mark.parametrize("path", codec.PATHS)
def test_bf16_and_fp64_word_extremes(dtype, path):
    width = np.dtype(dtype).itemsize
    unsigned = np.dtype("u" + str(width))
    rng = np.random.default_rng(5)
    bits = rng.integers(0, np.iinfo(unsigned).max, 257, dtype=unsigned)
    bits[:4] = [0, 1, 1 << (width * 8 - 1), (1 << (width * 8)) - 1]
    target, reference = bits.view(dtype), np.roll(bits, 13).view(dtype).copy()
    result = codec.decode_frame(codec.encode_frame(target, reference, path), reference)
    assert np.array_equal(result.view(unsigned), bits)


def test_bf16_direct_preserves_every_possible_word():
    values = np.arange(65536, dtype=np.uint16)
    assert np.array_equal(
        codec.decode_frame(codec.encode_frame(values, None, "Direct")), values
    )


def test_joint_symbol_widths_do_not_round_uint64_maximum():
    reference = np.array([0, 0, 2**64 - 1], np.uint64)
    target = np.array([0, 2**64 - 1, 0], np.uint64)
    symbols, _, widths = codec.joint_symbols(reference, target)
    assert list(symbols) == [64, 128, 0]
    assert list(widths) == [0, 63, 63]


def test_deterministic_shannon_model_and_no_trial_compression(monkeypatch):
    monkeypatch.setattr(
        codec,
        "require_native",
        lambda: (_ for _ in ()).throw(AssertionError("selector tried compression")),
    )
    reference = np.random.default_rng(7).standard_normal(4096).astype(np.float32)
    target = (reference + np.float32(0.0003)).astype(np.float32)
    result = codec.choose_path(target, reference)
    assert result == codec.choose_path(target, reference)
    assert result["path"] == "FM-Delta*"
    assert (
        result["sample_indices_sha256"]
        == "cffb915f5e9c3163d9e2e383bb3cfcc6a59f40bace78af3da817d1c0aa87d97b"
    )
    assert result["estimated_bits"] == pytest.approx(
        {
            "Direct": 26.402345965353252,
            "FM-Delta*": 15.026562029876272,
            "RotDelta": 17.019447444549513,
        }
    )
    large = np.zeros(70000, dtype=np.float32)
    assert codec.choose_path(large, large)["sample_elements"] == 65536


def test_missing_reference_and_sampling_bounds():
    values = np.arange(16, dtype=np.float32)
    for reference in [None, np.zeros(15, np.float32), np.zeros(16, np.float64)]:
        selected = codec.choose_path(values, reference)
        assert selected["path"] == "Direct"
        assert math.isinf(selected["estimated_bits"]["FM-Delta*"])
    for size in [0, 65537]:
        with pytest.raises(ValueError):
            codec.choose_path(values, values, size)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("Direct", "cc2f244d4a06975c5810178355786a57d2fe9b3faaa71b9e0874b05ffc1b3d56"),
        (
            "FM-Delta*",
            "de15dcf86e05344e2569de1910a1dd336268d53e1478fd663cacea28a8cf4b27",
        ),
        (
            "RotDelta",
            "5d7f4d88d4f77417bce0e59fc9d8c18578b3c0be36cc95485598065c8a041fa9",
        ),
    ],
)
def test_native_frames_match_original_implementation(path, expected):
    # Synthetic golden hashes from implementation snapshot 586c2dd; no model data.
    reference = np.random.default_rng(7).standard_normal(4096).astype(np.float32)
    target = (reference + np.float32(0.0003)).astype(np.float32)
    assert (
        hashlib.sha256(codec.encode_frame(target, reference, path)).hexdigest()
        == expected
    )


def test_native_byte_tans_saturated_alphabet():
    native = codec.require_native()
    raw = bytes(range(256)) * 3
    assert native.tans_decode(native.tans_encode(raw, 8)) == raw


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.float64])
def test_truncated_frame_rejected(dtype):
    values = np.arange(16, dtype=dtype)
    frame = codec.encode_frame(values, values, "Direct")
    for data in [frame[:3], frame[:-1], b"bad!" + frame[4:]]:
        with pytest.raises((ValueError, RuntimeError)):
            codec.decode_frame(data, values)
